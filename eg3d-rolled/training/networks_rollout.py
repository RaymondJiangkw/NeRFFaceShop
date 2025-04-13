import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from training.networks_stylegan2 import FullyConnectedLayer, MappingNetwork, modulated_conv2d
from training.volumetric_rendering.renderer import ImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler

import dnnlib
from torch_utils.ops import upfirdn2d, bias_act
from torch_utils import misc, persistence

from einops import rearrange

#----------------------------------------------------------------------------

@persistence.persistent_class
class TriplaneSynthesisLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels.
        out_channels,                           # Number of output channels.
        w_dim,                                  # Intermediate latent (W) dimensionality.
        resolution,                             # Resolution of this layer.
        kernel_size             = 3,            # Convolution kernel size.
        up                      = 1,            # Integer upsampling factor.
        use_noise               = True,         # Enable noise input?
        activation              = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter         = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp              = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last           = False,        # Use channels_last format for the weights?
        init_scale              = 1.,           # Initialized scale.
        trainable               = True,         
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        
        self.affine = FullyConnectedLayer(w_dim, in_channels * 3, bias_init=1, trainable=trainable)
        if trainable:
            self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels * 3, kernel_size, kernel_size]).to(memory_format=memory_format) * init_scale)
        else:
            self.register_buffer('weight', torch.randn([out_channels, in_channels * 3, kernel_size, kernel_size]).to(memory_format=memory_format) * init_scale)
        if trainable:
            self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        else:
            self.register_buffer('bias', torch.zeros([out_channels]))
        if use_noise:
            # Noise is applied after the convolution.
            self.register_buffer('noise_const', torch.randn([1, 1, resolution, resolution]))
            if trainable:
                self.noise_strength = torch.nn.Parameter(torch.zeros([]))
            else:
                self.register_buffer('noise_strength', torch.zeros([]))
    def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1, **unused_kwargs):
        assert noise_mode in ['random', 'const', 'none', 'random_store', 'random_retrieve']
        in_resolution = self.resolution // self.up
        misc.assert_shape(x, [None, 3, self.in_channels, in_resolution, in_resolution])
        styles = self.affine(w)
        
        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn([x.shape[0], 1, 1, self.resolution, self.resolution], device=x.device).repeat([1, 3, 1, 1, 1]) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const[None, ...].repeat([x.shape[0], 3, 1, 1, 1]) * self.noise_strength
        if self.use_noise and noise_mode == 'random_store':
            self._noise = torch.randn([x.shape[0], 1, 1, self.resolution, self.resolution], device=x.device).repeat([1, 3, 1, 1, 1])
            noise = self._noise * self.noise_strength
        if self.use_noise and noise_mode == 'random_retrieve':
            assert getattr(self, '_noise', None) is not None
            assert self._noise.shape[0] == x.shape[0]
            noise = self._noise * self.noise_strength
            # del self._noise
        
        xy, yz, zx = x.unbind(dim=1)
        noise_xy, noise_yz, noise_zx = noise.unbind(dim=1)
        
        # Conduct axis-wise pooling (average)
        x_ = xy.mean(dim=-2, keepdim=True).expand_as(xy)
        _y = xy.mean(dim=-1, keepdim=True).expand_as(xy)
        
        y_ = yz.mean(dim=-2, keepdim=True).expand_as(yz)
        _z = yz.mean(dim=-1, keepdim=True).expand_as(yz)
        
        z_ = zx.mean(dim=-2, keepdim=True).expand_as(zx)
        _x = zx.mean(dim=-1, keepdim=True).expand_as(zx)
        
        # Concatenate the features
        xy = torch.cat([xy, _x.transpose(-1, -2), y_.transpose(-1, -2)], dim=1)
        yz = torch.cat([yz, _y.transpose(-1, -2), z_.transpose(-1, -2)], dim=1)
        zx = torch.cat([zx, _z.transpose(-1, -2), x_.transpose(-1, -2)], dim=1)
        
        # Concatenate along the batch axis for speeding up
        x = torch.cat([xy, yz, zx], dim=0)
        noise = torch.cat([noise_xy, noise_yz, noise_zx], dim=0)
        
        flip_weight = (self.up == 1) # slightly faster
        x = modulated_conv2d(x=x, noise=noise, weight=self.weight, styles=styles.repeat([3, 1]), up=self.up, padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)
        
        # Rearrange
        xy, yz, zx = torch.chunk(x, 3, dim=0)
        return torch.stack([xy, yz, zx], dim=1)

    def extra_repr(self):
        return ' '.join([
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d},',
            f'resolution={self.resolution:d}, up={self.up}, activation={self.activation:s}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class TriplaneToRGBLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False, init_scale=1, disable_bias=False, trainable=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1, trainable=trainable)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        if trainable:
            self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format) * init_scale)
        else:
            self.register_buffer('weight', torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format) * init_scale)
        if not disable_bias:
            if trainable:
                self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
            else:
                self.register_buffer('bias', torch.zeros([out_channels]))
        else:
            self.bias = None
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

    def forward(self, x, w, fused_modconv=True, **unused_kwargs):
        styles = self.affine(w) * self.weight_gain
        xy, yz, zx = x.unbind(dim=1)
        
        # Concatenate along the batch axis for speeding up
        x = torch.cat([xy, yz, zx], dim=0)
        
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles.repeat([3, 1]), demodulate=False, fused_modconv=fused_modconv)
        if self.bias is not None:
            x = bias_act.bias_act(x, self.bias.to(x.dtype), clamp=self.conv_clamp)
        
        # Rearrange
        xy, yz, zx = torch.chunk(x, 3, dim=0)
        return torch.stack([xy, yz, zx], dim=1)

    def extra_repr(self):
        return f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}, w_dim={self.w_dim:d}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class TriplaneSynthesisBlock(torch.nn.Module):
    def __init__(self,
        in_channels,                            # Number of input channels, 0 = first block.
        out_channels,                           # Number of output channels.
        w_dim,                                  # Intermediate latent (W) dimensionality.
        resolution,                             # Resolution of this block.
        img_channels,                           # Number of output color channels.
        is_last,                                # Is this the last block?
        architecture            = 'skip',       # Architecture: 'orig', 'skip'.
        resample_filter         = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp              = 256,          # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16                = False,        # Use FP16 for this block?
        fp16_channels_last      = False,        # Use channels-last memory format with FP16?
        fused_modconv_default   = True,         # Default value of fused_modconv. 'inference_only' = True for inference, False for training.
        init_scale              = 1.,           # Initialized scale for ToRGB Layer.
        disable_bias            = False,        # Disable bias of ToRGB Layer.
        deformable              = False,        # Whether the features of this block is deformable.
        residual                = False,        # Whether there are residual features of this block.
        disable_upsample        = False,        # Whether to disable the upsampling.
        trainable               = True,               
        **layer_kwargs,                         # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.deformable = deformable
        self.residual = residual
        self.disable_upsample = disable_upsample
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0
        
        if in_channels == 0:
            if trainable:
                self.const = torch.nn.Parameter(torch.randn([3, out_channels, resolution, resolution]))
            else:
                self.register_buffer("const", torch.randn([3, out_channels, resolution, resolution]))
        
        if in_channels != 0:
            self.conv0 = TriplaneSynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2 if not disable_upsample else 1,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, trainable=trainable, **layer_kwargs)
            self.num_conv += 1
        
        self.conv1 = TriplaneSynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, trainable=trainable, **layer_kwargs)
        self.num_conv += 1
        
        if is_last or architecture == 'skip':
            self.torgb = TriplaneToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last, init_scale=init_scale if not self.deformable else 1., disable_bias=disable_bias if not self.deformable else False, trainable=trainable)
            self.num_torgb += 1
        
        if self.residual:
            self.rconv = TriplaneSynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, trainable=trainable, **layer_kwargs)
            self.num_dconv += 1

    def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, update_emas=False, inplace_add=True, **layer_kwargs):
        _ = update_emas # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1, 1])
        else:
            if not self.disable_upsample:
                misc.assert_shape(x, [None, 3, self.in_channels, self.resolution // 2, self.resolution // 2])
            else:
                misc.assert_shape(x, [None, 3, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)
        
        xd = x

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        
        # ToRGB.
        if img is not None:
            if not self.disable_upsample:
                misc.assert_shape(img, [None, 3, self.img_channels, self.resolution // 2, self.resolution // 2])
                img = img.view(-1, 3 * self.img_channels, self.resolution // 2, self.resolution // 2)
                img = upfirdn2d.upsample2d(img, self.resample_filter)
                img = img.view(-1, 3, self.img_channels, self.resolution, self.resolution)
            else:
                misc.assert_shape(img, [None, 3, self.img_channels, self.resolution, self.resolution])
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = (img.add_(y) if inplace_add else (img + y)) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class TriplaneSynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 4,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = TriplaneSynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)
        return img

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = TriplaneSynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler()
        self.backbone = Generator(z_dim, c_dim, w_dim, img_resolution=256, img_channels=32, mapping_kwargs=mapping_kwargs, **synthesis_kwargs)
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_num_fp16_res=sr_num_fp16_res, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
    
        self._last_planes = None
    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
                c = torch.zeros_like(c)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes
        
        # Reshape output into three 32-channel planes
        misc.assert_shape(planes, [None, 3, 32, 256, 256])

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image}
    
    def sample(self, coordinates, directions, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(self, coordinates, directions, ws, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas = update_emas, **synthesis_kwargs)
        planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features * 3, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = rearrange(sampled_features, 'b three m c -> b m (three c)')
        x = sampled_features

        N, M, C = x.shape
        x = x.reshape(N*M, C)

        x = self.net(x)
        x = x.reshape(N, M, -1)
        rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}