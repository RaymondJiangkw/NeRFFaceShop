import torch
from torch import nn
from torch.nn import functional as F

import torchfields
import numpy as np

from training.networks_stylegan2 import FullyConnectedLayer, MappingNetwork, modulated_conv2d
from training.volumetric_rendering.renderer import ImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler

import dnnlib
from torch_utils.ops import upfirdn2d, bias_act
from torch_utils import misc, persistence

from einops import rearrange

#----------------------------------------------------------------------------

def batch_inverse_fields(fields: torch.Tensor):
    BT, C, H, W = fields.shape[:-3], *fields.shape[-3:]
    fields = fields.reshape(-1, C, H, W).field_()
    return torch.cat([fields[_:_+1].inverse() for _ in range(fields.shape[0])]).tensor_().reshape(*BT, C, H, W)

#----------------------------------------------------------------------------

@persistence.persistent_class
class MipmapWarp(torch.nn.Module):
    """
    Module for applying spatial transforms with mipmap anti-aliasing.
    Code from Tim Brooks.
    """

    def __init__(self, max_num_levels: int = 3.5):
        """Initializes MipmapWarp class.
        Args:
          max_num_levels (int, optional): Max number of mipmap levels (default 3.5).
        """
        super().__init__()
        self.max_num_levels = max_num_levels
        self._register_blur_filter()
        self.levels_map = None

    def forward(  # pylint: disable=arguments-differ
            self, inputs: torch.Tensor, grid: torch.Tensor,
            min_level: float = 0.0, padding_mode: str = 'border', mode: str = 'bilinear', align_corners=False) -> torch.Tensor:
        """Applies spatial transform with antialiasing; analogous to grid_sample().
        Args:
          inputs (torch.Tensor): Input features on which to apply transform.
          grid (torch.Tensor): Sampling grid normalized to [-1, 1].
        Returns:
            torch.Tensor: Transformed features.
        """
        assert mode == 'bilinear'
        assert align_corners == False
        # Determines level in mipmap stack to sample at each pixel.
        _, _, height, width = inputs.size()
        coords = self._get_coordinates(grid, height, width)
        levels = self._get_mipmap_levels(coords, self.max_num_levels)
        levels = levels.clamp(min=min_level)

        # Computes total number of levels needed in stack for this sampling.
        num_levels = int(levels.max().ceil().item()) + 1

        # Creates a stack of Gaussian filtered features and warps each level.
        stack = self._create_stack(inputs, num_levels)
        stack = self._warp_stack(stack, grid, padding_mode=padding_mode)

        outputs = self._sample_mipmap(stack, levels)
        self.levels_map = levels / (self.max_num_levels - 1.0)
        return outputs

    @staticmethod
    def get_max_coord_distance(coords: torch.Tensor) -> torch.Tensor:
        """Computes max distance of neighboring coordinates.
        Args:
          coords (torch.Tensor): Coordinates of shape [N, H, W, 2].
        Returns:
          torch.Tensor: Maximum distances of shape [N, H, W].
        """
        # pylint: disable=too-many-locals

        # Pads coordinates.
        coords_padded = coords.permute(0, 3, 1, 2)
        coords_padded = nn.ReplicationPad2d(1)(coords_padded)
        coords_padded = coords_padded.permute(0, 2, 3, 1)
        # Gets neighboring coordinates on four sides of each sample.
        coords_l = coords_padded[:, 1:-1, :-2, :]
        coords_r = coords_padded[:, 1:-1, 2:, :]
        coords_u = coords_padded[:, :-2, 1:-1, :]
        coords_d = coords_padded[:, 2:, 1:-1, :]

        # Computes distance between coordinates and each neighbor.
        def _get_dist(other_coords):
            sq_dist = torch.sum((other_coords - coords) ** 2, dim=3)
            # Clamps at 1 to prevent numerical instability of square root. Does not
            # introduce bias since log2(1) = 0, which is the lowest mipmap level.
            return sq_dist.clamp(min=1.0) ** 0.5

        dist_l = _get_dist(coords_l)
        dist_r = _get_dist(coords_r)
        dist_u = _get_dist(coords_u)
        dist_d = _get_dist(coords_d)
        dists = torch.stack([dist_l, dist_r, dist_u, dist_d])

        # Determines stack level from maximum distance at each sample.
        dist_max, _ = torch.max(dists, dim=0)
        return dist_max

    ##############################################################################
    # Private Instance Methods
    ##############################################################################

    def _register_blur_filter(self):
        """Registers a Gaussian blurring filter to the module."""
        blur_filter = np.array([1., 3., 3., 1.])
        blur_filter = torch.Tensor(blur_filter[:, None] * blur_filter[None, :])
        blur_filter = blur_filter / torch.sum(blur_filter)
        blur_filter = blur_filter[None, None, ...]
        self.register_buffer('blur_filter', blur_filter)

    def _downsample_2x(self, inputs: torch.Tensor) -> torch.Tensor:
        """Gaussian blurs inputs along spatial dimensions."""
        num_channels = inputs.shape[1]
        blur_filter = self.blur_filter.repeat((num_channels, 1, 1, 1))
        inputs = nn.ReflectionPad2d(1)(inputs)
        outputs = F.conv2d(inputs, blur_filter, stride=2, groups=num_channels)
        return outputs

    def _create_stack(self, inputs: torch.Tensor,
                      num_levels: int) -> torch.Tensor:
        """Creates a Gaussian stack; blurs each level, but does not downsample.
        Args:
          inputs (torch.Tensor): Input features of shape [N, C, H, W].
          num_levels (int): Number of levels to create in stack.
        Returns:
          torch.Tensor: Gaussian stack of shape [N, C, D, H, W], were dimension
              D represents stack level.
        """
        # _, _, height, width = inputs.size()
        log_size = np.log2(inputs.size(-1))
        pad_needed = not log_size.is_integer()
        if pad_needed:
            target_size = 2 ** np.ceil(log_size)
            total_pad = target_size - inputs.size(-1)
            left_pad = int(total_pad // 2)
            right_pad = int(total_pad - left_pad)
            inputs = F.pad(inputs, pad=(left_pad, right_pad, left_pad, right_pad), mode='reflect')
        levels = [inputs]

        for i in range(1, num_levels):
            inputs = self._downsample_2x(inputs)
            scale_factor = 2.0 ** i
            level = self._upsample(inputs, scale_factor)
            levels.append(level)

        stack = torch.stack(levels, dim=2)
        if pad_needed:
            stack = stack[:, :, :, left_pad:-right_pad, left_pad:-right_pad]
        return stack

    ##############################################################################
    # Private Static Methods
    ##############################################################################

    @staticmethod
    def _upsample(inputs: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """Bilinear upsampling."""
        outputs = F.interpolate(
            inputs, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        return outputs

    @staticmethod
    def _warp_stack(stack: torch.Tensor, grid: torch.Tensor,
                    padding_mode='border') -> torch.Tensor:
        """Applies F.grid_sample() to each level of a stack.
        Args:
          stack (torch.Tensor): Stack of shape [N, C, D, H_in, W_in].
          grid (torch.Tensor): Sampling grid of shape [N, H_out, W_out, 2].
          padding_mode: 'zeros' or 'border' or 'reflection'.
        Returns:
          torch.Tensor: Warped stack of shape [N, C, D, H_out, W_out].
        """
        N, C, D, H_in, W_in = stack.size()  # pylint: disable=invalid-name
        _, H_out, W_out, _ = grid.size()  # pylint: disable=invalid-name

        stack = stack.reshape((N, C * D, H_in, W_in))
        stack = F.grid_sample(stack, grid, padding_mode=padding_mode, align_corners=False)
        stack = stack.reshape((N, C, D, H_out, W_out))
        return stack

    @staticmethod
    def _get_coordinates(grid: torch.Tensor, height: int,
                         width: int) -> torch.Tensor:
        """Converts a normalized grid in [-1, 1] to absolute coordinates.
        Args:
          grid (torch.Tensor): Sampling grid of shape [N, H, W, 2].
          height (int): Height of the source being sampled.
          width (int): Width of the source being sampled.
        Returns:
          torch.Tensor: Coordinates of shape [N, H, W, 2].
        """
        x_coord = (width - 1.0) * (grid[..., 0] + 1.0) / 2.0
        y_coord = (height - 1.0) * (grid[..., 1] + 1.0) / 2.0
        coords = torch.stack([x_coord, y_coord], dim=3)
        return coords

    @staticmethod
    def _get_mipmap_levels(coords: torch.Tensor,
                           max_num_levels: int) -> torch.Tensor:
        """Computes level in mipmap to sample at each pixel based on coordinates.
        Args:
          coords (torch.Tensor): Coordinates of shape [N, H, W, 2].
          max_num_levels (int): Max number of levels allowed in mipmap.
        Returns:
          torch.Tensor: Mipmap levels of shape [N, H, W].
        """
        dist_max = MipmapWarp.get_max_coord_distance(coords)
        levels = torch.log2(dist_max)
        levels = levels.clamp(min=0.0, max=max_num_levels - 1.0)
        return levels

    @staticmethod
    def _sample_mipmap(stack: torch.Tensor,
                       levels: torch.Tensor) -> torch.Tensor:
        """Linearly samples mipmap stack at levels.
        Args:
          stack (torch.Tensor): Gaussian stack of shape [N, C, D, H, W].
          levels (torch.Tensor): Mipmap levels of shape [N, H, W].
        Returns:
          torch.Tensor: Output samples of shape [N, C, H, W].
        """
        # Adds channel dim of size C and level dim of size 1.
        C = stack.shape[1]  # pylint: disable=invalid-name
        levels = torch.stack([levels] * C, dim=1)
        levels = levels[:, :, None, :, :]

        # Gets two levels to interpolate between at each pixel.
        level_0 = levels.floor().long()
        level_1 = levels.ceil().long()
        level_dim = 2
        output_0 = torch.gather(stack, level_dim, level_0)
        output_1 = torch.gather(stack, level_dim, level_1)

        # Linearly interpolates between levels.
        weight = levels % 1.0
        output = output_0 + weight * (output_1 - output_0)
        output = output[:, :, 0, :, :]
        return output

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
        deform_trainable        = True,         
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
        self.num_dconv = 0
        self.num_dtorgb = 0
        self.dimg_channels = 2 if deformable else None
        
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
        
        if self.deformable:
            # self.template_w = torch.nn.Parameter(torch.zeros([self.w_dim]))
            if in_channels != 0:
                self.dconv0 = TriplaneSynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2 if not disable_upsample else 1, resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, trainable=trainable and deform_trainable, **layer_kwargs)
                self.num_dconv += 1
            
            self.dconv1 = TriplaneSynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution, conv_clamp=conv_clamp, channels_last=self.channels_last, trainable=trainable and deform_trainable, **layer_kwargs)
            self.num_dconv += 1
            
            self.dtorgb = TriplaneToRGBLayer(out_channels, self.dimg_channels, w_dim=w_dim, conv_clamp=conv_clamp, channels_last=self.channels_last, init_scale=init_scale, disable_bias=disable_bias, trainable=trainable and deform_trainable)
            self.num_dtorgb += 1

    def forward(self, x, img, imgd, ws, wds, force_fp32=False, fused_modconv=None, update_emas=False, inplace_add=True, **layer_kwargs):
        _ = update_emas # unused
        misc.assert_shape(ws, [None, self.num_conv + self.num_torgb, self.w_dim])
        # if not self.deformable:
        w_iter = iter(ws.unbind(dim=1))
        # else:
        #     w_iter = iter([self.template_w[None, :].expand(ws.size(0), -1)] * (self.num_conv + self.num_torgb))
        if wds is not None and self.num_dconv + self.num_dtorgb > 0:
            misc.assert_shape(wds, [None, self.num_dconv + self.num_dtorgb, self.w_dim])
            wd_iter = iter(wds.unbind(dim=1))
        
        if ws.device.type != 'cuda' and wds.device.dtype != 'cuda':
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
            # Residual
            if wds is not None and self.residual:
                x = x + self.rconv(x, next(wd_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            
            if wds is not None and self.deformable:
                xd = self.dconv1(xd, next(wd_iter), fused_modconv=fused_modconv, **layer_kwargs)
        else:
            x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            # Residual
            if wds is not None and self.residual:
                x = x + self.rconv(x, next(wd_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            
            # Deformation.
            if wds is not None and self.deformable:
                xd = self.dconv0(xd, next(wd_iter), fused_modconv=fused_modconv, **layer_kwargs)
                xd = self.dconv1(xd, next(wd_iter), fused_modconv=fused_modconv, **layer_kwargs)
        
        # ToRGB.
        if img is not None:
            if not self.disable_upsample:
                misc.assert_shape(img, [None, 3, self.img_channels, self.resolution // 2, self.resolution // 2])
                img = img.view(-1, 3 * self.img_channels, self.resolution // 2, self.resolution // 2)
                img = upfirdn2d.upsample2d(img, self.resample_filter)
                img = img.view(-1, 3, self.img_channels, self.resolution, self.resolution)
            else:
                misc.assert_shape(img, [None, 3, self.img_channels, self.resolution, self.resolution])
        if imgd is not None and self.deformable:
            if not self.disable_upsample:
                misc.assert_shape(imgd, [None, 3, self.dimg_channels, self.resolution // 2, self.resolution // 2])
                imgd = imgd.view(-1, 3 * self.dimg_channels, self.resolution // 2, self.resolution // 2)
                imgd = upfirdn2d.upsample2d(imgd, self.resample_filter)
                imgd = imgd.view(-1, 3, self.dimg_channels, self.resolution, self.resolution)
            else:
                misc.assert_shape(imgd, [None, 3, self.dimg_channels, self.resolution, self.resolution])
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = (img.add_(y) if inplace_add else (img + y)) if img is not None else y
            
            if wds is not None and self.deformable:
                yd = self.dtorgb(xd, next(wd_iter), fused_modconv=fused_modconv)
                yd = yd.to(dtype=torch.float32, memory_format=torch.contiguous_format)
                imgd = (imgd.add_(yd) if inplace_add else (imgd + yd)) if imgd is not None else yd

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        assert imgd is None or imgd.dtype == torch.float32
        return x, img, imgd

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'

#----------------------------------------------------------------------------

@persistence.persistent_class
class TriplaneSynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                                          # Intermediate latent (W) dimensionality.
        img_resolution,                                 # Output image resolution.
        img_channels,                                   # Number of color channels.
        channel_base        = 16384,                    # Overall multiplier for the number of channels.
        channel_max         = 256,                      # Maximum number of channels in any layer.
        num_deformable_res  = 4,                        # Deformable features for the N lowest resolutions.
        deform_type         = 'template+mapping',       # Deformation type.
        num_fp16_res        = 4,                        # Use FP16 for the N highest resolutions.
        **block_kwargs,                                 # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        assert deform_type in ['instance-specific', 'template+mapping', 'template-only', 'residual', 'none']
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.deform_type = deform_type
        self.num_deformable_res = num_deformable_res
        self.num_fp16_res = num_fp16_res
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        self.channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)
        self.deformable_resolution = min(2 ** (num_deformable_res + 1), 2 ** self.img_resolution_log2)
        
        self.num_ws = 0
        self.num_wds = 0
        for res in self.block_resolutions:
            in_channels = self.channels_dict[res // 2] if res > 4 else 0
            out_channels = self.channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            deformable = (res <= self.deformable_resolution) and deform_type != 'none' and deform_type != 'template-only' and deform_type != 'residual'
            residual = (res <= self.deformable_resolution) and deform_type == 'residual'
            is_last = (res == self.img_resolution)
            block = TriplaneSynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, deformable=deformable, residual=residual, **block_kwargs)
            self.num_ws += block.num_conv
            self.num_wds += block.num_dconv
            if is_last:
                self.num_ws += block.num_torgb
            if (deformable or residual) and res == self.deformable_resolution:
                self.num_wds += block.num_dtorgb
            setattr(self, f'b{res}', block)
        
        self.register_buffer("identity_grid", F.affine_grid(torch.eye(2, 3)[None, ...], (1, 2, self.deformable_resolution, self.deformable_resolution), align_corners=False))
        self.antialias_grid_sample = MipmapWarp(3.5)
    
    def _deform(self, input: torch.Tensor, grid: torch.Tensor, antialias=False, permute=False) -> torch.Tensor:
        grid_sample = self.antialias_grid_sample if antialias else F.grid_sample
        if len(input.shape) == 5:
            if not permute:
                B, _, C, R, R = input.shape
                return grid_sample(input.view(B * 3, C, R, R), grid.view(B * 3, R, R, 2), mode='bilinear', align_corners=False, padding_mode='zeros').view(B, 3, C, R, R)
            else:
                B, _, R, R, C = input.shape
                return grid_sample(input.view(B * 3, R, R, C).permute(0, 3, 1, 2), grid.view(B * 3, R, R, 2), mode='bilinear', align_corners=False, padding_mode='zeros').view(B, 3, C, R, R).permute(0, 1, 3, 4, 2)
        else:
            if not permute:
                return grid_sample(input, grid, mode='bilinear', align_corners=False, padding_mode='zeros')
            else:
                return grid_sample(input.permute(0, 3, 1, 2), grid, mode='bilinear', align_corners=False, padding_mode='zeros').permute(0, 2, 3, 1)
    
    def forward(self, ws, wds, template_exp_mapping=None, deform_type='default', tar_shape_mapping=None, antialias=False, only_frontal=True, use_exp_mask=False, exp_mask=None, **block_kwargs):
        assert deform_type in ['default', 'disabled', 'template-only', 'instance-transfer']
        block_ws = []
        block_wds = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            if wds is not None and self.num_wds > 0:
                misc.assert_shape(wds, [None, self.num_wds, self.w_dim])
                wds = wds.to(torch.float32)
                wd_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv
                if wds is not None and self.num_wds > 0:
                    block_wds.append(wds.narrow(1, wd_idx, block.num_dconv + block.num_dtorgb))
                    wd_idx += block.num_dconv
                else:
                    block_wds.append(None)

        x = img = imgd = deformation_planes_delta = None
        for res, cur_ws, cur_wds in zip(self.block_resolutions, block_ws, block_wds):
            block = getattr(self, f'b{res}')
            x, img, imgd = block(x, img, imgd, cur_ws, cur_wds, **block_kwargs)
            
            if self.deform_type != 'none' and self.deform_type != 'residual':
                deformation_maps = None
                if deform_type == 'default':
                    if self.deform_type == 'instance-specific' and res == self.deformable_resolution and imgd is not None:
                        if use_exp_mask == True and exp_mask is not None:
                            imgd = imgd * exp_mask[:, :, None, :, :]
                        deformation_maps = imgd.permute(0, 1, 3, 4, 2) + self.identity_grid[:, None, ...]
                        deformation_planes_delta = deformation_maps - self.identity_grid[:, None, ...]
                    elif self.deform_type == 'template+mapping' and res == self.deformable_resolution and template_exp_mapping is not None:
                        exp_template_deformation_maps = template_exp_mapping.permute(0, 1, 3, 4, 2) + self.identity_grid[:, None, ...] # (B, 3, H, W, 2)
                        
                        template_to_instance = rearrange(batch_inverse_fields(imgd if tar_shape_mapping is None else tar_shape_mapping), 'b three two h w -> b three h w two', three=3) + self.identity_grid[:, None, ...]
                        instance_to_template = (imgd if tar_shape_mapping is None else tar_shape_mapping).permute(0, 1, 3, 4, 2) + self.identity_grid[:, None, ...]
                        
                        deformation_maps = self._deform(self._deform(template_to_instance, exp_template_deformation_maps, antialias=antialias, permute=True), instance_to_template, antialias=antialias, permute=True) # exp_instance_deformation_maps
                        deformation_planes_delta = deformation_maps - self.identity_grid[:, None, ...]
                    elif self.deform_type == 'template-only' and res == self.deformable_resolution and template_exp_mapping is not None:
                        deformation_maps = template_exp_mapping.permute(0, 1, 3, 4, 2) + self.identity_grid[:, None, ...]
                        deformation_planes_delta = deformation_maps - self.identity_grid[:, None, ...]
                elif deform_type == 'template-only':
                    assert self.deform_type == 'template+mapping', self.deform_type
                    if res == self.deformable_resolution:
                        exp_template_deformation_maps = (template_exp_mapping.permute(0, 1, 3, 4, 2) if template_exp_mapping is not None else 0) + self.identity_grid[:, None, ...].repeat(ws.size(0), 3, 1, 1, 1) # (B, 3, H, W, 2)
                        
                        template_to_instance = rearrange(batch_inverse_fields(imgd if tar_shape_mapping is None else tar_shape_mapping), 'b three two h w -> b three h w two', three=3) + self.identity_grid[:, None, ...]
                        
                        deformation_maps = self._deform(template_to_instance, exp_template_deformation_maps, antialias=antialias, permute=True)
                        deformation_planes_delta = deformation_maps - self.identity_grid[:, None, ...]
                
                if deformation_maps is not None:
                    if not only_frontal:
                        x = self._deform(x, deformation_maps, antialias=antialias)
                        img = self._deform(img, deformation_maps, antialias=antialias)
                    else:
                        x = torch.stack((self._deform(x[:, 0], deformation_maps[:, 0], antialias=antialias), x[:, 1], x[:, 2]), dim=1)
                        img = torch.stack((self._deform(img[:, 0], deformation_maps[:, 0], antialias=antialias), img[:, 1], img[:, 2]), dim=1)
                
        return x, img, imgd, deformation_planes_delta

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d}, num_wds={self.num_wds:d}, deform_type={self.deform_type}',
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_res={self.num_fp16_res:d}, num_deformable_res={self.num_deformable_res:d}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class TriplaneSynthesisLighting(torch.nn.Module):
    def __init__(self,
        w_dim,                                          # Intermediate latent (W) dimensionality.
        img_resolution,                                 # Output image resolution.
        img_channels,                                   # Number of color channels.
        channels,                                       # Fixed Number of channels.
        num_blocks,                                     # Number of synthesis blocks.
        num_fp16_blk        = 0,                        # Use FP16 for the N highest blocks.
        **block_kwargs,                                 # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.channels = channels
        self.num_blocks = num_blocks
        self.num_fp16_blk = num_fp16_blk
        
        self.num_ws = 0
        for num_block in range(num_blocks):
            use_fp16 = (num_block >= num_blocks - num_fp16_blk)
            is_last = (num_block == num_blocks - 1)
            block = TriplaneSynthesisBlock(channels, channels, w_dim=w_dim, resolution=img_resolution,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, disable_upsample=True, deformable=False, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{num_block}', block)
    
    def forward(self, x, img, ws, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for num_block in range(self.num_blocks):
                block = getattr(self, f'b{num_block}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv
        misc.assert_shape(x, [None, 3, self.channels, self.img_resolution, self.img_resolution])
        misc.assert_shape(img, [None, 3, self.img_channels, self.img_resolution, self.img_resolution])
        for num_block, cur_ws in zip(range(self.num_blocks), block_ws):
            block = getattr(self, f'b{num_block}')
            x, img, _ = block(x, img, None, cur_ws, None, inplace_add=False, **block_kwargs)
        return img

    def extra_repr(self):
        return ' '.join([
            f'w_dim={self.w_dim:d}, num_ws={self.num_ws:d},', 
            f'img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d},',
            f'num_fp16_blk={self.num_fp16_blk:d}'])

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                                          # Input latent (Z) dimensionality.
        c_dim,                                          # Conditioning label (C) dimensionality.
        w_dim,                                          # Intermediate latent (W) dimensionality.
        d_dim,                                          # Input deformation latent (D) dimensionality.
        l_dim,                                          # Conditioning lighting (L) dimensionality.
        img_resolution,                                 # Output resolution.
        img_channels,                                   # Number of output color channels.
        mapping_kwargs      = {},                       # Arguments for MappingNetwork.
        lmapping_kwargs     = {},                       # Arguments for MappingNetwork of Lighting.
        synthesis_kwargs    = {},                       # Arguments for SynthesisNetwork.
        lsynthesis_kwargs   = {},                       # Arguments for SynthesisLighting.
        lock_deformation    = False, 
    ):
        deform_type = synthesis_kwargs['deform_type']
        assert deform_type in ['instance-specific', 'template+mapping', 'template-only', 'residual']
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.l_dim = l_dim
        self.d_dim = d_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.deform_type = deform_type
        self.lock_deformation = lock_deformation
        
        if deform_type == 'instance-specific':
            self.synthesis = TriplaneSynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, init_scale=1e-4, deform_trainable=not lock_deformation, **synthesis_kwargs)
            self.num_ws = self.synthesis.num_ws
            self.num_wds = self.synthesis.num_wds
            self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
            self.dmapping = MappingNetwork(z_dim=d_dim, c_dim=0, w_dim=d_dim, num_ws=self.num_wds, trainable=not lock_deformation, **mapping_kwargs)
        elif deform_type == 'template+mapping' or deform_type == 'template-only':
            self.synthesis = TriplaneSynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, init_scale=1e-4, disable_bias=True, **synthesis_kwargs)
            self.num_ws = self.synthesis.num_ws
            self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
            
            self.dsynthesis = TriplaneSynthesisNetwork(w_dim=d_dim, img_resolution=2 ** (synthesis_kwargs['num_deformable_res'] + 1), img_channels=2, init_scale=1e-4, disable_bias=True, deform_type='none', num_deformable_res=0, trainable=not lock_deformation, **{k: v for k, v in synthesis_kwargs.items() if not k in ['num_deformable_res', 'deform_type']})
            self.num_wds = self.dsynthesis.num_ws
            self.dmapping = MappingNetwork(z_dim=d_dim, c_dim=0, w_dim=d_dim, num_ws=self.num_wds, trainable=not lock_deformation, **mapping_kwargs)
        elif deform_type == 'residual':
            self.synthesis = TriplaneSynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, deform_trainable=not lock_deformation, **synthesis_kwargs)
            self.num_ws = self.synthesis.num_ws
            self.num_wds = self.synthesis.num_wds
            self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
            self.dmapping = MappingNetwork(z_dim=d_dim, c_dim=0, w_dim=d_dim, num_ws=self.num_wds, trainable=not lock_deformation, **mapping_kwargs)
        
        if lock_deformation:
            # Transform the distribution of expression deformation from the domain of video dataset into image dataset
            self.dmapping_t = MappingNetwork(z_dim=d_dim, c_dim=0, w_dim=d_dim, num_ws=1, **mapping_kwargs)
        
        # Lighting Modules
        self.lsynthesis = TriplaneSynthesisLighting(w_dim=16, img_resolution=img_resolution, img_channels=img_channels, channels=self.synthesis.channels_dict[img_resolution], **lsynthesis_kwargs)
        self.num_wls = self.lsynthesis.num_ws
        self.lmapping = MappingNetwork(z_dim=0, c_dim=l_dim, w_dim=16, num_ws=self.num_wls, **lmapping_kwargs)
    def apply_d(self, d, update_emas=False):
        if d is None: return None
        return self.dmapping_t(d, None, update_emas=update_emas).squeeze(1)
    def forward(self, z, c, d, l, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        wds = self.dmapping(d, None, update_emas=update_emas)
        
        if self.deform_type == 'instance-specific':
            x, img, imgd, _ = self.synthesis(ws, wds, update_emas=update_emas, **synthesis_kwargs)
        elif self.deform_type == 'template+mapping' or self.deform_type == 'template-only':
            _, template_exp_mapping, _ = self.dsynthesis(wds, None, update_emas=update_emas, **synthesis_kwargs)
            x, img, imgd, _ = self.synthesis(ws, ws[:, :self.synthesis.num_wds, :], (template_exp_mapping) if template_exp_mapping is not None else None, update_emas=update_emas, **synthesis_kwargs)
        elif self.deform_type == 'residual':
            x, img, imgd, _ = self.synthesis(ws, wds, update_emas=update_emas, **synthesis_kwargs)
        
        wls = self.lmapping(l, None, update_emas=update_emas)
        imgl = self.lsynthesis(x, img, wls, **synthesis_kwargs)
        
        return img, imgd, imgl

#----------------------------------------------------------------------------

@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        d_dim,                      # Input deformation latent (D) dimensionality.
        l_dim,                      # Conditioning lighting (L) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        lmapping_kwargs     = {},   # Arguments for MappingNetwork of Lighting.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
        lsynthesis_kwargs   = {},   # Arguments for SynthesisLighting.
        rendering_kwargs    = {},
        sr_kwargs           = {},
        lock_deformation    = False, 
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.d_dim=d_dim
        self.l_dim=l_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.deform_type = synthesis_kwargs['deform_type']
        self.renderer = LitImportanceRenderer()
        self.ray_sampler = RaySampler()
        self.backbone = Generator(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, d_dim=d_dim, l_dim=l_dim, img_resolution=256, img_channels=32, mapping_kwargs=mapping_kwargs, lmapping_kwargs=lmapping_kwargs, synthesis_kwargs=synthesis_kwargs, lsynthesis_kwargs=lsynthesis_kwargs, lock_deformation=lock_deformation)
        self.superresolution = dnnlib.util.construct_class_by_name(class_name=rendering_kwargs['superresolution_module'], channels=32, img_resolution=img_resolution, sr_antialias=rendering_kwargs['sr_antialias'], **sr_kwargs)
        
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.shading_decoder = ShadingDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 1, 'alpha': 0.1})
        
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
    
        self._last_planes = None
    
    def mapping(self, z, c, d, l, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if c is not None and self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return \
            self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas) if z is not None and c is not None else None, \
            self.backbone.dmapping(d, None, update_emas=update_emas) if d is not None else None, \
            self.backbone.lmapping(None, l, update_emas=update_emas) if l is not None else None

    def synthesis(self, ws, wds, wls, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, use_exp_mask=False, tar_template_exp_mapping=None, overrided_lplanes=None, return_planes=False, skip_sr=False, portrait_only=False, skip_dm=True, projection='perspective', only_frontal=True, **synthesis_kwargs):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:].view(-1, int((c.shape[1]-16)**0.5), int((c.shape[1]-16)**0.5))
        
        if len(ws.shape) == 2: ws = ws[:, None, :].expand(-1, self.backbone.num_ws, -1)
        if wds is not None and len(wds.shape) == 2: wds = wds[:, None, :].expand(-1, self.backbone.num_wds, -1)
        if len(wls.shape) == 2: wls = wls[:, None, :].expand(-1, self.backbone.num_wls, -1)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(projection, cam2world_matrix, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        deformation_planes_delta = None
        if use_cached_backbone and self._last_planes is not None:
            planes, lplanes = self._last_planes
            return_dict = {}
        else:
            if self.deform_type == 'instance-specific':
                x, planes, deformation_maps, deformation_planes_delta = self.backbone.synthesis(ws, wds, update_emas=update_emas, use_exp_mask=use_exp_mask, exp_mask=self.exp_mask, **synthesis_kwargs)
                return_dict = {'deformation_maps': deformation_maps, 'deformation_planes_delta': deformation_planes_delta}
            elif self.deform_type == 'template+mapping' or self.deform_type == 'template-only':
                if tar_template_exp_mapping is not None:
                    template_exp_mapping = tar_template_exp_mapping
                elif wds is not None:
                    _, template_exp_mapping, _, _ = self.backbone.dsynthesis(wds, None, update_emas=update_emas, **synthesis_kwargs)
                    if only_frontal:
                        template_exp_mapping = torch.stack([
                            template_exp_mapping[:, 0], 
                            torch.zeros_like(template_exp_mapping[:, 1]), 
                            torch.zeros_like(template_exp_mapping[:, 2]), 
                        ], dim=1)
                    if use_exp_mask == True and getattr(self, 'exp_mask', None) is not None:
                        template_exp_mapping = template_exp_mapping * self.exp_mask[:, :, None, :, :]
                else:
                    template_exp_mapping = None
                x, planes, shape_mapping, deformation_planes_delta = self.backbone.synthesis(ws, ws[:, :self.backbone.synthesis.num_wds, :], (template_exp_mapping) if template_exp_mapping is not None else None, update_emas=update_emas, only_frontal=only_frontal, **synthesis_kwargs)
                return_dict = {'template_exp_mapping': template_exp_mapping, 'shape_mapping': shape_mapping, 'deformation_planes_delta': deformation_planes_delta}
            elif self.deform_type == 'residual':
                x, planes, deformation_maps, deformation_planes_delta = self.backbone.synthesis(ws, wds, update_emas=update_emas, **synthesis_kwargs)
                return_dict = {}
            if overrided_lplanes is None:
                lplanes = self.backbone.lsynthesis(x, planes, wls, update_emas=update_emas, **synthesis_kwargs)
            else:
                lplanes = overrided_lplanes
        if cache_backbone:
            self._last_planes = planes, lplanes
        
        # Reshape output into three 32-channel planes
        misc.assert_shape(planes, [None, 3, 32, 256, 256])
        misc.assert_shape(lplanes, [None, 3, 32, 256, 256])
        if return_planes:
            return {'planes': planes, 'lplanes': lplanes}

        # Perform volume rendering
        feature_samples, depth_samples, albedo_samples, shading_samples, deform_samples, weights_samples = self.renderer(planes, lplanes, deformation_planes_delta.permute(0, 1, 4, 2, 3) if not skip_dm and not portrait_only and deformation_planes_delta is not None else None, self.decoder, self.shading_decoder, ray_origins, ray_directions, self.rendering_kwargs | {'portrait_only': portrait_only}) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        albedo_image = albedo_samples.permute(0, 2, 1).reshape(N, albedo_samples.shape[-1], H, W) if albedo_samples is not None else None
        shading_image = shading_samples.permute(0, 2, 1).reshape(N, shading_samples.shape[-1], H, W) if shading_samples is not None else None
        deform_image = deform_samples.permute(0, 2, 1).reshape(N, deform_samples.shape[-1], H, W) if deform_samples is not None else None
        
        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if not k in ['noise_mode', 'deform_type', 'tar_shape_mapping', 'antialias']}) if not skip_sr else None
        
        albedo_image_raw = albedo_image[:, :3] if albedo_image is not None else None
        albedo_image_sr = self.superresolution(albedo_image_raw, albedo_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if not k in ['noise_mode', 'deform_type', 'tar_shape_mapping', 'antialias']}) if not skip_sr and albedo_image_raw is not None else None
        
        return {'image': sr_image, 'image_feature': feature_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'image_albedo': albedo_image_sr, 'image_albedo_raw': albedo_image_raw, 'image_shading': shading_image, 'image_deforming': deform_image} | return_dict

    def upsample(self, feature_image, ws, **synthesis_kwargs):
        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if not k in ['noise_mode']})
        return sr_image

    def sample_mixed(self, coordinates, directions, ws, update_emas=False, **synthesis_kwargs):
        if self.deform_type == 'instance-specific':
            _, planes, _, _ = self.backbone.synthesis(ws, None, update_emas=update_emas, **synthesis_kwargs)
        elif self.deform_type == 'template+mapping' or self.deform_type == 'template-only':
            _, planes, _, _ = self.backbone.synthesis(ws, ws[:, :self.backbone.synthesis.num_wds, :], update_emas=update_emas, **synthesis_kwargs)
        
        return self.renderer.run_model(planes, None, self.decoder, None, coordinates, directions, self.rendering_kwargs)
    
    def sample_mixed_with_wds(self, coordinates, directions, ws, wds, update_emas=False, only_frontal=True, use_exp_mask=True, **synthesis_kwargs):
        if self.deform_type == 'instance-specific':
            _, planes, _, _ = self.backbone.synthesis(ws, wds, update_emas=update_emas, **synthesis_kwargs)
        elif self.deform_type == 'template+mapping' or self.deform_type == 'template-only':
            _, template_exp_mapping, _, _ = self.backbone.dsynthesis(wds, None, update_emas=update_emas, **synthesis_kwargs)
            if only_frontal:
                template_exp_mapping = torch.stack([
                    template_exp_mapping[:, 0], 
                    torch.zeros_like(template_exp_mapping[:, 1]), 
                    torch.zeros_like(template_exp_mapping[:, 2]), 
                ], dim=1)
            if use_exp_mask == True and getattr(self, 'exp_mask', None) is not None:
                template_exp_mapping = template_exp_mapping * self.exp_mask[:, :, None, :, :]
            _, planes, _, _ = self.backbone.synthesis(ws, ws[:, :self.backbone.synthesis.num_wds, :], template_exp_mapping, only_frontal=only_frontal, update_emas=update_emas, **synthesis_kwargs)
        
        return self.renderer.run_model(planes, None, self.decoder, None, coordinates, directions, self.rendering_kwargs)
    
    def forward(self, z, c, d, l, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        ws, wds, wls = self.mapping(z, c, d, l, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, wds, wls, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)

#----------------------------------------------------------------------------

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
    
#----------------------------------------------------------------------------

class ShadingDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features * 3, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'], activation='softexp', alpha=options['alpha'])
        )
        
    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = rearrange(sampled_features, 'b three m c -> b m (three c)')
        x = sampled_features

        N, M, C = x.shape
        x = x.reshape(N*M, C)
        x = self.net(x)
        x = x.reshape(N, M, -1)
        
        return {'shading': x}

#----------------------------------------------------------------------------

from training.volumetric_rendering import math_utils
from training.volumetric_rendering.ray_marcher import MipRayMarcher2
from training.volumetric_rendering.renderer import ImportanceRenderer, sample_from_planes

class LitImportanceRenderer(ImportanceRenderer):
    def __init__(self):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
    def run_model(
        self, 
        planes, 
        lit_planes, 
        decoder, 
        decoder_shading, 
        sample_coordinates, 
        sample_directions, 
        options, 
    ):
        sampled_features = sample_from_planes(
            self.plane_axes, 
            planes, 
            sample_coordinates, 
            padding_mode='zeros', 
            box_warp=options['box_warp']
        )
        
        out = decoder(sampled_features, sample_directions)
        
        if options.get('density_noise', 0) > 0:
            out['sigma'] += torch.randn_like(out['sigma']) * options['density_noise']

        if lit_planes is not None and decoder_shading is not None:
            sampled_shading_features = sample_from_planes(
                self.plane_axes, 
                lit_planes, 
                sample_coordinates, 
                padding_mode='zeros', 
                box_warp=options['box_warp']
            )
            
            # Merge `shading` information
            out |= decoder_shading(sampled_shading_features, sample_directions)
        return out
    def unify_samples(self, 
        depths1, colors1, densities1, albedo1, shading1, deform1, 
        depths2, colors2, densities2, albedo2, shading2, deform2, 
        rendering_options):
        all_depths = torch.cat([depths1, depths2], dim = -2)
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        
        all_shading = None
        if shading1 is not None and shading2 is not None:
            all_shading = torch.cat([shading1, shading2], dim = -2)
            all_shading = torch.gather(all_shading, -2, indices.expand(-1, -1, -1, all_shading.shape[-1]))
        
        all_densities = torch.cat([densities1, densities2], dim = -2)
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        
        all_colors = torch.cat([colors1, colors2], dim = -2)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        
        all_albedo = None
        if albedo1 is not None and albedo2 is not None:
            all_albedo = torch.cat([albedo1, albedo2], dim = -2)
            all_albedo = torch.gather(all_albedo, -2, indices.expand(-1, -1, -1, all_albedo.shape[-1]))
        
        all_deform = None
        if deform1 is not None and deform2 is not None:
            all_deform = torch.cat([deform1, deform2], dim = -2)
            all_deform = torch.gather(all_deform, -2, indices.expand(-1, -1, -1, all_deform.shape[-1]))

        return all_depths, all_colors, all_densities, all_albedo, all_shading, all_deform
    def forward(
        self, 
        planes, 
        lit_planes, 
        deform_planes, 
        decoder, 
        decoder_shading, 
        ray_origins, 
        ray_directions, 
        rendering_options, 
    ):
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
            ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape
        
        # Quick Path for Portrait Only
        if rendering_options.get('portrait_only', False):
            # Coarse Pass
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)
            
            out = self.run_model(planes, lit_planes, decoder, decoder_shading, sample_coordinates, sample_directions, rendering_options)
            colors_coarse = (out['rgb'] * out['shading']).reshape(batch_size, num_rays, samples_per_ray, -1)
            densities_coarse = out['sigma'].reshape(batch_size, num_rays, samples_per_ray, 1)
            
            # Fine Pass
            N_importance = rendering_options['depth_resolution_importance']
            assert N_importance > 0
            
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)
            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)
            
            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            out = self.run_model(planes, lit_planes, decoder, decoder_shading, sample_coordinates, sample_directions, rendering_options)
            colors_fine = (out['rgb'] * out['shading']).reshape(batch_size, num_rays, N_importance, -1)
            densities_fine = out['sigma'].reshape(batch_size, num_rays, N_importance, 1)
            
            all_depths, all_colors, all_densities, _, _, _ = self.unify_samples(
                depths_coarse, colors_coarse, densities_coarse, None, None, None, 
                depths_fine, colors_fine, densities_fine, None, None, None, 
                rendering_options = rendering_options, 
            )
            
            # Aggregate
            rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
            return rgb_final, depth_final, None, None, None, weights.sum(2)

        # Coarse Pass
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)
        
        out = self.run_model(planes, lit_planes, decoder, decoder_shading, sample_coordinates, sample_directions, rendering_options)
        deform_coarse = rearrange(sample_from_planes(
            self.plane_axes, 
            deform_planes, 
            sample_coordinates, 
            padding_mode='zeros', 
            box_warp=rendering_options['box_warp']
        ), 'b three m c -> b m (three c)').reshape(batch_size, num_rays, samples_per_ray, -1) if deform_planes is not None else None
        shading_coarse = out['shading']
        densities_coarse = out['sigma']

        # Apply Shading
        albedo_coarse = out['rgb']
        colors_coarse = albedo_coarse * shading_coarse
        albedo_coarse = albedo_coarse.reshape(batch_size, num_rays, samples_per_ray, albedo_coarse.shape[-1])
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        
        shading_coarse = shading_coarse.reshape(batch_size, num_rays, samples_per_ray, shading_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        
        # Fine Pass
        N_importance = rendering_options['depth_resolution_importance']
        assert N_importance > 0
        
        _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

        depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
        sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

        out = self.run_model(planes, lit_planes, decoder, decoder_shading, sample_coordinates, sample_directions, rendering_options)
        deform_fine = rearrange(sample_from_planes(
            self.plane_axes, 
            deform_planes, 
            sample_coordinates, 
            padding_mode='zeros', 
            box_warp=rendering_options['box_warp']
        ), 'b three m c -> b m (three c)').reshape(batch_size, num_rays, samples_per_ray, -1) if deform_planes is not None else None
        shading_fine = out['shading']
        densities_fine = out['sigma']
        
        # Apply Shading
        albedo_fine = out['rgb']
        colors_fine = albedo_fine * shading_fine
        albedo_fine = albedo_fine.reshape(batch_size, num_rays, N_importance, albedo_fine.shape[-1])
        colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
        
        shading_fine = shading_fine.reshape(batch_size, num_rays, N_importance, shading_fine.shape[-1])
        densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)
        
        all_depths, all_colors, all_densities, all_albedos, all_shadings, all_deforms = self.unify_samples(
            depths_coarse, colors_coarse, densities_coarse, albedo_coarse, shading_coarse, deform_coarse, 
            depths_fine, colors_fine, densities_fine, albedo_fine, shading_fine, deform_fine, 
            rendering_options = rendering_options, 
        )

        # Aggregate
        rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
        
        albedo_final = torch.sum(weights * ((all_albedos[:, :, :-1] + all_albedos[:, :, 1:]) / 2), dim=-2) * 2 - 1 # Scale to (-1, 1)
        shading_final = torch.sum(weights * ((all_shadings[:, :, :-1] + all_shadings[:, :, 1:]) / 2), dim=-2) # (0, 1)
        deform_final = torch.sum(weights * ((all_deforms[:, :, :-1] + all_deforms[:, :, 1:]) / 2), dim=-2) if all_deforms is not None else None
        
        return rgb_final, depth_final, albedo_final, shading_final, deform_final, weights.sum(2)

#----------------------------------------------------------------------------