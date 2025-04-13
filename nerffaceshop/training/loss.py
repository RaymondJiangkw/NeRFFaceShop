# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing

from lpips import LPIPS

from torchmetrics.functional.image import image_gradients

class SparsityCriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: torch.Tensor):
        misc.assert_shape(x, [None, 2, None, None])
        dy, dx = image_gradients(x)
        return torch.linalg.vector_norm(x, dim=1, ord=2).mean() + \
            torch.linalg.vector_norm(dy, dim=1, ord=2).mean() + \
            torch.linalg.vector_norm(dx, dim=1, ord=2).mean()

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0, gpc_reg_fade_kimg=1000, gpc_reg_prob=None, filter_mode='antialiased', deform_type=None):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        assert self.style_mixing_prob == 0
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)
        self.deform_type = deform_type
        self.lpips_loss = LPIPS(net='vgg', verbose=False).to(device)
        self.sparsity_loss = SparsityCriterion()
    
    def run_G(self, z, c, d, l, swapping_prob, neural_rendering_resolution, update_emas=False, injected_cond=None, **kwargs):
        if injected_cond is not None:
            c_gen_conditioning = injected_cond
        else:
            if swapping_prob is not None:
                c_swapped = torch.roll(c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
            else:
                c_gen_conditioning = torch.zeros_like(c)

        ws, wds, wls = self.G.mapping(z, c_gen_conditioning, d, l, update_emas=update_emas)
        # c_new, delta_c = self.G.apply_delta_c(z, c, update_emas=update_emas)
        gen_output = self.G.synthesis(ws, wds, wls, c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas, **kwargs)
        return gen_output | {'ws': ws, 'wds': wds, 'wls': wls, 'c_gen_conditioning': c_gen_conditioning} # , 'delta_c': delta_c}

    def run_D(self, frameU, cU, lU, frameV, cV, lV, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=frameU['image'].device).div(blur_sigma).square().neg().exp2()
                frameU['image'] = upfirdn2d.filter2d(frameU['image'], f / f.sum())
                frameV['image'] = upfirdn2d.filter2d(frameV['image'], f / f.sum())
        
        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([
                frameU['image'], 
                torch.nn.functional.interpolate(frameU['image_raw'], size=frameU['image'].shape[2:], mode='bilinear', antialias=True), 
                frameV['image'], 
                torch.nn.functional.interpolate(frameV['image_raw'], size=frameV['image'].shape[2:], mode='bilinear', antialias=True), 
            ], dim=1))
            _imageU, _image_rawU, _imageV, _image_rawV = torch.split(augmented_pair, 3, dim=1)
            frameU['image'] = _imageU
            frameU['image_raw'] = torch.nn.functional.interpolate(_image_rawU, size=frameU['image_raw'].shape[2:], mode='bilinear', antialias=True)
            frameV['image'] = _imageV
            frameV['image_raw'] = torch.nn.functional.interpolate(_image_rawV, size=frameV['image_raw'].shape[2:], mode='bilinear', antialias=True)
        
        logits = self.D(frameU, cU, lU, frameV, cV, lV, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, 
        phase, 
        # Real data of two frames in the videos.
        real_imgU, real_cU, real_lU, 
        real_imgV, real_cV, real_lV, 
        # Sampled data of two poses and two animation for a single identity.
        gen_z, 
        gen_cU, gen_dU, gen_lU, 
        gen_cV, gen_dV, gen_lV, 
        # Misc.
        gain, cur_nimg
    ):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial
        
        real_img_rawU = filtered_resizing(real_imgU, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)
        real_img_rawV = filtered_resizing(real_imgV, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_rawU.device).div(blur_sigma).square().neg().exp2()
                real_img_rawU = upfirdn2d.filter2d(real_img_rawU, f / f.sum())
                real_img_rawV = upfirdn2d.filter2d(real_img_rawV, f / f.sum())

        real_imgU = {'image': real_imgU, 'image_raw': real_img_rawU}
        real_imgV = {'image': real_imgV, 'image_raw': real_img_rawV}
        
        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_imgU = self.run_G(gen_z, gen_cU, gen_dU, gen_lU, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, noise_mode='random_store', portrait_only=True)
                gen_imgV = self.run_G(gen_z, gen_cV, (gen_dU + gen_dV) / 2 ** 0.5, gen_lV, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, injected_cond=gen_imgU['c_gen_conditioning'], noise_mode='random_retrieve', portrait_only=True)
                gen_logits = torch.cat(self.run_D(gen_imgU, gen_cU, gen_lU, gen_imgV, gen_cV, gen_lV, blur_sigma=blur_sigma))
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)
                
                # Scale Constraint
                # (Optional) Jacobian Smoothness Regularization
                if 'template_exp_mapping' in gen_imgU:
                    misc.assert_shape(gen_imgU['template_exp_mapping'], [None, 3, 2, None, None])
                    loss_Gexp = self.sparsity_loss(gen_imgU['template_exp_mapping'][:, 0])
                    training_stats.report('Loss/G/exp', loss_Gexp)
                else:
                    loss_Gexp = 0.
                if 'shape_mapping' in gen_imgU:
                    misc.assert_shape(gen_imgU['shape_mapping'], [None, 3, 2, None, None])
                    loss_Gshape = self.sparsity_loss(gen_imgU['shape_mapping'][:, 0])
                    training_stats.report('Loss/G/shape', loss_Gshape)
                else:
                    loss_Gshape = 0.
                if 'deformation_maps' in gen_imgU and not 'shape_mapping' in gen_imgU:
                    misc.assert_shape(gen_imgU['deformation_maps'], [None, 3, 2, None, None])
                    loss_Gdeform = self.sparsity_loss(gen_imgU['deformation_maps'][:, 0])
                    training_stats.report('Loss/G/deform', loss_Gdeform)
                else:
                    loss_Gdeform = 0.
            
            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain.mean() + self.G.rendering_kwargs['exp_reg'] * loss_Gexp + self.G.rendering_kwargs['shape_reg'] * loss_Gshape + self.G.rendering_kwargs['exp_reg'] * loss_Gdeform).mul(gain).backward()
        
        # Correspondence & Lighting & Density Regularization
        assert self.G.rendering_kwargs['reg_type'] == 'l1'
        if phase in ['Greg', 'Gboth']:
            cut_size = gen_z.size(0) // 2
            gen_z, gen_c, gen_d, gen_l = gen_z[:cut_size], gen_cU[:cut_size], gen_dU[:cut_size], gen_lU[:cut_size]
            
            gen_c_swapped = torch.roll(gen_c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((gen_c.shape[0], 1), device=gen_c.device) < swapping_prob, gen_c_swapped, gen_c)

            ws, wds, wls = self.G.mapping(gen_z, c_gen_conditioning, gen_d, gen_l)
            _ws, _, _ = self.G.mapping(torch.randn_like(gen_z), c_gen_conditioning, None, None)
            
            # === Lighting Regularization ===
            gen_i0 = self.G.synthesis(ws, wds, wls, gen_c, neural_rendering_resolution=neural_rendering_resolution, noise_mode="random_store")
            
            # Mixing Another Latent Codes
            _ws, _, _ = self.G.mapping(torch.randn_like(gen_z), c_gen_conditioning, None, None)
            _ws[:, :7] = ws[:, :7] # Expect the geometry is similar
            _gen_i0 = self.G.synthesis(_ws, wds, wls, gen_c, return_planes=True, noise_mode="random_retrieve")
            gen_i0_hat = self.G.synthesis(ws, wds, wls, gen_c, overrided_lplanes=_gen_i0['lplanes'], portrait_only=True, noise_mode="random_retrieve") # Replace the Shading Tri-plane
            
            # Same Shading, Different Texture
            loss_Gcross_T = torch.nn.L1Loss()(gen_i0['image'], gen_i0_hat['image'])
            training_stats.report(f'Loss/G/crossTexture', loss_Gcross_T)
            
            # Another Shading Latent Codes
            _gen_l = gen_l + self.G.dataset_sh_std * torch.randn_like(gen_l) * 0.1 # gen_l.std(0)
            _, _, _wls = self.G.mapping(None, None, None, _gen_l)
            gen_i0_s1 = self.G.synthesis(ws, wds, _wls, gen_c, skip_sr=True, noise_mode="random_retrieve")
            
            # Same Texture, Different Shading
            loss_Gcross_S = self.lpips_loss(gen_i0_s1['image_albedo_raw'], ((gen_i0_s1['image_raw']/2+.5)/(gen_i0['image_shading'] + 1e-5))*2-1).mean()
            training_stats.report(f'Loss/G/crossShading', loss_Gcross_S)
            
            ((loss_Gcross_T + loss_Gcross_S) * self.G.rendering_kwargs['lighting_reg']).mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_imgU = self.run_G(gen_z, gen_cU, gen_dU, gen_lU, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, noise_mode='random_store', update_emas=True, portrait_only=True)
                gen_imgV = self.run_G(gen_z, gen_cV, (gen_dU + gen_dV) / 2 ** 0.5, gen_lV, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, injected_cond=gen_imgU['c_gen_conditioning'], noise_mode='random_retrieve', update_emas=True, portrait_only=True)
                
                gen_logits = torch.cat(self.run_D(gen_imgU, gen_cU, gen_lU, gen_imgV, gen_cV, gen_lV, blur_sigma=blur_sigma, update_emas=True))
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_imgU_tmp_image = real_imgU['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_imgU_tmp_image_raw = real_imgU['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_imgU_tmp = {'image': real_imgU_tmp_image, 'image_raw': real_imgU_tmp_image_raw}
                
                real_imgV_tmp_image = real_imgV['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_imgV_tmp_image_raw = real_imgV['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_imgV_tmp = {'image': real_imgV_tmp_image, 'image_raw': real_imgV_tmp_image_raw}

                real_logits = torch.cat(self.run_D(real_imgU_tmp, real_cU, real_lU, real_imgV_tmp, real_cV, real_lV, blur_sigma=blur_sigma))
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_imgU_tmp['image'], real_imgU_tmp['image_raw'], real_imgV_tmp['image'], real_imgV_tmp['image_raw']], create_graph=True, only_inputs=True)
                        r1_grads_imageU = r1_grads[0]
                        r1_grads_imageU_raw = r1_grads[1]
                        r1_grads_imageV = r1_grads[2]
                        r1_grads_imageV_raw = r1_grads[3]
                    r1_penalty = r1_grads_imageU.square().sum([1,2,3]) + r1_grads_imageU_raw.square().sum([1,2,3]) + r1_grads_imageV.square().sum([1,2,3]) + r1_grads_imageV_raw.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)
            
            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
