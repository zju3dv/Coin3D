from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.io import imsave
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import imageio

from ldm.base_utils import read_pickle, concat_images_list
from ldm.models.diffusion.sync_dreamer_utils import get_warp_coordinates, create_target_volume, get_proxy_warp_coordinates
from ldm.models.diffusion.sync_dreamer_network import NoisyTargetViewEncoder, ControlSpatialTime3DNet, FrustumTV3DNet
from ldm.modules.diffusionmodules.util import make_ddim_timesteps, timestep_embedding
from ldm.modules.encoders.modules import FrozenCLIPImageEmbedder
from ldm.util import instantiate_from_config, get_3x4_RT_matrix_from_az_el, save_pickle, read_pickle
from ldm.models.diffusion.sync_dreamer import SyncMultiviewDiffusion, disable_training_module, disabled_train, repeat_to_batch, UNetWrapper, SyncDDIMSampler, SpatialVolumeNet
from externs.pvcnn.modules import ProxyVoxelConv
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler
from ldm.DPMPPScheduler import DPMPPScheduler

class ControlSpatialVolumeNet(SpatialVolumeNet):
    def __init__(self, time_dim, view_dim, view_num,
                 input_image_size=256, frustum_volume_depth=48,
                 spatial_volume_size=32, spatial_volume_length=0.5,
                 frustum_volume_length=0.86603, # sqrt(3)/2
                 block=(1, 1, 3, 32), feature_scale=1
                 ):
        super().__init__(time_dim, view_dim, view_num,
                 input_image_size, frustum_volume_depth,
                 spatial_volume_size, spatial_volume_length,
                 frustum_volume_length)
        self.feature_scale = feature_scale
        if block is not None:
            in_channels, out_channels, kernal_size, resolution = block
            self.pvcnn = ProxyVoxelConv(in_channels, out_channels, kernal_size, resolution)
            self.controlnet = ControlSpatialTime3DNet(input_dim=16 * view_num, time_dim=time_dim, proxy_input_dim=1, dims=(64, 128, 256, 512))

    def construct_spatial_volume(self, x, t_embed, v_embed, target_poses, target_Ks, proxy=None):
        """
        @param x:            B,N,4,H,W
        @param t_embed:      B,t_dim
        @param v_embed:      B,N,v_dim
        @param target_poses: N,3,4
        @param target_Ks:    N,3,3
        @return:
        """
        B, N, _, H, W = x.shape
        V = self.spatial_volume_size
        device = x.device

        spatial_volume_verts = torch.linspace(-self.spatial_volume_length, self.spatial_volume_length, V, dtype=torch.float32, device=device)
        spatial_volume_verts = torch.stack(torch.meshgrid(spatial_volume_verts, spatial_volume_verts, spatial_volume_verts), -1)
        spatial_volume_verts = spatial_volume_verts.reshape(1, V ** 3, 3)[:, :, (2, 1, 0)]
        spatial_volume_verts = spatial_volume_verts.view(1, V, V, V, 3).permute(0, 4, 1, 2, 3).repeat(B, 1, 1, 1, 1)

        # encode source features
        t_embed_ = t_embed.view(B, 1, self.time_dim).repeat(1, N, 1).view(B, N, self.time_dim)
        # v_embed_ = v_embed.view(1, N, self.view_dim).repeat(B, 1, 1).view(B, N, self.view_dim)
        v_embed_ = v_embed
        target_Ks = target_Ks.unsqueeze(0).repeat(B, 1, 1, 1)
        target_poses = target_poses.unsqueeze(0).repeat(B, 1, 1, 1)

        proxy_video = []
        # extract 2D image features
        spatial_volume_feats = []
        # project source features
        for ni in range(0, N):
            pose_source_ = target_poses[:, ni]
            K_source_ = target_Ks[:, ni]
            x_ = self.target_encoder(x[:, ni], t_embed_[:, ni], v_embed_[:, ni])
            C = x_.shape[1]

            coords_source = get_warp_coordinates(spatial_volume_verts, x_.shape[-1], self.input_image_size, K_source_, pose_source_).view(B, V, V * V, 2)
            unproj_feats_ = F.grid_sample(x_, coords_source, mode='bilinear', padding_mode='zeros', align_corners=True)
            unproj_feats_ = unproj_feats_.view(B, C, V, V, V)
            spatial_volume_feats.append(unproj_feats_)

        spatial_volume_feats = torch.stack(spatial_volume_feats, 1) # B,N,C,V,V,V
        N = spatial_volume_feats.shape[1]
        spatial_volume_feats = spatial_volume_feats.view(B, N*C, V, V, V)

        if proxy is not None:
            # proxy in [-0.5, 0.5]
            _, num_proxy, _ = proxy.shape
            proxy += 0.5 # [0, 1]
            proxy = proxy.permute(0, 2, 1)
            proxy_feature = torch.ones([B, 1, num_proxy], dtype=proxy.dtype).to(proxy.device) * self.feature_scale
            proxy_feature, _ = self.pvcnn([proxy_feature, proxy])
            proxy_feature = proxy_feature.permute(0, 1, 4, 3, 2)

            proxy_residual = self.controlnet(spatial_volume_feats, t_embed, proxy_feature)
        else:
            proxy_residual = None
        spatial_volume_feats = self.spatial_volume_feats(spatial_volume_feats, t_embed, proxy_residual)  # b,64,32,32,32

        return spatial_volume_feats
class CtrlDemo(SyncMultiviewDiffusion):
    def __init__(self, unet_config, scheduler_config,
                 finetune_unet=False, finetune_projection=True,
                 view_num=16, image_size=256,
                 cfg_scale=3.0, output_num=8, batch_view_num=4,
                 drop_conditions=False, drop_scheme='default',
                 clip_image_encoder_path="/apdcephfs/private_rondyliu/projects/clip/ViT-L-14.pt",
                 sample_type='ddim', sample_steps=200, feature_scale=1):
        pl.LightningModule.__init__(self)

        self.finetune_unet = finetune_unet
        self.finetune_projection = finetune_projection

        self.view_num = view_num
        self.viewpoint_dim = 4
        self.output_num = output_num
        self.image_size = image_size

        self.batch_view_num = batch_view_num
        self.cfg_scale = cfg_scale

        self.clip_image_encoder_path = clip_image_encoder_path

        self._init_time_step_embedding()
        self._init_first_stage()
        self._init_schedule()
        self._init_multiview()
        self._init_clip_image_encoder()
        self._init_clip_projection()

        self.spatial_volume = ControlSpatialVolumeNet(self.time_embed_dim, self.viewpoint_dim, self.view_num, feature_scale=feature_scale)
        self.model = UNetWrapper(unet_config, drop_conditions=drop_conditions, drop_scheme=drop_scheme)
        self.scheduler_config = scheduler_config

        latent_size = image_size//8
        self._init_sampler(latent_size, sample_steps)
    def _init_sampler(self, latent_size, sample_steps):
        self.sampler = CtrlDemoSampler(self, sample_steps , 'ddim', "uniform", 1.0, latent_size=latent_size)

    def prepare(self, batch):
        x, clip_embed, input_info = super().prepare(batch)
        if 'proxy' in batch:
            input_info['proxy'] = batch['proxy']
        return x, clip_embed, input_info

    def inference(self, sampler, batch, cfg_scale, batch_view_num, return_inter_results=False, inter_interval=50, inter_view_interval=2, callback=None):
        _, clip_embed, input_info = self.prepare(batch)
        x_sample, _, total_spatial_volume = sampler.inference(input_info, clip_embed, unconditional_scale=cfg_scale, log_every_t=inter_interval, batch_view_num=batch_view_num,callback=callback)

        return x_sample, total_spatial_volume


    def decode_latents(self, x_sample):
        images = self.decode_first_stage(x_sample)
        return images

    def get_target_view_feats(self, x_input, spatial_volume, clip_embed, t_embed, v_embed, target_index, spatial_volume_params):
        """
        @param x_input:        B,4,H,W
        @param spatial_volume: B,C,V,V,V
        @param clip_embed:     B,1,768
        @param t_embed:        B,t_dim
        @param v_embed:        B,N,v_dim
        @param target_index:   B,TN
        @return:
            tensors of size B*TN,*
        """
        B, _, H, W = x_input.shape

        frustum_volume_feats, frustum_volume_depth = self.spatial_volume.construct_view_frustum_volume(spatial_volume, t_embed, v_embed, **spatial_volume_params)

        # clip
        TN = target_index.shape[1]
        v_embed_ = v_embed[torch.arange(B)[:,None], target_index].view(B*TN, self.viewpoint_dim) # B*TN,v_dim
        clip_embed_ = clip_embed.unsqueeze(1).repeat(1,TN,1,1).view(B*TN,1,768)
        clip_embed_ = self.cc_projection(torch.cat([clip_embed_, v_embed_.unsqueeze(1)], -1))  # B*TN,1,768

        x_input_ = x_input.unsqueeze(1).repeat(1, TN, 1, 1, 1).view(B * TN, 4, H, W)

        x_concat = x_input_
        return clip_embed_, frustum_volume_feats, x_concat

    def training_step(self, batch):
        B = batch['target_image'].shape[0]
        time_steps = torch.randint(0, self.num_timesteps, (B,), device=self.device).long()

        x, clip_embed, input_info = self.prepare(batch)
        x_noisy, noise = self.add_noise(x, time_steps)  # B,N,4,H,W

        N = self.view_num
        target_index = torch.randint(0, N, (B, 1), device=self.device).long() # B, 1
        v_embed = self.get_viewpoint_embedding(B, input_info['elevation']) # N,v_dim
        proxy_ = input_info['proxy'].detach().clone()
        t_embed = self.embed_time(time_steps)
        spatial_volume = self.spatial_volume.construct_spatial_volume(x_noisy, t_embed, v_embed, self.poses, self.Ks, proxy=proxy_)
        spatial_volume_params = {'poses': self.poses, 'Ks': self.Ks, 'target_indices': target_index}
        clip_embed, volume_feats, x_concat = self.get_target_view_feats(input_info['x'], spatial_volume, clip_embed, t_embed, v_embed, target_index, spatial_volume_params=spatial_volume_params)

        x_noisy_ = x_noisy[torch.arange(B)[:,None],target_index][:,0] # B,4,H,W
        noise_predict = self.model(x_noisy_, time_steps, clip_embed, volume_feats, x_concat, is_train=True) # B,4,H,W

        noise_target = noise[torch.arange(B)[:,None],target_index][:,0] # B,4,H,W
        # loss simple for diffusion
        loss_simple = torch.nn.functional.mse_loss(noise_target, noise_predict, reduction='none')
        loss = loss_simple.mean()
        self.log('sim', loss_simple.mean(), prog_bar=True, logger=True, on_step=True, on_epoch=True, rank_zero_only=True)

        # log others
        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        self.log("step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        return loss

    def configure_optimizers(self):
        lr = self.learning_rate
        print(f'setting learning rate to {lr:.4f} ...')
        paras = []
        paras.append({"params": self.spatial_volume.controlnet.parameters(), "lr": lr},)

        opt = torch.optim.AdamW(paras, lr=lr)

        scheduler = instantiate_from_config(self.scheduler_config)
        print("Setting up LambdaLR scheduler...")
        scheduler = [{'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule), 'interval': 'step', 'frequency': 1}]
        return [opt], scheduler

class CtrlDemoSampler:
    def __init__(self, model: CtrlDemo, scheduler_steps, scheduler_name='ddim', ddim_discretize="uniform", ddim_eta=1.0, latent_size=32):
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.latent_size = latent_size
        self.eta = ddim_eta
        self.scheduler_name = scheduler_name
        self.scheduler_steps = scheduler_steps
        if scheduler_name == 'ddim':
            self.scheduler=DDIMScheduler(num_train_timesteps=self.ddpm_num_timesteps, beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", set_alpha_to_one=False, clip_sample=False, steps_offset=1, trained_betas=None)
        elif scheduler_name == 'dpm++':
            self.scheduler=DPMPPScheduler(num_train_timesteps=self.ddpm_num_timesteps, beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", use_karras_sigmas=True)
            # self.scheduler=DPMSolverMultistepScheduler(num_train_timesteps=self.ddpm_num_timesteps, beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", use_karras_sigmas=True)
        
        self.scheduler.set_timesteps(scheduler_steps, device=self.model.device)
        self.set_ctrl3D_params([{"start_percent":0.0, "end_percent":1.0}], strength=1.0)
    
    def parameterization(self):
        sampler_params = {
                            'scheduler_steps' : self.scheduler_steps,
                            'scheduler_name': self.scheduler_name,
                            'ddim_eta': self.eta, 
                            'latent_size': self.latent_size,
                         }
        return sampler_params
        # save_pickle(sampler_params, save_path)

    @classmethod
    def from_pkl(cls, model, pkl_dir):
        params = read_pickle(pkl_dir)
        return cls(model, **params)

    def set_ctrl3D_params(self, ctrl3D_params_list, strength: float=1.0):
        self.ctrl3D_params_list = ctrl3D_params_list
        self.model.spatial_volume.controlnet.ctrl_strength = strength
  
    def concat_proxy(self, proxys, inferenc_step):
        step = inferenc_step/1000
        valid_proxys = []
        for params, pxy in zip(self.ctrl3D_params_list, proxys):
            if not isinstance(pxy, torch.Tensor):
                continue
            if (1-params.end_percent) < step < (1-params.start_percent):
                valid_proxys.append(pxy)
        if len(valid_proxys) == 0:
            return None
        valid_proxys=torch.cat(valid_proxys, dim=1)
        return valid_proxys

    @torch.no_grad()
    def denoise_apply_impl(self, x_target_noisy, time_steps, noise_pred, is_step0=False):
        if self.scheduler_name == 'ddim':
            result = self.scheduler.step(noise_pred, time_steps, x_target_noisy, return_dict=True, eta=self.eta if not is_step0 else 0)
        elif self.scheduler_name == 'dpm++':
            result = self.scheduler.step(noise_pred, time_steps, x_target_noisy, return_dict=True)
        x_pred, x_origin = result[0], result[1]
        return x_pred, x_origin

    @torch.no_grad()
    def denoise_apply(self, x_target_noisy, input_info, v_embed, clip_embed, time_steps, index, unconditional_scale, batch_view_num=1, is_step0=False, spatial_volume=None):
        """
        @param x_target_noisy:   B,N,4,H,W
        @param input_info:
        @param clip_embed:       B,M,768
        @param time_steps:       B,
        @param index:            int
        @param unconditional_scale:
        @param batch_view_num:   int
        @param is_step0:         bool
        @return:
        """
        x_input = input_info['x']
        B, N, C, H, W = x_target_noisy.shape

        # construct source data
        t_embed = self.model.embed_time(time_steps)  # B,t_dim
        if spatial_volume is None:
            proxy = None if 'proxy' not in input_info else input_info['proxy'].detach().clone()
            spatial_volume = self.model.spatial_volume.construct_spatial_volume(x_target_noisy, t_embed, v_embed, self.model.poses, self.model.Ks, proxy=proxy)

        e_t = []
        target_indices = torch.arange(N) # N
        for ni in range(0, N, batch_view_num):
            x_target_noisy_ = x_target_noisy[:, ni:ni + batch_view_num]
            VN = x_target_noisy_.shape[1]
            x_target_noisy_ = x_target_noisy_.reshape(B*VN,C,H,W)

            time_steps_ = repeat_to_batch(time_steps, B, VN)
            target_indices_ = target_indices[ni:ni+batch_view_num].unsqueeze(0).repeat(B,1)
            spatial_volume_params = {'poses': self.model.poses, 'Ks': self.model.Ks, 'target_indices': target_indices_}
            clip_embed_, volume_feats_, x_concat_ = self.model.get_target_view_feats(x_input, spatial_volume, clip_embed, t_embed, v_embed, target_indices_, spatial_volume_params)
            if unconditional_scale!=1.0:
                noise = self.model.model.predict_with_unconditional_scale(x_target_noisy_, time_steps_, clip_embed_, volume_feats_, x_concat_, unconditional_scale)
            else:
                noise = self.model.model(x_target_noisy_, time_steps_, clip_embed_, volume_feats_, x_concat_, is_train=False)
            e_t.append(noise.view(B,VN,4,H,W))

        e_t = torch.cat(e_t, 1)
        x_prev, _ = self.denoise_apply_impl(x_target_noisy, int(time_steps[0]), e_t, is_step0)
        return x_prev, spatial_volume

    @torch.no_grad()
    def inference(self, input_info, clip_embed, unconditional_scale=1.0, log_every_t=50, batch_view_num=1, callback=None):
        """
        @param input_info:      x, elevation
        @param clip_embed:      B,M,768
        @param unconditional_scale:
        @param log_every_t:
        @param batch_view_num:
        @return:
        """
        print(f"unconditional scale {unconditional_scale:.1f}")
        C, H, W = 4, self.latent_size, self.latent_size
        B = clip_embed.shape[0]
        N = self.model.view_num
        device = self.model.device
        x_target_noisy = torch.randn([B, N, C, H, W], device=device)

        elevation_input = input_info['elevation']
        v_embed = self.model.get_viewpoint_embedding(B, elevation_input) # B,N,v_dim
        
        timesteps = self.scheduler.timesteps
        intermediates = {'x_inter': []}
        # time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]

        iterator = tqdm(timesteps, desc='DDIM Sampler', total=total_steps)
        
        condition_name = ['proxy']
        total_volume_feature = []
        for i, step in enumerate(iterator):
            index = total_steps - i - 1 # index in ddim state
            time_steps = torch.full((B,), step, device=device, dtype=torch.long)
            t_input_info = {k:v for k, v in input_info.items() if k not in condition_name}
            valid_proxy = self.concat_proxy(input_info['proxy'], int(step))
            if valid_proxy is not None:
                t_input_info['proxy'] = valid_proxy
            x_target_noisy, spatial_volume = self.denoise_apply(x_target_noisy, t_input_info, v_embed, clip_embed, time_steps, index, unconditional_scale, batch_view_num=batch_view_num, is_step0=index==0)
            total_volume_feature.append(spatial_volume)
            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(x_target_noisy)
            if callback is not None:
                callback(i, total_steps)

        return x_target_noisy, intermediates, total_volume_feature

    def get_clip_feature(self, x_input, clip_embed, v_embed, target_index):
        B, _, H, W = x_input.shape
        TN = target_index.shape[1]
        viewpoint_dim = 4
        v_embed_ = v_embed[torch.arange(B)[:,None], target_index].view(B*TN, self.model.viewpoint_dim) # B*TN,v_dim
        clip_embed_ = clip_embed.unsqueeze(1).repeat(1,TN,1,1).view(B*TN,1,768)
        clip_embed_ = self.model.cc_projection(torch.cat([clip_embed_, v_embed_.unsqueeze(1)], -1))  # B*TN,1,768

        x_input_ = x_input.unsqueeze(1).repeat(1, TN, 1, 1, 1).view(B * TN, 4, H, W)

        x_concat = x_input_
        return clip_embed_, x_concat