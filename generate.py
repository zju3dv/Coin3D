import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.io import imsave
import sys
import os
# os.chdir(os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(__file__))
from ldm.models.diffusion.sync_dreamer import SyncMultiviewDiffusion, SyncDDIMSampler
from ldm.models.diffusion.ctrldemo_sync_dreamer import CtrlDemo, CtrlDemoSampler
from ldm.util import instantiate_from_config, prepare_inputs, prepare_proxy
from ldm.util import Ctrl3DParams


def load_model(cfg,ckpt,strict=True):
    config = OmegaConf.load(cfg)
    model = instantiate_from_config(config.model)
    print(f'loading model from {ckpt} ...')
    ckpt = torch.load(ckpt,map_location='cpu')
    model.load_state_dict(ckpt['state_dict'],strict=strict)
    model = model.cuda().eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',type=str, default='configs/syncdreamer.yaml')
    parser.add_argument('--ckpt',type=str, default='ckpt/syncdreamer-step80k.ckpt')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--input_proxy', type=str, default=None)
    parser.add_argument('--start_view', type=int, default=0)
    parser.add_argument('--elevation', type=float, required=True)

    parser.add_argument('--sample_num', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=-1)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    parser.add_argument('--ctrl_start_step', type=float, default=0.0)
    parser.add_argument('--ctrl_end_step', type=float, default=1.0)
    parser.add_argument('--batch_view_num', type=int, default=8)
    parser.add_argument('--seed', type=int, default=6033)

    parser.add_argument('--sampler', type=str, default='ddim_sync')
    parser.add_argument('--sample_steps', type=int, default=50)
    flags = parser.parse_args()

    torch.random.manual_seed(flags.seed)
    np.random.seed(flags.seed)

    model = load_model(flags.cfg, flags.ckpt, strict=False)
    if flags.input_proxy is not None:
        assert isinstance(model, CtrlDemo)
    else:
        assert isinstance(model, SyncMultiviewDiffusion)
    Path(f'{flags.output}').mkdir(exist_ok=True, parents=True)

    # prepare data
    if flags.elevation != 30:
        raise ValueError("The elevation needs to be set to 30.")
    data = prepare_inputs(flags.input, flags.elevation, flags.crop_size)
    if flags.input_proxy is not None:
        data['proxy'] = prepare_proxy(flags.input_proxy, flags.start_view)
    for k, v in data.items():
        data[k] = v.unsqueeze(0).cuda()
        data[k] = torch.repeat_interleave(data[k], flags.sample_num, dim=0)
    if flags.sampler=='ddim_sync':
        sampler = SyncDDIMSampler(model, flags.sample_steps)
    elif flags.sampler=='ddim_demo':
        data['proxy'] = [data['proxy']]
        sampler = CtrlDemoSampler(model, flags.sample_steps)
        ctrl3D_params = [Ctrl3DParams(256, flags.ctrl_start_step, flags.ctrl_end_step)]
        sampler.set_ctrl3D_params(ctrl3D_params, 1.0)
    else:
        raise NotImplementedError

    x_sample = model.inference(sampler, data, flags.cfg_scale, flags.batch_view_num)[0]

    images = model.decode_latents(x_sample[0]).unsqueeze(0)
    B, N, _, H, W = images.shape
    images = (torch.clamp(images,max=1.0,min=-1.0) + 1) * 0.5
    images = (images.permute(0, 1, 3, 4, 2).cpu().numpy() * 255).astype(np.uint8)

    for bi in range(B):
        output_fn = Path(flags.output)/ f'{bi}.png'
        imsave(output_fn, np.concatenate([images[bi,ni] for ni in range(N)], 1))

if __name__=="__main__":
    main()

