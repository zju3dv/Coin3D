# Coin3D: Controllable and Interactive 3D Assets Generation with Proxy-Guided Conditioning
### [Project Page](https://zju3dv.github.io/coin3d/) | [Video](https://www.youtube.com/watch?v=d6p3LLbmOnI) | [Paper](https://arxiv.org/abs/2405.08054)
<div align=center>
<img src="media/coin3d_highlight.gif" width="100%"/>
</div>



### ToDo List
- [x] Inference code and pretrained models.
- [ ] Interactive workflow.
- [ ] Training data.
- [ ] Blender Addons
- [ ] Reconstruction using volume-SDS loss



### Preparation for inference
1. Install packages in `requirements.txt`. We test our model on a 80G A100 GPU with 11.8 CUDA and 2.0.1 pytorch.
```angular2html
conda create -n coin3d
conda activate coin3d
pip install -r requirements.txt
```
2. Download checkpoints 

```
mkdir ckpt
cd ckpt
wget https://huggingface.co/WenqiDong/Coin3D-v1/resolve/main/ViT-L-14.pt

wget https://huggingface.co/WenqiDong/Coin3D-v1/resolve/main/model.ckpt
```

### Inference
1. Make sure you have the following models.
```bash
Coin3D
|-- ckpt
    |-- ViT-L-14.ckpt
    |-- model.ckpt
```
2. (Optional) Make sure the input image has a white background. Here we refer to [SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer) and use the following tools for foreground segmentation. Predict foreground mask as the alpha channel. We use [Paint3D](https://apps.microsoft.com/store/detail/paint-3d/9NBLGGH5FV99) to segment the foreground object interactively. 
We also provide a script `foreground_segment.py` using `carvekit` to predict foreground masks and you need to first crop the object region before feeding it to `foreground_segment.py`. We may double check the predicted masks are correct or not.
```bash
python3 foreground_segment.py --input <image-file-to-input> --output <image-file-in-png-format-to-output>
```
3. Using coarse proxy to control 3D generation of multi-view images.
```bash
python3 generate.py \
        --cfg configs/ctrldemo.yaml \
        --ckpt ckpt/model.ckpt \
        --input example/panda/input.png \
        --input_proxy example/panda/proxy.txt \
        --output output/custom \
        --sample_num 1 \
        --cfg_scale 2.0 \
        --elevation 30 \
        --ctrl_end_step 1.0 \
        --sampler ddim_demo
```
Explanation: 
- `--cfg` is the model configuration.
- `--ckpt` is the checkpoint to load.
- `--input` is the input image in the RGBA form. The alpha value means the foreground object mask.
- `--input_proxy` is the input coarse proxy. The proxy contains 256 points by default. [misc.ipynb](misc.ipynb) contains code for using the coarse mesh sampling proxy.
- `--output` is the output directory. Results would be saved to `output/custom/0.png` which contains 16 images of predefined viewpoints per `png` file. 
- `--sample_num` is the number of instances we will generate. 
- `--cfg_scale` is the *classifier-free-guidance*. `2.0` is OK for most cases.
- `--elevation` is the elevation angle of the input image in degree. Need to be set to 30.
- `--ctrl_end_step` is the timestamp of ending 3D control, from `0` to `1.0`, usually set to `0.6` to `1.0`.

4. Run a NeuS or a NeRF for 3D reconstruction.
```bash
# train a neus
python train_renderer.py -i output/custom/0.png \
                         -n custom-neus \
                         -b configs/neus.yaml \
                         -l output/renderer 
# train a nerf
python train_renderer.py -i output/custom/0.png \
                         -n custom-nerf \
                         -b configs/nerf.yaml \
                         -l output/renderer
```
Explanation:
- `-i` contains the multiview images generated by SyncDreamer. Since SyncDreamer does not always produce good results, we may need to select a good generated image set (from `0.png` to `3.png`) for reconstruction.
- `-n` means the name. `-l` means the log dir. Results will be saved to `<log_dir>/<name>` i.e. `output/renderer/custom-neus` and `output/renderer/custom-nerf`.



## Acknowledgement

We deeply appreciate the authors of the following repositories for generously sharing their code, which we have extensively utilized. Their contributions have been invaluable to our work, and we are grateful for their openness and willingness to share their expertise. Our project has greatly benefited from their efforts and dedication.

- [SyncDreamer](https://github.com/liuyuan-pal/SyncDreamer)
- [latent-diffusion](https://github.com/CompVis/latent-diffusion)
- [threestudio](https://github.com/threestudio-project/threestudio)
- [pvcnn](https://github.com/mit-han-lab/pvcnn)
- [stable diffusion](https://github.com/CompVis/stable-diffusion)
- [zero123](https://github.com/cvlab-columbia/zero123)
- [COLMAP](https://colmap.github.io/)
- [NeuS](https://github.com/Totoro97/NeuS)

## Citation
If you find this repository useful in your project, please cite the following work. :)
```
@article{dong2024coin3d,
  title={Coin3D: Controllable and Interactive 3D Assets Generation with Proxy-Guided Conditioning},
  author={Dong, Wenqi and Yang, Bangbang and Ma, Lin and Liu, Xiao and Cui, Liyuan and Bao, Hujun and Ma, Yuewen and Cui, Zhaopeng},
  year={2024},
  eprint={2405.08054},
  archivePrefix={arXiv},
  primaryClass={cs.GR}
}
```
