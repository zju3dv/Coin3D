import pytorch_lightning as pl
import numpy as np
import torch
import PIL
import os
from skimage.io import imread
import webdataset as wds
import PIL.Image as Image
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

from ldm.base_utils import read_pickle, pose_inverse
from ldm.data.sync_dreamer import SyncDreamerTrainData, SyncDreamerDataset
import torchvision.transforms as transforms
import torchvision
from einops import rearrange

from ldm.util import prepare_inputs, prepare_proxy



class ControlSyncDreamerTrainData(SyncDreamerTrainData):
    def __init__(self, target_dir, input_dir, proxy_dir, uid_set_pkl, image_size=256):
        self.default_image_size = 256
        self.image_size = image_size
        self.target_dir = Path(target_dir)
        self.input_dir = Path(input_dir)
        self.proxy_dir = Path(proxy_dir)

        self.proxy_uids = read_pickle(uid_set_pkl)
        # self.proxy_uids = ['0012053f094f4309808f52b3efb88977.txt']
        self.uids = [i.split('.')[0] for i in self.proxy_uids]
        assert len(self.proxy_uids) == len(self.uids)
        print('============= length of dataset %d =============' % len(self.uids))

        image_transforms = []
        image_transforms.extend([transforms.ToTensor(), transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)
        self.num_images = 16
    
    def get_data_for_index(self, index):
        target_dir = os.path.join(self.target_dir, self.uids[index])
        input_dir = os.path.join(self.input_dir, self.uids[index])

        views = np.arange(0, self.num_images)
        start_view_index = np.random.randint(0, self.num_images)
        # start_view_index = 0
        views = (views + start_view_index) % self.num_images

        target_images = []
        for si, target_index in enumerate(views):
            img = self.load_index(target_dir, target_index)
            target_images.append(img)
        target_images = torch.stack(target_images, 0)
        input_img = self.load_index(input_dir, start_view_index)

        K, azimuths, elevations, distances, cam_poses = read_pickle(os.path.join(input_dir, f'meta.pkl'))
        input_elevation = torch.from_numpy(elevations[start_view_index:start_view_index+1].astype(np.float32))
        result =  {"target_image": target_images, "input_image": input_img, "input_elevation": input_elevation}
        
        proxy_path = os.path.join(self.proxy_dir, self.proxy_uids[index])
        proxy = prepare_proxy(proxy_path)
        rot_rad = np.deg2rad(-22.5*start_view_index)
        rotate_matrix = torch.from_numpy(np.array([[np.cos(rot_rad), -np.sin(rot_rad), 0], [np.sin(rot_rad), np.cos(rot_rad), 0], [0, 0, 1]]))
        proxy = (rotate_matrix * proxy[:, None, :]).sum(-1).float()
        result['proxy'] = proxy
        return result

class ControlSyncDreamerEvalData(Dataset):
    def __init__(self, image_dir, proxy_dir, uid_set_pkl):
        self.image_size = 256
        self.image_dir = Path(image_dir)
        self.proxy_dir = Path(proxy_dir)
        self.crop_size = 20

        self.proxy_uids = read_pickle(uid_set_pkl)
        # self.proxy_uids = ['0012053f094f4309808f52b3efb88977.txt']
        self.uids = [i.split('.')[0] for i in self.proxy_uids]
        assert len(self.proxy_uids) == len(self.uids)
        print('============= length of dataset %d =============' % len(self.proxy_uids))

    def __len__(self):
        return len(self.uids)

    def get_data_for_index(self, index):
        input_img_path = os.path.join(self.image_dir, self.uids[index], '000.png')
        proxy_path = os.path.join(self.proxy_dir, self.proxy_uids[index])
        elevation = 30
        result = prepare_inputs(input_img_path, elevation, 200)
        result['proxy'] = prepare_proxy(proxy_path)
        return result

    def __getitem__(self, index):
        return self.get_data_for_index(index)

class ControlSyncDreamerDataset(SyncDreamerDataset):
    def __init__(self, target_dir, input_dir, validation_dir, proxy_dir, batch_size, uid_set_pkl, valid_uid_set_pkl, image_size=256, num_workers=4, seed=0, **kwargs):
        pl.LightningDataModule.__init__(self)
        self.target_dir = target_dir
        self.input_dir = input_dir
        self.validation_dir = validation_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.uid_set_pkl = uid_set_pkl
        self.valid_uid_set_pkl = valid_uid_set_pkl
        self.seed = seed
        self.additional_args = kwargs
        self.image_size = image_size

        # --------------------------
        self.proxy_dir = proxy_dir

    def setup(self, stage):
        if stage in ['fit']:
            self.train_dataset = ControlSyncDreamerTrainData(self.target_dir, self.input_dir, self.proxy_dir, uid_set_pkl=self.uid_set_pkl, image_size=256)
            self.val_dataset = ControlSyncDreamerEvalData(image_dir=self.validation_dir, proxy_dir=self.proxy_dir, uid_set_pkl=self.valid_uid_set_pkl)
        else:
            raise NotImplementedError
