# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import random
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import kiui
from core.options import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter

from kiui.cam import orbit_camera

import tarfile
from io import BytesIO


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def load_np_array_from_tar(tar, path):
    array_file = BytesIO()
    array_file.write(tar.extractfile(path).read())
    array_file.seek(0)
    return np.load(array_file)

def interpolate_tensors(tensor):
    # Extract the first and last tensors along the first dimension (B)
    start_tensor = tensor[0]    # shape [4, 3, 256, 256]
    end_tensor = tensor[-1]     # shape [4, 3, 256, 256]
    tensor_interp = copy.deepcopy(tensor)

    # Iterate over the range from 1 to second-last index
    for i in range(1, tensor.size(0) - 1):
        # Calculate the weight for interpolation

        weight = (i - 0) / (tensor.size(0) - 1)
        # Interpolate between start_tensor and end_tensor
        tensor_interp[i] = torch.lerp(start_tensor, end_tensor, weight)


    return tensor_interp

class ObjaverseDataset(Dataset):

    def _warn(self):
        raise NotImplementedError('this dataset is just an example and cannot be used directly, you should modify it to your own setting! (search keyword TODO)')

    def __init__(self, opt: Options, training=True, evaluating=False):
        
        self.opt = opt
        self.training = training
        self.evaluating = evaluating

        self.items = []
        with open(self.opt.datalist, 'r') as f:
            for line in f.readlines():
                self.items.append(line.strip())
        
        anim_map = {}
        for x in self.items:
            k = x.split('-')[1]
            if k in anim_map:
                anim_map[k] += '|'+x
            else:
                anim_map[k] = x
        self.items = list(anim_map.values())



        # naive split
        if self.training:
            self.items = self.items[:-self.opt.batch_size]
        elif self.evaluating:
            self.items = self.items[::1000]
        else:
            self.items = self.items[-self.opt.batch_size:]


        # default camera intrinsics
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1


    def __len__(self):
        return len(self.items)

    def _get_batch(self, idx):
        # uid = self.items[idx]
        if self.training:
            uid = random.choice(self.items[idx].split('|'))
        else:
            uid = self.items[idx].split('|')[0]

        results = {}

        # load num_views images
        images = []
        masks = []
        cam_poses = []
        
        if self.training and self.opt.shuffle_input:
            vids = np.random.permutation(np.arange(32, 48))[:self.opt.num_input_views].tolist() + np.random.permutation(32).tolist()
        else:
            vids = np.arange(32, 48, 4).tolist() + np.arange(32).tolist()

        random_tar_name = 'random_24fps/' + uid
        fixed_16_tar_name = 'fixed_16_24fps/' + uid

        local_random_tar_name = os.environ["DATA_HOME"] + random_tar_name.replace('/', '-')
        local_fixed_16_tar_name = os.environ["DATA_HOME"] + fixed_16_tar_name.replace('/', '-')

        tar_random = tarfile.open(local_random_tar_name)
        tar_fixed = tarfile.open(local_fixed_16_tar_name)
        
        max_T = 24

        T = self.opt.num_frames

        start_frame = np.random.randint(max_T - T)

        for t_idx in range(T):
            t = start_frame + t_idx
            vid_cnt = 0
            for vid in vids:
                if vid >= 32:
                    vid = vid % 32
                    tar = tar_fixed
                else:
                    tar = tar_random

                image_path = os.path.join('.', f'{vid:03d}/img', f'{t:03d}.jpg')
                mask_path = os.path.join('.', f'{vid:03d}/mask', f'{t:03d}.png')

                elevation_path = os.path.join('.', f'{vid:03d}/camera', f'elevation.npy')
                rotation_path = os.path.join('.', f'{vid:03d}/camera', f'rotation.npy')

                try :
                    image = np.frombuffer(tar.extractfile(image_path).read(), np.uint8)
                    image = torch.from_numpy(cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255) # [512, 512, 4] in [0, 1]

                    azi = load_np_array_from_tar(tar, rotation_path)[t, None]
                    elevation = load_np_array_from_tar(tar, elevation_path)[t, None] * -1       # to align with pretrained LGM
                    azi = float(azi)
                    elevation = float(elevation)
                    c2w = torch.from_numpy(orbit_camera(elevation, azi, radius=1.5, opengl=True))

                    image = image.permute(2, 0, 1) # [4, 512, 512]

                    mask = np.frombuffer(tar.extractfile(mask_path).read(), np.uint8)
                    mask = torch.from_numpy(cv2.imdecode(mask, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255).unsqueeze(0)  # [512, 512, 4] in [0, 1]
                except:

                    return self.__getitem__(idx - 1)
                image = F.interpolate(image.unsqueeze(0), size=(512, 512), mode='nearest').squeeze(0)
                mask = F.interpolate(mask.unsqueeze(0), size=(512, 512), mode='nearest').squeeze(0)

                image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
                image = image[[2,1,0]].contiguous() # bgr to rgb

                images.append(image)
                masks.append(mask.squeeze(0))
                cam_poses.append(c2w)

                vid_cnt += 1
                if vid_cnt == self.opt.num_views:
                    break

            if vid_cnt < self.opt.num_views:
                print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
                n = self.opt.num_views - vid_cnt
                images = images + [images[-1]] * n
                masks = masks + [masks[-1]] * n
                cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]
        
        images_input = F.interpolate(images.reshape(T, self.opt.num_views, *images.shape[1:])[:, :self.opt.num_input_views].reshape(-1, *images.shape[1:]).clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses.reshape(T, self.opt.num_views, *cam_poses.shape[1:])[:, :self.opt.num_input_views].reshape(-1, *cam_poses.shape[1:]).clone()
    
        # data augmentation
        if self.training:
            images_input = images_input.reshape(T, self.opt.num_input_views, *images_input.shape[1:])
            cam_poses_input = cam_poses_input.reshape(T, self.opt.num_input_views, *cam_poses.shape[1:])

            # apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                for t in range(T):
                    images_input[t, 1:] = grid_distortion(images_input[t, 1:])
            # apply camera jittering (only to input!)
            if random.random() < self.opt.prob_cam_jitter:
                for t in range(T):
                    cam_poses_input[t, 1:] = orbit_camera_jitter(cam_poses_input[t, 1:])

            images_input = images_input.reshape(-1, *images_input.shape[2:])
            cam_poses_input = cam_poses_input.reshape(-1, *cam_poses.shape[1:])

        # masking other views
        images_input = images_input.reshape(T, self.opt.num_input_views, *images_input.shape[1:])

        images_input_interp = interpolate_tensors(images_input)

        images_input[1:-1, :] = images_input_interp[1:-1, :]
        images_input = images_input.reshape(-1, *images_input.shape[2:])

        cam_poses_input = cam_poses_input.reshape(T, self.opt.num_input_views, *cam_poses.shape[1:])
        cam_poses_input[1:, 1:] = cam_poses_input[0:1, 1:]
        cam_poses_input = cam_poses_input.reshape(-1, *cam_poses.shape[1:])

        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        # build rays for input views
        rays_embeddings = []
        for i in range(self.opt.num_input_views * T):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

     
        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]

        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        results['input'] = final_input

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        return results

    def __getitem__(self, idx):
        while True:
            try:
                results = self._get_batch(idx)
                break
            except Exception as e:
                # print(f"{e}")
                idx = random.randint(0, len(self.items) - 1)
        return results