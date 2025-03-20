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

import imageio.v3 as iio
import cv2
import numpy as np
import imageio

import os
import tyro
import glob
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import time

import kiui
from kiui.cam import orbit_camera

from core.options import AllConfigs, Options
from core.models import LGM
from mvdream.pipeline_mvdream import MVDreamPipeline

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

opt = tyro.cli(AllConfigs)

# model
model = LGM(opt)

# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {opt.resume}')
else:
    print(f'[WARN] model randomly initialized, are you sure?')

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.half().to(device)
model.eval()

bg_color = torch.tensor([255, 255, 255], dtype=torch.float32, device="cuda") / 255.

rays_embeddings = model.prepare_default_rays(device)

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

# load image dream
pipe = MVDreamPipeline.from_pretrained(
    "ashawkey/imagedream-ipmv-diffusers", # remote weights
    torch_dtype=torch.float16,
    trust_remote_code=True,
    # local_files_only=True,
)
pipe = pipe.to(device)


def process_eval_video(video_path, T):
    frames = iio.imread(video_path)
    frames = [frames[x] for x in range(frames.shape[0])]
    V = opt.num_input_views
    img_TV = []
    for t in range(T):

        img = frames[t]
        
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        img_V = []
        for v in range(V):
            img_V.append(img)
        img_TV.append(np.stack(img_V, axis=0))

    return np.stack(img_TV, axis=0)


# process function
def process(opt: Options, path):
    name = os.path.splitext(os.path.basename(path))[0]
    print(f'[INFO] Processing {path} --> {name}')
    os.makedirs(opt.workspace, exist_ok=True)

    ref_video = process_eval_video(path, opt.num_frames) # [TV, 512, 512, 3]


    end_time = time.time()

    cv2.imwrite(os.path.join(opt.workspace, f'{name}_orig.png'), ref_video[0,0][..., ::-1] * 255)

    mv_image = pipe('', ref_video[0,0], guidance_scale=5, num_inference_steps=30, elevation=0)
    for v in range(4):
        cv2.imwrite(os.path.join(opt.workspace, f'{name}_mv_{(v-1)%4:03d}.png'), mv_image[v][..., ::-1] * 255)
    mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32


    # generate gaussians
    input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
    input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
    input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            gaussians_all_frame = model.forward_gaussians(input_image)
            
            B, T, V = 1, gaussians_all_frame.shape[0]//opt.batch_size, opt.num_views
            gaussians_all_frame = gaussians_all_frame.reshape(B, T, *gaussians_all_frame.shape[1:])

            # align azimuth
            best_azi = 0
            best_diff = 1e8
            for v, azi in enumerate(np.arange(-180, 180, 1)):
                gaussians = gaussians_all_frame[:, 0]
                
                cam_poses = torch.from_numpy(orbit_camera(0, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                result = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), bg_color=bg_color)
                image = result['image']
                alpha = result['alpha']

                image = image.squeeze(1).permute(0,2,3,1).squeeze(0).contiguous().float().cpu().numpy()
                image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

                diff = np.mean((image- ref_video[0,0]) ** 2)

                if diff < best_diff:
                    best_diff = diff
                    best_azi = azi

            print("Best aligned azimuth: ", best_azi)

            mv_image = []
            for v, azi in enumerate(np.arange(0, 360, 90)):
                gaussians = gaussians_all_frame[:, 0]
                
                cam_poses = torch.from_numpy(orbit_camera(0, azi + best_azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                scale = 1

                result = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), bg_color=bg_color)
                image = result['image']
                alpha = result['alpha']

                imageio.imwrite(os.path.join(opt.workspace, f'{name}_{v:03d}.png'), (image.squeeze(1).permute(0,2,3,1).squeeze(0).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
                
                if azi in [0, 90, 180, 270]:     
                    rendered_image = image.squeeze(1)
                    rendered_image = F.interpolate(rendered_image, (256, 256))
                    rendered_image = rendered_image.permute(0,2,3,1).contiguous().float().cpu().numpy()
                    mv_image.append(rendered_image)
            mv_image = np.concatenate(mv_image, axis=0)
            print(f"Generate 3D takes {time.time()-end_time} s")
            
            images = []
            azimuth = np.arange(0, 360, 4, dtype=np.int32)
            elevation = 0
            for azi in azimuth:
                gaussians = gaussians_all_frame[:, 0]
                
                cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                
                # cameras needed by gaussian rasterizer
                cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                scale = 1

                image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), bg_color=bg_color)['image']
                images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

            images = np.concatenate(images, axis=0)
            imageio.mimwrite(os.path.join(opt.workspace, f'{name}.mp4'), images, fps=30)


    torch.cuda.empty_cache()



assert opt.test_path is not None
if os.path.isdir(opt.test_path):
    file_paths = glob.glob(os.path.join(opt.test_path, "*"))
else:
    file_paths = [opt.test_path]

for path in sorted(file_paths):
    process(opt, path)
