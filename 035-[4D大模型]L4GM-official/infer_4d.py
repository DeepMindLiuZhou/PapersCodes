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

from copy import deepcopy
import os
import tyro
import glob
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file

import kiui
from kiui.cam import orbit_camera

from core.options import AllConfigs, Options
from core.models import LGM
import time

from core.utils import get_rays, grid_distortion, orbit_camera_jitter

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


USE_INTERPOLATION = True    # set to false to disable interpolation
MAX_RUNS = 100
VIDEO_FPS = 30

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
rays_embeddings = torch.cat([rays_embeddings for _ in range(opt.num_frames)])


interp_opt = deepcopy(opt)
interp_opt.num_frames = 4
model_interp = LGM(interp_opt)
# resume pretrained checkpoint
if interp_opt.interpresume is not None:
    if interp_opt.interpresume.endswith('safetensors'):
        ckpt = load_file(interp_opt.interpresume, device='cpu')
    else:
        ckpt = torch.load(interp_opt.interpresume, map_location='cpu')
    model_interp.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded Interp checkpoint from {interp_opt.interpresume}')
else:
    print(f'[WARN] model_interp randomly initialized, are you sure?')

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_interp = model_interp.half().to(device)
model_interp.eval()


interp_rays_embeddings = model_interp.prepare_default_rays(device)
interp_rays_embeddings = torch.cat([interp_rays_embeddings for _ in range(interp_opt.num_frames)])

tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
proj_matrix[2, 3] = 1

def interpolate_tensors(tensor):
    # Extract the first and last tensors along the first dimension (B)
    start_tensor = tensor[0]    # shape [4, 3, 256, 256]
    end_tensor = tensor[-1]     # shape [4, 3, 256, 256]
    tensor_interp = deepcopy(tensor)

    # Iterate over the range from 1 to second-last index

    for i in range(1, tensor.shape[0] - 1):
        # Calculate the weight for interpolation

        weight = (i - 0) / (tensor.shape[0] - 1)
        # Interpolate between start_tensor and end_tensor
        tensor_interp[i] = torch.lerp(start_tensor, end_tensor, weight)


    return tensor_interp

def process_eval_video(frames, video_path, T, start_t=0, downsample_rate=1):
    L = frames.shape[0]
    vid_name =video_path.split('/')[-1].split('.')[0]
    total_frames = L//downsample_rate
    print(f'{start_t} / {total_frames}')
    frames = [frames[x] for x in range(frames.shape[0])]
    V = opt.num_input_views
    img_TV = []
    for t in range(T):
        t += start_t
        t = min(t, L//downsample_rate-1)
        t*=downsample_rate

        img = frames[t]

        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        img_V = []
        for v in range(V):
            img_V.append(img)
        img_TV.append(np.stack(img_V, axis=0))

    return np.stack(img_TV, axis=0), L//downsample_rate- start_t

def load_mv_img(name, img_dir):
    img_list = []
    for v in range(4):
        img = kiui.read_image(os.path.join(img_dir, name + f'_{v:03d}.png'), mode='uint8')
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img = img / 255.
        img_list.append(img)
    return np.stack(img_list, axis=0)



# process function
def process(opt: Options, path):
    name = os.path.splitext(os.path.basename(path))[0]
    print(f'[INFO] Processing {path} --> {name}')
    os.makedirs(opt.workspace, exist_ok=True)
    frames = iio.imread(path)
    img_dir = opt.workspace
    mv_image = load_mv_img(name, img_dir)

    print(iio.immeta(path))
    FPS = int(iio.immeta(path)['fps'])
    downsample_rate = FPS // 15 if FPS > 15 else 1    # default reconstruction fps 15

    

    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            start_t = 0
            gaussians_all_frame_all_run = []
            gaussians_all_frame_all_run_w_interp = []
            for run_idx in range(MAX_RUNS):
                ref_video, end_t = process_eval_video(frames, path, opt.num_frames, start_t, downsample_rate=downsample_rate)
                ref_video[:, 1:] = mv_image[None, 1:]   # repeat
                input_image = torch.from_numpy(ref_video).reshape([-1, *ref_video.shape[2:]]).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
                input_image = F.interpolate(input_image, size=(opt.input_size, opt.input_size), mode='bilinear', align_corners=False)
                input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
                input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

                end_time = time.time()

                gaussians_all_frame = model.forward_gaussians(input_image)
                print(f"Forward pass takes {time.time()-end_time} s")

                B, T, V = 1, gaussians_all_frame.shape[0]//opt.batch_size, opt.num_views
                gaussians_all_frame = gaussians_all_frame.reshape(B, T, *gaussians_all_frame.shape[1:])

                if run_idx > 0:
                    gaussians_all_frame_wo_inter = gaussians_all_frame[:, 1:max(end_t, 1)]
                else:
                    gaussians_all_frame_wo_inter = gaussians_all_frame

                if gaussians_all_frame_wo_inter.shape[1] > 0 and USE_INTERPOLATION:
                    # render multiview video
                    render_img_TV = []
                    for t in range(gaussians_all_frame.shape[1]):
                        render_img_V = []
                        for v, azi in enumerate(np.arange(0, 360, 90)):

                            gaussians = gaussians_all_frame[:, t]

                            cam_poses = torch.from_numpy(orbit_camera(0, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                            cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction

                            # cameras needed by gaussian rasterizer
                            cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                            cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                            cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                            rendered_image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), bg_color=bg_color)['image']
                            rendered_image = rendered_image.squeeze(1)
                            rendered_image = F.interpolate(rendered_image, (256, 256))
                            rendered_image = rendered_image.permute(0,2,3,1).contiguous().float().cpu().numpy() # B H W C

                            render_img_V.append(rendered_image)
                        render_img_V = np.concatenate(render_img_V, axis=0) # V H W C
                        render_img_TV.append(render_img_V)
                    render_img_TV = np.stack(render_img_TV, axis=0)   # T V H W C
                    ref_video = np.concatenate([np.stack([ref_video[ttt] for _ in range(opt.interpolate_rate)], 0)  for ttt in range(ref_video.shape[0])], 0)


                    for tt in range(gaussians_all_frame_wo_inter.shape[1] -1 ):

                        curr_ref_video = deepcopy( ref_video[ tt * opt.interpolate_rate:  tt * opt.interpolate_rate + interp_opt.num_frames ])
                        curr_ref_video[0, 1:] = render_img_TV[tt, 1:]

                        curr_ref_video[-1, 1:] = render_img_TV[tt+1, 1:]


                        curr_ref_video = torch.from_numpy(curr_ref_video).float().to(
                            device)  # [4, 3, 256, 256]

                        images_input_interp = interpolate_tensors(curr_ref_video)

                        curr_ref_video[1:-1, :] = images_input_interp[1:-1, :]

                        input_image_interp = curr_ref_video.reshape([-1, *curr_ref_video.shape[2:]]).permute(0, 3, 1,  2).float().to(device)  # [4, 3, 256, 256]
                        input_image_interp = F.interpolate(input_image_interp, size=(interp_opt.input_size, interp_opt.input_size), mode='bilinear',
                                                    align_corners=False)
                        input_image_interp = TF.normalize(input_image_interp, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

                        input_image_interp = torch.cat([input_image_interp, interp_rays_embeddings], dim=1).unsqueeze(0)  # [1, 4, 9, H, W]

                        end_time = time.time()
                        gaussians_interp_all_frame = model_interp.forward_gaussians(input_image_interp)
                        print(f"Interpolate forward pass takes {time.time()-end_time} s")

                        B, T, V = 1, gaussians_interp_all_frame.shape[0] // opt.batch_size, opt.num_views
                        gaussians_interp_all_frame = gaussians_interp_all_frame.reshape(B, T, *gaussians_interp_all_frame.shape[1:])

                        if tt > 0:
                            gaussians_interp_all_frame = gaussians_interp_all_frame[:, 1:]

                        gaussians_all_frame_all_run_w_interp.append(gaussians_interp_all_frame)

                        

                    gaussians_all_frame_all_run.append(gaussians_all_frame_wo_inter)
                    start_t += opt.num_frames -1

                    mv_image = []
                    for v, azi in enumerate(np.arange(0, 360, 90)):
                        gaussians = gaussians_all_frame_wo_inter[:, -1]
                        cam_poses = torch.from_numpy(orbit_camera(0, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
                        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
                        # cameras needed by gaussian rasterizer
                        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                        cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                        cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                        rendered_image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), bg_color=bg_color)['image']
                        rendered_image = rendered_image.squeeze(1)
                        rendered_image = F.interpolate(rendered_image, (256, 256))
                        rendered_image = rendered_image.permute(0,2,3,1).contiguous().float().cpu().numpy()
                        mv_image.append(rendered_image)
                    mv_image = np.concatenate(mv_image, axis=0)
                elif gaussians_all_frame_wo_inter.shape[1] > 0:
                    gaussians_all_frame_all_run.append(gaussians_all_frame_wo_inter)
                    start_t += opt.num_frames -1
                else:
                    break

            gaussians_all_frame_wo_interp = torch.cat(gaussians_all_frame_all_run, dim=1)
            if USE_INTERPOLATION:
                gaussians_all_frame_w_interp = torch.cat(gaussians_all_frame_all_run_w_interp, dim=1)

            if USE_INTERPOLATION:
                zip_dump = zip(["wo_interp", "w_interp"], [gaussians_all_frame_wo_interp, gaussians_all_frame_w_interp])
            else:
                zip_dump = zip(["wo_interp"], [gaussians_all_frame_wo_interp])

            for sv_name, gaussians_all_frame in zip_dump:
                if sv_name == "w_interp":
                    ANIM_FPS = FPS / downsample_rate * gaussians_all_frame_w_interp.shape[1] / gaussians_all_frame_wo_interp.shape[1]
                else:
                    ANIM_FPS = FPS / downsample_rate
                print(f"{sv_name} | input video fps: {FPS} | downsample rate: {downsample_rate} | animation fps: {ANIM_FPS} | output video fps: {VIDEO_FPS}")
                render_img_TV = []
                for t in range(gaussians_all_frame.shape[1]):
                    render_img_V = []
                    for v, azi in enumerate(np.arange(0, 360, 90)):

                        gaussians = gaussians_all_frame[:, t]

                        cam_poses = torch.from_numpy(orbit_camera(0, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction

                        # cameras needed by gaussian rasterizer
                        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                        cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                        cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                        result = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), bg_color=bg_color)
                        image = result['image']
                        alpha = result['alpha']

                        render_img_V.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
                    render_img_V = np.concatenate(render_img_V, axis=2)
                    render_img_TV.append(render_img_V)
                render_img_TV = np.concatenate(render_img_TV, axis=0)


                images = []
                azimuth = np.arange(0, 360, 1*30/VIDEO_FPS, dtype=np.int32)
                elevation = 0
                t = 0
                delta_t = ANIM_FPS / VIDEO_FPS
                for azi in azimuth:
                    if azi in [0, 90, 180, 270]:
                        cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
                        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction

                        # cameras needed by gaussian rasterizer
                        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                        cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                        cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                        for _ in range(45):
                            gaussians = gaussians_all_frame[:, int(t) % gaussians_all_frame.shape[1]]
                            t += delta_t
                            image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), bg_color=bg_color)['image']
                            images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
                    else:
                        gaussians = gaussians_all_frame[:, int(t) % gaussians_all_frame.shape[1]]
                        t += delta_t

                        cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=opt.cam_radius, opengl=True)).unsqueeze(0).to(device)

                        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction

                        # cameras needed by gaussian rasterizer
                        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
                        cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
                        cam_pos = - cam_poses[:, :3, 3] # [V, 3]

                        image = model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), bg_color=bg_color)['image']
                        images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

                images = np.concatenate(images, axis=0)

                torch.cuda.empty_cache()


                imageio.mimwrite(os.path.join(opt.workspace, f'{sv_name}_{name}_fixed.mp4'), render_img_TV, fps=ANIM_FPS)
                print("Fixed video saved.")
                imageio.mimwrite(os.path.join(opt.workspace, f'{sv_name}_{name}.mp4'), images, fps=VIDEO_FPS)
                print("Stop video saved.")


assert opt.test_path is not None

if os.path.isdir(opt.test_path):
    file_paths = glob.glob(os.path.join(opt.test_path, "*"))
else:
    file_paths = [opt.test_path]

for path in sorted(file_paths):
    process(opt, path)
