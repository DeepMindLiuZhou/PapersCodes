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

import tyro
import time
import random

import torch
from core.options import AllConfigs
from core.models import LGM

from accelerate import Accelerator, DistributedDataParallelKwargs
from safetensors.torch import load_file

import kiui
from PIL import Image

import json
import os
import numpy as np
import imageio

def main():    
    opt = tyro.cli(AllConfigs)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        # kwargs_handlers=[ddp_kwargs],
    )
    if accelerator.is_main_process:
        print(opt)

    # model
    model = LGM(opt)

    epoch_start = 0
    if os.path.exists(f'{opt.workspace}/model.safetensors') and os.path.exists(f'{opt.workspace}/metadata.json'):
        opt.resume = f'{opt.workspace}/model.safetensors'
        with open(f'{opt.workspace}/metadata.json', 'r') as f:
            dc = json.load(f)
            epoch_start = dc['epoch'] + 1


    # resume
    if opt.resume is not None and opt.resume != 'None':
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        
        # tolerant load (only load matching shapes)
        # model.load_state_dict(ckpt, strict=False)
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:
                    accelerator.print(f'[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                accelerator.print(f'[WARN] unexpected param {k}: {v.shape}')
    
    # data
    if opt.data_mode == '4d':
        from core.provider_objaverse_4d import ObjaverseDataset as Dataset
    elif opt.data_mode == '4d_interp':
        from core.provider_objaverse_4d_interp import ObjaverseDataset as Dataset
    else:
        raise NotImplementedError

    train_dataset = Dataset(opt, training=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = Dataset(opt, training=False)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.05, betas=(0.9, 0.95))

    # scheduler (per-iteration)
    total_steps = opt.num_epochs * len(train_dataloader)
    pct_start = 3000 / total_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=opt.lr, total_steps=total_steps, pct_start=pct_start)

    if epoch_start > 0:
        optimizer.load_state_dict(torch.load(os.path.join(opt.workspace, 'optimizer.pth'), map_location='cpu'))
        scheduler.load_state_dict(torch.load(os.path.join(opt.workspace, 'scheduler.pth')))

    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )



    # loop
    os.makedirs(opt.workspace, exist_ok=True)
    end_time = time.time()
    for epoch in range(epoch_start, opt.num_epochs):
        # train
        model.train()
        total_loss = 0
        total_psnr = 0
        for i, data in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                optimizer.zero_grad()

                step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs

                out = model(data, step_ratio)
                loss = out['loss']
                psnr = out['psnr']
                accelerator.backward(loss)

                # gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                optimizer.step()
                scheduler.step()

                total_loss += loss.detach()
                total_psnr += psnr.detach()

            if accelerator.is_main_process:
                # logging
                if i % 10 == 0:
                    mem_free, mem_total = torch.cuda.mem_get_info()    
                    print(f"[INFO] {i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} step_ratio: {step_ratio:.4f} loss: {loss.item():.6f} time: {time.time() - end_time:.6f}")
                    end_time = time.time()
                
                # save log images
                if i % 500 == 0:
                    if '4d' in opt.data_mode:
                        B, T, V = opt.batch_size, opt.num_frames, opt.num_views

                        gt_images = data['images_output'].reshape(B, T, V, *data['images_output'].shape[2:]).detach() # [B, V, 3, output_size, output_size]
                        pred_images = out['images_pred'].reshape(B, T, V, *out['images_pred'].shape[2:]).detach() # [B, V, 3, output_size, output_size]

                        train_gt_images = []
                        train_pred_images = []
                        for t in range(T):
                            train_gt_images_V = []
                            train_pred_images_V = []
                            for v in range(V):
                                train_gt_images_V.append((gt_images[:, t, v].permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
                                train_pred_images_V.append((pred_images[:, t, v].permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
                            train_gt_images.append(np.concatenate(train_gt_images_V, axis=2))
                            train_pred_images.append(np.concatenate(train_pred_images_V, axis=2))
                        train_gt_images = np.concatenate(train_gt_images, axis=0)
                        train_pred_images = np.concatenate(train_pred_images, axis=0)
                        imageio.mimwrite(f'{opt.workspace}/train_gt_images_{epoch}_{i}.mp4', train_gt_images, fps=8)
                        imageio.mimwrite(f'{opt.workspace}/train_pred_images_{epoch}_{i}.mp4', train_pred_images, fps=8)
                        

                    elif '3d' in opt.data_mode:
                        gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                        kiui.write_image(f'{opt.workspace}/train_gt_images_{epoch}_{i}.jpg', gt_images)

                        pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/train_pred_images_{epoch}_{i}.jpg', pred_images)
                    else:
                        raise NotImplementedError
                    

        total_loss = accelerator.gather_for_metrics(total_loss).mean()
        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if accelerator.is_main_process:
            total_loss /= len(train_dataloader)
            total_psnr /= len(train_dataloader)
            accelerator.print(f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}")
        
        # checkpoint
        accelerator.wait_for_everyone()
        accelerator.save_model(model, opt.workspace)
        accelerator.save_model(model, os.path.join(opt.workspace, 'backup'))
        if accelerator.is_main_process:
            torch.save(optimizer.state_dict(), os.path.join(opt.workspace, 'optimizer.pth'))
            torch.save(scheduler.state_dict(), os.path.join(opt.workspace, 'scheduler.pth'))
            with open(f'{opt.workspace}/metadata.json', 'w') as f:
                json.dump({'epoch': epoch}, f)

            torch.save(optimizer.state_dict(), os.path.join(opt.workspace, 'backup', 'optimizer.pth'))
            torch.save(scheduler.state_dict(), os.path.join(opt.workspace, 'backup', 'scheduler.pth'))
            with open(f'{opt.workspace}/backup/metadata.json', 'w') as f:
                json.dump({'epoch': epoch}, f)


        # eval
        with torch.no_grad():
            model.eval()
            total_psnr = 0
            for i, data in enumerate(test_dataloader):

                out = model(data)
    
                psnr = out['psnr']
                total_psnr += psnr.detach()
                
                # save some images
                if accelerator.is_main_process:
                    if '4d' in opt.data_mode:
                        B, T, V = opt.batch_size, opt.num_frames, opt.num_views

                        gt_images = data['images_output'].reshape(-1, T, V, *data['images_output'].shape[2:]).detach() # [B, V, 3, output_size, output_size]
                        pred_images = out['images_pred'].reshape(-1, T, V, *out['images_pred'].shape[2:]).detach() # [B, V, 3, output_size, output_size]

                        eval_gt_images = []
                        eval_pred_images = []
                        for t in range(T):
                            eval_gt_images_V = []
                            eval_pred_images_V = []
                            for v in range(V):
                                eval_gt_images_V.append((gt_images[:, t, v].permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
                                eval_pred_images_V.append((pred_images[:, t, v].permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
                            eval_gt_images.append(np.concatenate(eval_gt_images_V, axis=2))
                            eval_pred_images.append(np.concatenate(eval_pred_images_V, axis=2))
                        eval_gt_images = np.concatenate(eval_gt_images, axis=0)
                        eval_pred_images = np.concatenate(eval_pred_images, axis=0)
                        imageio.mimwrite(f'{opt.workspace}/eval_gt_images_{epoch}_{i}.mp4', eval_gt_images, fps=8)
                        imageio.mimwrite(f'{opt.workspace}/eval_pred_images_{epoch}_{i}.mp4', eval_pred_images, fps=8)
                        
                    elif '3d' in opt.data_mode:
                        gt_images = data['images_output'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(-1, gt_images.shape[1] * gt_images.shape[3], 3) # [B*output_size, V*output_size, 3]
                        kiui.write_image(f'{opt.workspace}/eval_gt_images_{epoch}_{i}.jpg', gt_images)

                        pred_images = out['images_pred'].detach().cpu().numpy() # [B, V, 3, output_size, output_size]
                        pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[3], 3)
                        kiui.write_image(f'{opt.workspace}/eval_pred_images_{epoch}_{i}.jpg', pred_images)
                    else:
                        raise NotImplementedError

            torch.cuda.empty_cache()

            total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
            if accelerator.is_main_process:
                total_psnr /= len(test_dataloader)
                accelerator.print(f"[eval] epoch: {epoch} psnr: {psnr:.4f}")



if __name__ == "__main__":
    main()
