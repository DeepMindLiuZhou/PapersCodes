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

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Tuple, Literal
from functools import partial

from core.attention import MemEffAttention


class MVAttention(nn.Module):
    def __init__(
        self, 
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        groups: int = 32,
        eps: float = 1e-5,
        residual: bool = True,
        skip_scale: float = 1,
        num_views: int = 4, 
        num_frames: int = 8
    ):
        super().__init__()

        self.residual = residual
        self.skip_scale = skip_scale
        self.num_views = num_views
        self.num_frames = num_frames

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim, eps=eps, affine=True)
        self.attn = MemEffAttention(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)

    def forward(self, x):
        # x: [B*T*V, C, H, W]
        BTV, C, H, W = x.shape
        BT = BTV // self.num_views # assert BV % self.num_views == 0

        res = x
        x = self.norm(x)

        x = x.reshape(BT, self.num_views, C, H, W).permute(0, 1, 3, 4, 2).contiguous().reshape(BT, -1, C).contiguous()
        x = self.attn(x)
        x = x.reshape(BT, self.num_views, H, W, C).permute(0, 1, 4, 2, 3).contiguous().reshape(BTV, C, H, W).contiguous()

        if self.residual:
            x = (x + res) * self.skip_scale
        return x


class TempAttention(nn.Module):
    def __init__(
        self, 
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        groups: int = 32,
        eps: float = 1e-5,
        residual: bool = True,
        skip_scale: float = 1,
        num_views: int = 4,
        num_frames: int = 8
    ):
        super().__init__()

        self.residual = residual
        self.skip_scale = skip_scale
        self.num_views = num_views
        self.num_frames = num_frames

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=dim, eps=eps, affine=True)
        self.attn = MemEffAttention(dim, num_heads, qkv_bias, proj_bias, attn_drop, proj_drop)

    def forward(self, x):
        # x: [B*T*V, C, H, W]
        BTV, C, H, W = x.shape
        BV = BTV // self.num_frames # assert BV % self.num_views == 0
        B = BV // self.num_views

        res = x
        x = self.norm(x)

        # BTV -> BVT
        x = x.reshape(B, self.num_frames, self.num_views, C, H, W).permute(0, 2, 1, 3, 4, 5).contiguous()

        x = x.reshape(BV, self.num_frames, C, H, W).permute(0, 1, 3, 4, 2).contiguous().reshape(BV, -1, C).contiguous().contiguous()
        x = self.attn(x)
        x = x.reshape(BV, self.num_frames, H, W, C).permute(0, 1, 4, 2, 3).contiguous().reshape(BTV, C, H, W).contiguous().contiguous()

        # BVT -> BTV
        x = x.reshape(B, self.num_views, self.num_frames, C, H, W).permute(0, 2, 1, 3, 4, 5).contiguous().reshape(BTV, C, H, W).contiguous()

        if self.residual:
            x = (x + res) * self.skip_scale
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resample: Literal['default', 'up', 'down'] = 'default',
        groups: int = 32,
        eps: float = 1e-5,
        skip_scale: float = 1, # multiplied to output
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_scale = skip_scale

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.act = F.silu

        self.resample = None
        if resample == 'up':
            self.resample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
        elif resample == 'down':
            self.resample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    
    def forward(self, x):
        res = x

        x = self.norm1(x)
        x = self.act(x)

        if self.resample:
            res = self.resample(res)
            x = self.resample(x)
        
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = (x + self.shortcut(res)) * self.skip_scale

        return x

class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        downsample: bool = True,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
        num_views: int = 4,
        num_frames: int = 8,
        use_temp_attn=True
    ):
        super().__init__()
 
        nets = []
        attns = []
        t_attns = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            nets.append(ResnetBlock(in_channels, out_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention(out_channels, attention_heads, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames))
                t_attns.append(TempAttention(out_channels, attention_heads, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames) if use_temp_attn else None)
            else:
                attns.append(None)
                t_attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        self.t_attns = nn.ModuleList(t_attns)

        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        xs = []

        for attn, t_attn, net in zip(self.attns, self.t_attns, self.nets):
            x = net(x)
            if attn:
                x = attn(x)
            if t_attn:
                x = t_attn(x)
            xs.append(x)

        if self.downsample:
            x = self.downsample(x)
            xs.append(x)
  
        return x, xs


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
        num_views: int = 4,
        num_frames: int = 8,
        use_temp_attn=True
    ):
        super().__init__()

        nets = []
        attns = []
        t_attns = []
        # first layer
        nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
        # more layers
        for i in range(num_layers):
            nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention(in_channels, attention_heads, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames))
                t_attns.append(TempAttention(in_channels, attention_heads, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames) if use_temp_attn else None)
            else:
                attns.append(None)
                t_attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        self.t_attns = nn.ModuleList(t_attns)
        
    def forward(self, x):
        x = self.nets[0](x)
        for attn, t_attn,net in zip(self.attns, self.t_attns, self.nets[1:]):
            if attn:
                x = attn(x)
            if t_attn:
                x = t_attn(x)
            x = net(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_out_channels: int,
        out_channels: int,
        num_layers: int = 1,
        upsample: bool = True,
        attention: bool = True,
        attention_heads: int = 16,
        skip_scale: float = 1,
        num_views: int = 4,
        num_frames: int = 8,
        use_temp_attn=True
    ):
        super().__init__()

        nets = []
        attns = []
        t_attns = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            cskip = prev_out_channels if (i == num_layers - 1) else out_channels

            nets.append(ResnetBlock(cin + cskip, out_channels, skip_scale=skip_scale))
            if attention:
                attns.append(MVAttention(out_channels, attention_heads, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames))
                t_attns.append(TempAttention(out_channels, attention_heads, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames) if use_temp_attn else None)
            else:
                attns.append(None)
                t_attns.append(None)
        self.nets = nn.ModuleList(nets)
        self.attns = nn.ModuleList(attns)
        self.t_attns = nn.ModuleList(t_attns)

        self.upsample = None
        if upsample:
            self.upsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, xs):

        for attn, t_attn, net in zip(self.attns, self.t_attns, self.nets):
            res_x = xs[-1]
            xs = xs[:-1]
            x = torch.cat([x, res_x], dim=1)
            x = net(x)
            if attn:
                x = attn(x)
            if t_attn:
                x = t_attn(x)
            
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
            x = self.upsample(x)

        return x


# it could be asymmetric!
class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024),
        down_attention: Tuple[bool, ...] = (False, False, False, True, True),
        mid_attention: bool = True,
        up_channels: Tuple[int, ...] = (1024, 512, 256),
        up_attention: Tuple[bool, ...] = (True, True, False),
        layers_per_block: int = 2,
        skip_scale: float = np.sqrt(0.5),
        num_views: int = 4,
        num_frames: int = 8,
        use_temp_attn: bool = True
    ):
        super().__init__()

        # first
        self.conv_in = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, stride=1, padding=1)

        # down
        down_blocks = []
        cout = down_channels[0]
        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]

            down_blocks.append(DownBlock(
                cin, cout, 
                num_layers=layers_per_block, 
                downsample=(i != len(down_channels) - 1), # not final layer
                attention=down_attention[i],
                skip_scale=skip_scale,
                num_views=num_views, 
                num_frames=num_frames,
                use_temp_attn=use_temp_attn
            ))
        self.down_blocks = nn.ModuleList(down_blocks)

        # mid
        self.mid_block = MidBlock(down_channels[-1], attention=mid_attention, skip_scale=skip_scale, num_views=num_views, num_frames=num_frames, use_temp_attn=use_temp_attn)

        # up
        up_blocks = []
        cout = up_channels[0]
        for i in range(len(up_channels)):
            cin = cout
            cout = up_channels[i]
            cskip = down_channels[max(-2 - i, -len(down_channels))] # for assymetric

            up_blocks.append(UpBlock(
                cin, cskip, cout, 
                num_layers=layers_per_block + 1, # one more layer for up
                upsample=(i != len(up_channels) - 1), # not final layer
                attention=up_attention[i],
                skip_scale=skip_scale,
                num_views=num_views, 
                num_frames=num_frames,
                use_temp_attn=use_temp_attn
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

        # last
        self.norm_out = nn.GroupNorm(num_channels=up_channels[-1], num_groups=32, eps=1e-5)
        self.conv_out = nn.Conv2d(up_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x, return_mid_feature=False):
        # x: [B, Cin, H, W]

        # first
        x = self.conv_in(x)
        
        # down
        xss = [x]
        for block in self.down_blocks:
            x, xs = block(x)
            xss.extend(xs)
        
        # mid
        x = self.mid_block(x)
        mid_feature = (x, xss)

        # up
        for block in self.up_blocks:
            xs = xss[-len(block.nets):]
            xss = xss[:-len(block.nets)]
            x = block(x, xs)

        # last
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x) # [B, Cout, H', W']

        if return_mid_feature:
            return x, *mid_feature

        return x
