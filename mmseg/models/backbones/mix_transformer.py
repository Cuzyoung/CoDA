# Obtained from: https://github.com/NVlabs/SegFormer
# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
# A copy of the license is available at resources/license_segformer

import math
import warnings
from functools import partial

import torch
import torch.nn as nn
from mmcv.runner import BaseModule, _load_checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger 
from mmseg.models.utils.dacs_transforms import denorm
import matplotlib.pyplot as plt
import sys
import cv2
import numpy as np
import torch.nn.functional as F
import torch.nn.init as init

class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by ' \
                                     f'num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1,
                                                           3).contiguous()

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(
                                         2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).permute(
                                        2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better
        # than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)

        return x, H, W

class Adapter(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels):
        super(Adapter, self).__init__()

        # 投影到较小维度
        self.down_projection = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1)
        init.normal_(self.down_projection.weight, mean=0, std=0.01)
        # 非线性激活函数
        self.activation = nn.ReLU()
        # self.scene_aware_module = SceneAwareModule(bottleneck_channels)
        # 向上投射到原有维度
        self.up_projection = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1)
        init.normal_(self.up_projection.weight, mean=0, std=0.01)
    def forward(self, x):
        # 残差连接
        residual = x
        # 下投影
        x = self.down_projection(x)
        # 非线性激活函数
        # x = self.scene_aware_module(x)
        x = self.activation(x)
        x = self.up_projection(x)
        # 残差连接
        x += residual
        
        return x

class MetaLearner(nn.Module):
    def __init__(self, input_dim, mid_dim, output_dim, attention_dim):
        super(MetaLearner, self).__init__()
        self.attention = nn.MultiheadAttention(attention_dim, num_heads=4)
        # self.Meta_ResidualBlock = Meta_ResidualBlock(input_dim, output_dim)
        # two expert MLPs
        self.mlp_fog_rain_snow = Adapter(input_dim, mid_dim, output_dim)
        self.mlp_night = Adapter(input_dim, mid_dim, output_dim)

    def forward(self, x, condition):
        batch_size, channels, height, width = x.size()
        # x = x.view(batch_size, channels, -1).transpose(1, 2)
        # attn_output, _ = self.attention(x, x, x)
        # attn_output = attn_output.transpose(1, 2).view(batch_size, channels, height, width)
        attn_output = x
        new_attn_output = torch.zeros_like(attn_output)
        # 根据天气条件选择相应的MLP层
        for i in range(len(condition)):
            # attn_output_flat = attn_output[i].view(channels, -1).transpose(0, 1)
            if condition[i] == 'easy':
                new_attn_output[i] = self.mlp_fog_rain_snow(attn_output[i].unsqueeze(0)).squeeze(0)
            elif condition[i] == 'hard':
                new_attn_output[i] = self.mlp_night(attn_output[i].unsqueeze(0)).squeeze(0)
            else:
                raise NotImplementedError
            
        return new_attn_output
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()

            # Convolutional layer to process each prompt region
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        # init.normal_(self.conv.weight, mean=0, std=0.01)
            # Softmax layer to generate attention weights
        self.softmax = nn.Softmax(dim=-1)
        self.count = 0
    def forward(self, x, visualization=False):
        # x is the input feature map containing prompt regions

        # Process each prompt region with convolutional layer
        attention_weights = self.conv(x)

        # Apply softmax to get attention weights
        attention_weights = self.softmax(attention_weights)
        # print('attention_weights.shape:', attention_weights.shape)

        attention_weights_save = attention_weights
        if visualization:
            attention_weights_save = attention_weights_save.detach().cpu().numpy()
            attention_weights_save = attention_weights_save[0, 0]
            # print('attention_weights_save.shape:', attention_weights_save.shape)
            # sys.exit()
            # attention_weights_save = attention_weights_save * 255
            
            if self.count % 2 ==0:
                plt.imshow(attention_weights_save, cmap='viridis')  # viridis
                plt.savefig(f"/share/home/dq070/CVPR_HRDA/MIC/seg/workdir/gta2cs/source_prompt_64size_1block_fixed_corner_center_adapter/visualization_adapter_stage1/attention_weights{self.count}.png", bbox_inches='tight', pad_inches=0)
                plt.close()
            self.count += 1

        # Multiply input feature map by attention weights
        weighted_imgs = x * attention_weights
        # Sum along the channel dimension to get the final output

        return weighted_imgs
@BACKBONES.register_module()
class MixVisionTransformer(BaseModule):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 style=None,
                 pretrained=None,
                 init_cfg=None,
                 freeze_patch_embed=False):
        super().__init__(init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str) or pretrained is None:
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
        else:
            raise TypeError('pretrained must be a str or None')

        self.num_classes = num_classes
        self.depths = depths
        self.pretrained = pretrained
        self.init_cfg = init_cfg
        self.MetaLearner = MetaLearner(input_dim=64, mid_dim=16 ,output_dim=64, attention_dim=64).train()
        self.AttentionModule_easy = AttentionModule(in_channels=64).train()
        self.AttentionModule_hard = AttentionModule(in_channels=64).train()
        self.LayerNorm = nn.LayerNorm(128).train()
        
        self.initial = torch.rand(3, 64, 64).cuda()
        self.initial_pad = torch.rand(3, 512, 64).cuda()
        self.initial_pad2 = torch.rand(3, 64, 512).cuda()
        self.prompt = nn.parameter.Parameter(torch.zeros_like(self.initial).cuda(), requires_grad=True)
        self.prompt1 = nn.parameter.Parameter(torch.zeros_like(self.initial).cuda(), requires_grad=True)
        self.prompt2 = nn.parameter.Parameter(torch.zeros_like(self.initial).cuda(), requires_grad=True)
        self.prompt3 = nn.parameter.Parameter(torch.zeros_like(self.initial).cuda(), requires_grad=True)
        self.prompt4 = nn.parameter.Parameter(torch.zeros_like(self.initial).cuda(), requires_grad=True)

        self.prompth = nn.parameter.Parameter(torch.zeros_like(self.initial).cuda(), requires_grad=True)
        self.prompt1h = nn.parameter.Parameter(torch.zeros_like(self.initial).cuda(), requires_grad=True)
        self.prompt2h = nn.parameter.Parameter(torch.zeros_like(self.initial).cuda(), requires_grad=True)
        self.prompt3h = nn.parameter.Parameter(torch.zeros_like(self.initial).cuda(), requires_grad=True)
        self.prompt4h = nn.parameter.Parameter(torch.zeros_like(self.initial).cuda(), requires_grad=True)

        # self.prompt = nn.parameter.Parameter(torch.rand(3, 64, 64).cuda(), requires_grad=True)
        # self.prompt1 = nn.parameter.Parameter(torch.rand(3, 64, 64).cuda(), requires_grad=True)
        # self.prompt2 = nn.parameter.Parameter(torch.rand(3, 64, 64).cuda(), requires_grad=True)
        # self.prompt3 = nn.parameter.Parameter(torch.rand(3, 64, 64).cuda(), requires_grad=True)
        # self.prompt4 = nn.parameter.Parameter(torch.rand(3, 64, 64).cuda(), requires_grad=True)
        # self.prompth = nn.parameter.Parameter(torch.rand(3, 64, 64).cuda(), requires_grad=True)
        # self.prompt1h = nn.parameter.Parameter(torch.rand(3, 64, 64).cuda(), requires_grad=True)
        # self.prompt2h = nn.parameter.Parameter(torch.rand(3, 64, 64).cuda(), requires_grad=True)
        # self.prompt3h = nn.parameter.Parameter(torch.rand(3, 64, 64).cuda(), requires_grad=True)
        # self.prompt4h = nn.parameter.Parameter(torch.rand(3, 64, 64).cuda(), requires_grad=True)
        
        # self.prompt = nn.parameter.Parameter(torch.ones_like(self.initial).cuda(), requires_grad=True)
        # self.prompt1 = nn.parameter.Parameter(torch.ones_like(self.initial).cuda(), requires_grad=True)
        # self.prompt2 = nn.parameter.Parameter(torch.ones_like(self.initial).cuda(), requires_grad=True)
        # self.prompt3 = nn.parameter.Parameter(torch.ones_like(self.initial).cuda(), requires_grad=True)
        # self.prompt4 = nn.parameter.Parameter(torch.ones_like(self.initial).cuda(), requires_grad=True)
        # self.prompth = nn.parameter.Parameter(torch.ones_like(self.initial).cuda(), requires_grad=True)
        # self.prompt1h = nn.parameter.Parameter(torch.ones_like(self.initial).cuda(), requires_grad=True)
        # self.prompt2h = nn.parameter.Parameter(torch.ones_like(self.initial).cuda(), requires_grad=True)
        # self.prompt3h = nn.parameter.Parameter(torch.ones_like(self.initial).cuda(), requires_grad=True)
        # self.prompt4h = nn.parameter.Parameter(torch.ones_like(self.initial).cuda(), requires_grad=True)

        # 可视化使用
        # self.promptv = nn.parameter.Parameter(torch.rand(3, 64, 64).cuda(), requires_grad=True)
        # self.prompt1v = nn.parameter.Parameter(torch.rand(3, 64, 64).cuda(), requires_grad=True)
        # self.prompt2v = nn.parameter.Parameter(torch.rand(3, 64, 64).cuda(), requires_grad=True)
        # self.prompt3v = nn.parameter.Parameter(torch.rand(3, 64, 64).cuda(), requires_grad=True)
        # self.prompt4h = nn.parameter.Parameter(torch.rand(3, 64, 64).cuda(), requires_grad=True)

        # self.MetaLearner2 = MetaLearner(input_dim=128, output_dim=128, attention_dim=128)
        # self.MetaLearner3 = MetaLearner(input_dim=320, output_dim=320, attention_dim=320)
        # self.MetaLearner4 = MetaLearner(input_dim=512, output_dim=512, attention_dim=512)
        # self.initial_pad = torch.rand(3, 512, 64).cuda()
        # self.initial_pad2 = torch.rand(3, 64, 512).cuda()

        # self.prompt_pad = nn.parameter.Parameter(torch.rand(3, 512,64).cuda(), requires_grad=True)
        # self.prompt_pad2 = nn.parameter.Parameter(torch.rand(3, 512,64).cuda(), requires_grad=True)
        # self.prompt_pad3 = nn.parameter.Parameter(torch.rand(3, 64,512).cuda(), requires_grad=True)
        # self.prompt_pad4 = nn.parameter.Parameter(torch.rand(3, 64,512).cuda(), requires_grad=True)
        # self.prompt_padh = nn.parameter.Parameter(torch.rand(3, 512,64).cuda(), requires_grad=True)
        # self.prompt_padh2 = nn.parameter.Parameter(torch.rand(3, 512,64).cuda(), requires_grad=True)
        # self.prompt_padh3 = nn.parameter.Parameter(torch.rand(3, 64,512).cuda(), requires_grad=True)
        # self.prompt_padh4 = nn.parameter.Parameter(torch.rand(3, 64,512).cuda(), requires_grad=True)

        # self.prompt_pad = nn.parameter.Parameter(torch.zeros_like(self.initial_pad).cuda(), requires_grad=True)
        # self.prompt_pad2 = nn.parameter.Parameter(torch.zeros_like(self.initial_pad).cuda(), requires_grad=True)
        # self.prompt_pad3 = nn.parameter.Parameter(torch.zeros_like(self.initial_pad2).cuda(), requires_grad=True)
        # self.prompt_pad4 = nn.parameter.Parameter(torch.zeros_like(self.initial_pad2).cuda(), requires_grad=True)
        # self.prompt_padh = nn.parameter.Parameter(torch.zeros_like(self.initial_pad).cuda(), requires_grad=True)
        # self.prompt_padh2 = nn.parameter.Parameter(torch.zeros_like(self.initial_pad).cuda(), requires_grad=True)
        # self.prompt_padh3 = nn.parameter.Parameter(torch.zeros_like(self.initial_pad2).cuda(), requires_grad=True)
        # self.prompt_padh4 = nn.parameter.Parameter(torch.zeros_like(self.initial_pad2).cuda(), requires_grad=True)

        # self.prompt_pad = nn.parameter.Parameter(torch.ones_like(self.initial_pad).cuda(), requires_grad=True)
        # self.prompt_pad2 = nn.parameter.Parameter(torch.ones_like(self.initial_pad).cuda(), requires_grad=True)
        # self.prompt_pad3 = nn.parameter.Parameter(torch.ones_like(self.initial_pad2).cuda(), requires_grad=True)
        # self.prompt_pad4 = nn.parameter.Parameter(torch.ones_like(self.initial_pad2).cuda(), requires_grad=True)
        # self.prompt_padh = nn.parameter.Parameter(torch.ones_like(self.initial_pad).cuda(), requires_grad=True)
        # self.prompt_padh2 = nn.parameter.Parameter(torch.ones_like(self.initial_pad).cuda(), requires_grad=True)
        # self.prompt_padh3 = nn.parameter.Parameter(torch.ones_like(self.initial_pad2).cuda(), requires_grad=True)
        # self.prompt_padh4 = nn.parameter.Parameter(torch.ones_like(self.initial_pad2).cuda(), requires_grad=True)

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size,
            patch_size=7,
            stride=4,
            in_chans=in_chans,
            embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16,
            patch_size=3,
            stride=2,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3])
        if freeze_patch_embed:
            self.freeze_patch_emb()

        # transformer encoder
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([
            Block(
                dim=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[0]) for i in range(depths[0])
        ])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[1]) for i in range(depths[1])
        ])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[2]) for i in range(depths[2])
        ])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[3]) for i in range(depths[3])
        ])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) \
        #     if num_classes > 0 else nn.Identity()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self):
        logger = get_root_logger()
        if self.pretrained is None:
            logger.info('Init mit from scratch.')
            for m in self.modules():
                self._init_weights(m)
        elif isinstance(self.pretrained, str):
            logger.info('Load mit checkpoint.')
            checkpoint = _load_checkpoint(
                self.pretrained, logger=logger, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            self.load_state_dict(state_dict, False)
    
    def reset_drop_path(self, drop_path_rate):
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(self.depths))
        ]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'
        }  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def locate_levels(self, target_img):
        # denorm
        means, stds = 114.495, 57.63
        target_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
        # Save target_img
        target_img = torch.Tensor.cpu(target_img)
        target_img = np.array(target_img)
        # plt.imsave('/share/home/dq070/CoT/MIC-metawithdot/seg/visualization/target_img.png', target_img[0].transpose(1, 2, 0))
        # sys.exit()
        levels = [0]*2
        # to np.array
        # target_img = torch.Tensor.cpu(target_img)
        # target_img = np.array(target_img)

        # switch channel to third dimension
        target_img1 = target_img[0]
        target_img2 = target_img[1]
        del target_img
        target_img1 = np.transpose(target_img1, (1, 2, 0))
        target_img2 = np.transpose(target_img2, (1, 2, 0))
        target_img1_ori = target_img1
        # First img
        gray_img = cv2.cvtColor(target_img1, cv2.COLOR_BGR2GRAY)
    
        # print('gray_img.shape:', gray_img.shape)
        # sys.exit()
        dark_sum = 0  
        dark_threshold = 40 / 255
        # plt.imsave('/share/home/dq070/CoT/MIC-metawithdot/seg/visualization/target_img1_gray.png', gray_img, cmap='gray')
        # for row in range(0, 512):
        #     for colum in range(0, 512):
        #         if gray_img[row, colum] < dark_threshold:  # gray value < 40 is dark
        #             dark_sum += 1
        #             target_img1_ori[row, colum, 0] = 72/255
        #             target_img1_ori[row, colum, 1] = 27/255
        #             target_img1_ori[row, colum, 2] = 84/255
        #         else:
        #             target_img1_ori[row, colum, 0] = 206/255   
        #             target_img1_ori[row, colum, 1] = 231/255
        #             target_img1_ori[row, colum, 2] = 95/255
        # target_img1_ori = target_img1_ori * 255
        # plt.imsave('/share/home/dq070/CoT/MIC-metawithdot/seg/visualization/target_img1_grayv.png', target_img1_ori)
        # sys.exit()
        dark_sum = np.sum(gray_img < dark_threshold)
        dark_prop = dark_sum / (gray_img.size)
        if dark_prop >= 0.05:
            levels[0] = 'hard'
        else:
            levels[0] = 'easy'
        # Second img
        gray_img2 = cv2.cvtColor(target_img2, cv2.COLOR_BGR2GRAY)
        dark_sum = 0   
        dark_sum = np.sum(gray_img2 < dark_threshold)
        dark_prop = dark_sum / (gray_img2.size)
        if dark_prop >= 0.05:
            levels[1] = 'hard'
        else:
            levels[1] = 'easy'
        return levels


    def forward_features(self, x, levels=None):
        B = x.shape[0]
        outs = []
        
        if levels:
            prompt_size = self.prompt.shape[-1]
            # print('prompt_size:', prompt_size)
            size = x[0].shape[-1]
            start_h_left_up = 448
            start_w_left_up = 0
            start_h_left_down = 0
            start_w_left_down = 0
            start_h_right_up = 448
            start_w_right_up = 448
            start_h_right_down = 0
            start_w_right_down = 448
            start_h_center = 224
            start_w_center = 224
            for i in range(2):
                if levels[i] == 'easy':
                    x[i, :, start_h_left_down:start_h_left_down+prompt_size, start_w_left_down:start_w_left_down+prompt_size] += self.prompt
                    x[i, :, start_h_left_up:start_h_left_up+prompt_size, start_w_left_up:start_w_left_up+prompt_size] += self.prompt1
                    x[i, :, start_h_right_down:start_h_right_down+prompt_size, start_w_right_down:start_w_right_down+prompt_size] += self.prompt2
                    x[i, :, start_h_right_up:start_h_right_up+prompt_size, start_w_right_up:start_w_right_up+prompt_size] += self.prompt3
                    x[i, :, start_h_center:start_h_center+prompt_size, start_w_center:start_w_center+prompt_size] += self.prompt4
                else:
                    x[i, :, start_h_left_down:start_h_left_down+prompt_size, start_w_left_down:start_w_left_down+prompt_size] += self.prompth
                    x[i, :, start_h_left_up:start_h_left_up+prompt_size, start_w_left_up:start_w_left_up+prompt_size] += self.prompt1h
                    x[i, :, start_h_right_down:start_h_right_down+prompt_size, start_w_right_down:start_w_right_down+prompt_size] += self.prompt2h
                    x[i, :, start_h_right_up:start_h_right_up+prompt_size, start_w_right_up:start_w_right_up+prompt_size] += self.prompt3h
                    x[i, :, start_h_center:start_h_center+prompt_size, start_w_center:start_w_center+prompt_size] += self.prompt4h
                # x_save = x.clone()
                # x_save = torch.clamp(denorm(x_save, 114.495, 57.63), 0, 1)
                # x_save = torch.Tensor.cpu(x_save)
                # x_save = np.array(x_save)
                # plt.imsave('/share/home/dq070/CoT/MIC-metawithdot/seg/visualization/center_vp_nor.png', x_save[0].transpose(1, 2, 0))
                # sys.exit()    
        # Add prompt blocks to x
        
        # if levels:
        #     # prompt_size = self.prompt.shape[-1]
        #     # print('prompt_size:', prompt_size)
        #     size = x[0].shape[-1]
        #     h = self.prompt_pad.shape[-1]
        #     for i in range(2):
        #         if levels[i] == 'easy':
        #             x[i, :, :, 0:h] += self.prompt_pad
        #             x[i, :, :, size-h:size] += self.prompt_pad2
        #             x[i, :, 0:h, :] += self.prompt_pad3
        #             x[i, :, size-h:size, :] += self.prompt_pad4
        #         else:
        #             x[i, :, :, 0:h] += self.prompt_padh
        #             x[i, :, :, size-h:size] += self.prompt_padh2
        #             x[i, :, 0:h, :] += self.prompt_padh3
        #             x[i, :, size-h:size, :] += self.prompt_padh4
            
        # Randomly add prompt blocks to x
        # if levels:
        #     prompt_size = self.prompt.shape[-1]
        #     for i in range(2):
        #         if levels[i] == 'easy':
        #             # Randomly select positions to add prompt blocks
        #             start_h_left_down = torch.randint(0, x.size(2) - prompt_size + 1, (1,))
        #             start_w_left_down = torch.randint(0, x.size(3) - prompt_size + 1, (1,))
        #             start_h_left_up = torch.randint(0, x.size(2) - prompt_size + 1, (1,))
        #             start_w_left_up = torch.randint(0, x.size(3) - prompt_size + 1, (1,))
        #             start_h_right_down = torch.randint(0, x.size(2) - prompt_size + 1, (1,))
        #             start_w_right_down = torch.randint(0, x.size(3) - prompt_size + 1, (1,))
        #             start_h_right_up = torch.randint(0, x.size(2) - prompt_size + 1, (1,))
        #             start_w_right_up = torch.randint(0, x.size(3) - prompt_size + 1, (1,))
        #             start_h_center = torch.randint(0, x.size(2) - prompt_size + 1, (1,))
        #             start_w_center = torch.randint(0, x.size(3) - prompt_size + 1, (1,))
        #             # Add prompt blocks to x
        #             x[i, :, start_h_left_down:start_h_left_down+prompt_size, start_w_left_down:start_w_left_down+prompt_size] += self.prompt
        #             x[i, :, start_h_left_up:start_h_left_up+prompt_size, start_w_left_up:start_w_left_up+prompt_size] += self.prompt1
        #             x[i, :, start_h_right_down:start_h_right_down+prompt_size, start_w_right_down:start_w_right_down+prompt_size] += self.prompt2
        #             x[i, :, start_h_right_up:start_h_right_up+prompt_size, start_w_right_up:start_w_right_up+prompt_size] += self.prompt3
        #             x[i, :, start_h_center:start_h_center+prompt_size, start_w_center:start_w_center+prompt_size] += self.prompt4
        #         else:
        #             # Randomly select positions to add prompt blocks
        #             start_h_left_down = torch.randint(0, x.size(2) - prompt_size + 1, (1,))
        #             start_w_left_down = torch.randint(0, x.size(3) - prompt_size + 1, (1,))
        #             start_h_left_up = torch.randint(0, x.size(2) - prompt_size + 1, (1,))
        #             start_w_left_up = torch.randint(0, x.size(3) - prompt_size + 1, (1,))
        #             start_h_right_down = torch.randint(0, x.size(2) - prompt_size + 1, (1,))
        #             start_w_right_down = torch.randint(0, x.size(3) - prompt_size + 1, (1,))
        #             start_h_right_up = torch.randint(0, x.size(2) - prompt_size + 1, (1,))
        #             start_w_right_up = torch.randint(0, x.size(3) - prompt_size + 1, (1,))
        #             start_h_center = torch.randint(0, x.size(2) - prompt_size + 1, (1,))
        #             start_w_center = torch.randint(0, x.size(3) - prompt_size + 1, (1,))
        #             # Add prompt blocks to x
        #             x[i, :, start_h_left_down:start_h_left_down+prompt_size, start_w_left_down:start_w_left_down+prompt_size] += self.prompth
        #             x[i, :, start_h_left_up:start_h_left_up+prompt_size, start_w_left_up:start_w_left_up+prompt_size] += self.prompt1h
        #             x[i, :, start_h_right_down:start_h_right_down+prompt_size, start_w_right_down:start_w_right_down+prompt_size] += self.prompt2h
        #             x[i, :, start_h_right_up:start_h_right_up+prompt_size, start_w_right_up:start_w_right_up+prompt_size] += self.prompt3h
        #             x[i, :, start_h_center:start_h_center+prompt_size, start_w_center:start_w_center+prompt_size] += self.prompt4h
        #         x_save = x.clone()
        #         x_save = torch.clamp(denorm(x_save, 114.495, 57.63), 0, 1)
        #         x_save = torch.Tensor.cpu(x_save)
        #         x_save = np.array(x_save)
        #         plt.imsave('/share/home/dq070/CoT/MIC-metawithdot/seg/visualization/random_vp_1_final.png', x_save[0].transpose(1, 2, 0))
        #         sys.exit()    

            
                
        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        new_x = torch.zeros_like(x)
        new_x2 = torch.zeros_like(x)
        # new_x = x 
        if levels:
            new_x = self.MetaLearner(x, levels)
            for i in range(2):
                if levels[i] == 'easy':
                    new_x2[i] = self.AttentionModule_easy(new_x[i].unsqueeze(0)).squeeze(0)
                elif levels[i] == 'hard':
                    new_x2[i] = self.AttentionModule_hard(new_x[i].unsqueeze(0)).squeeze(0)
            new_x2 = self.LayerNorm(new_x2)
            # x = new_x2
        outs.append(new_x2)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x, levels=None):
        level = 1
        if levels:
            levels = self.locate_levels(x)
        x = self.forward_features(x, levels)
        # x = self.head(x)

        return x


class DWConv(nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        return x


@BACKBONES.register_module()
class mit_b0(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b0, self).__init__(
            patch_size=4,
            embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class mit_b1(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b1, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class mit_b2(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b2, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class mit_b3(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b3, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class mit_b4(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b4, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 8, 27, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)


@BACKBONES.register_module()
class mit_b5(MixVisionTransformer):

    def __init__(self, **kwargs):
        super(mit_b5, self).__init__(
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1],
            **kwargs)
