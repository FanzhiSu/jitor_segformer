# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 11:25:01 2022

@author: Clark
"""
from utils import SegFormerDecoder, SegFormerSegmentationHead, MixVisionTransformer
import torch.nn as nn
from typing import List
import torch

class SegFormer(nn.Module):
    def __init__(
        self,
        in_chans: int,
        embed_dims: List[int],
        depths: List[int],
        num_heads: List[int],
        overlap_sizes: List[int],
        reduction_ratios: List[int],
        mlp_ratios: List[int],
        decoder_channels: int,
        sr_ratios: List[int],
        num_classes: int
    ):

        super().__init__()
        self.encoder = MixVisionTransformer(patch_size=4, in_chans = in_chans, num_classes = num_classes, embed_dims = embed_dims,
             num_heads = num_heads, mlp_ratios = mlp_ratios, depths = depths, sr_ratios = sr_ratios)
        self.decoder = SegFormerDecoder(decoder_channels, embed_dims[::-1], sr_ratios)
        self.head = SegFormerSegmentationHead(
            decoder_channels, num_classes, num_features=len(embed_dims)
        )

    def forward(self, x):
        features = self.encoder(x)
        features = self.decoder(features[::-1])
        segmentation = self.head(features)
        return segmentation
    
# segformer = SegFormer(
#         in_chans = 3,
#         embed_dims = [64, 128, 256, 512],
#         depths = [2, 2, 2, 2],
#         num_heads = [1, 2, 4, 8], # 5?
#         overlap_sizes = [4, 2, 2, 2],
#         reduction_ratios = [8, 4, 2, 1],
#         mlp_ratios = [4, 4, 4, 4],
#         decoder_channels = 256,
#         sr_ratios  =[8, 4, 2, 1],
#         num_classes = 100
# )

# segmentation = segformer(torch.randn((1, 3, 224, 224)))
# print(segmentation.shape) # torch.Size([1, 100, 56, 56])


# @BACKBONES.register_module()
# class mit_b0(MixVisionTransformer):
#     def __init__(self, **kwargs):
#         super(mit_b0, self).__init__(
#             patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)


# @BACKBONES.register_module()
# class mit_b1(MixVisionTransformer):
#     def __init__(self, **kwargs):
#         super(mit_b1, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)


# @BACKBONES.register_module()
# class mit_b2(MixVisionTransformer):
#     def __init__(self, **kwargs):
#         super(mit_b2, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)


# @BACKBONES.register_module()
# class mit_b3(MixVisionTransformer):
#     def __init__(self, **kwargs):
#         super(mit_b3, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)


# @BACKBONES.register_module()
# class mit_b4(MixVisionTransformer):
#     def __init__(self, **kwargs):
#         super(mit_b4, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)


# @BACKBONES.register_module()
# class mit_b5(MixVisionTransformer):
#     def __init__(self, **kwargs):
#         super(mit_b5, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1)