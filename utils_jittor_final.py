

import jittor as jt
from jittor import nn
from typing import List
import collections.abc
from itertools import repeat




class DWConv(nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv(dim, dim, 3, stride=1, padding=1, bias=True, groups=dim)

    def execute(self, x, H, W):
        (B, N, C) = x.shape
        x = x.transpose(1, 2).view((B, C, H, W))
        x = self.dwconv(x)
        x = x.flatten(start_dim=2).transpose(1, 2)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + jt.rand(shape, dtype=x.dtype)
    random_tensor.floor()  # binarize
    output = x / keep_prob * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        return drop_path(x, self.drop_prob, self.is_training())




def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)







class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.gelu, drop=0.0):
        super().__init__()
        out_features = (out_features or in_features)
        hidden_features = (hidden_features or in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)


    def execute(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert ((dim % num_heads) == 0), f'dim {dim} should be divisible by num_heads {num_heads}.'
        self.dim = dim
        self.num_heads = num_heads
        head_dim = (dim // num_heads)
        self.scale = (qk_scale or (head_dim ** (- 0.5)))
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, (dim * 2), bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if (sr_ratio > 1):
            self.sr = nn.Conv(dim, dim, sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def execute(self, x, H, W):
        (B, N, C) = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, (C // self.num_heads)).permute((0, 2, 1, 3))
        if (self.sr_ratio > 1):
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)

            x_ = self.sr(x_).reshape(B, C, (- 1)).permute((0, 2, 1))
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, (- 1), 2, self.num_heads, (C // self.num_heads)).permute((2, 0, 3, 1, 4))
        else:
            kv = self.kv(x).reshape(B, (- 1), 2, self.num_heads, (C // self.num_heads)).permute((2, 0, 3, 1, 4))
        (k, v) = (kv[0], kv[1])
        attn = ((q @ k.transpose((- 2), (- 1))) * self.scale)
        attn = attn.softmax(dim=(- 1))
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, act_layer= nn.gelu, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if (drop_path > 0.0) else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int((dim * mlp_ratio))
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def execute(self, x, H, W):
        x = (x + self.drop_path(self.attn(self.norm1(x), H, W)))
        x = (x + self.drop_path(self.mlp(self.norm2(x), H, W)))
        return x

class OverlapPatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        (self.H, self.W) = ((img_size[0] // patch_size[0]), (img_size[1] // patch_size[1]))
        self.num_patches = (self.H * self.W)
        self.proj = nn.Conv(in_chans, embed_dim, patch_size, stride=stride, padding=((patch_size[0] // 2), (patch_size[1] // 2)))
        self.norm = nn.LayerNorm(embed_dim)

    def execute(self, x):
        x = self.proj(x)
        (_, _, H, W) = x.shape
        x = x.flatten(start_dim=2).transpose(0, 1, 2)
        x = self.norm(x)
        return (x, H, W)

class MixVisionTransformer(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, 
                 num_classes=1000, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], 
                 qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=(img_size // 4), patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=(img_size // 8), patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=(img_size // 16), patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        dpr = [x.item() for x in jt.misc.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, 
                                           qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[(cur + i)], norm_layer=norm_layer, sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, 
                                           qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[(cur + i)], norm_layer=norm_layer, sr_ratio=sr_ratios[1]) for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, 
                                           qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[(cur + i)], norm_layer=norm_layer, sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, 
                                           qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[(cur + i)], norm_layer=norm_layer, sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])



    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in jt.misc.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[(cur + i)]
        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[(cur + i)]
        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[(cur + i)]
        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[(cur + i)]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    # @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x):
        B = x.shape[0]
        outs = []
        (x, H, W) = self.patch_embed1(x)
        for (i, blk) in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, (- 1)).permute((0, 3, 1, 2)).contiguous()
        outs.append(x)
        (x, H, W) = self.patch_embed2(x)
        for (i, blk) in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, (- 1)).permute((0, 3, 1, 2)).contiguous()
        outs.append(x)
        (x, H, W) = self.patch_embed3(x)
        for (i, blk) in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, (- 1)).permute((0, 3, 1, 2)).contiguous()
        outs.append(x)
        (x, H, W) = self.patch_embed4(x)
        for (i, blk) in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, (- 1)).permute((0, 3, 1, 2)).contiguous()
        outs.append(x)
        return outs

    def execute(self, x):
        x = self.forward_features(x)
        return x

class SegFormerDecoderBlock(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, sr_ratios: int=2):
        super().__init__(nn.Conv(in_channels, out_channels, 1), 
                         nn.UpsamplingBilinear2d(scale_factor=sr_ratios))
       

class SegFormerDecoder(nn.Module):

    def __init__(self, out_channels: int, embed_dims: List[int], sr_ratios: List[int]):
        super().__init__()
        self.stages = nn.ModuleList([SegFormerDecoderBlock(in_channels, out_channels, sr_ratios) for (in_channels, sr_ratios) in zip(embed_dims, sr_ratios)])

    def execute(self, features):
        new_features = []
        for (feature, stage) in zip(features, self.stages):
            x = stage(feature)
            new_features.append(x)
        return new_features

class SegFormerSegmentationHead(nn.Module):

    def __init__(self, channels: int, num_classes: int, num_features: int=4):
        super().__init__()
        self.fuse = nn.Sequential(nn.Conv((channels * num_features), channels, 1, bias=False), nn.BatchNorm(channels))
        self.predict = nn.Conv(channels, num_classes, 1)

    def execute(self, features):
        x = jt.contrib.concat(features, dim=1)
        x = self.fuse(x)
        x = self.predict(x)
        x = nn.interpolate(x, size=[(x.shape[2] * 4), (x.shape[3] * 4)], mode='bilinear', align_corners=False)
        return x