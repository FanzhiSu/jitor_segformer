# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 15:32:56 2022

@author: Clark
"""
from torchvision.datasets import Cityscapes
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from utils import encode_segmap, decode_segmap

dataset = Cityscapes('./data/', split='train', mode='fine',target_type='semantic')


# dataset[0][0].size

# fig,ax=plt.subplots(ncols=2,figsize=(12,8))
# ax[0].imshow(dataset[0][0])
# ax[1].imshow(dataset[0][1],cmap='gray')




ignore_index=255
void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [ignore_index,7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', \
               'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', \
               'train', 'motorcycle', 'bicycle'] # 19 classes + 1 unlablled(background)
#why 20 classes
#https://stackoverflow.com/a/64242989

class_map = dict(zip(valid_classes, range(len(valid_classes))))
n_classes=len(valid_classes)
# print(class_map)



colors = [   [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

label_colours = dict(zip(range(n_classes), colors))



# def encode_segmap(mask):
#     #remove unwanted classes and recitify the labels of wanted classes
#     for _voidc in void_classes:
#         mask[mask == _voidc] = ignore_index
#     for _validc in valid_classes:
#         mask[mask == _validc] = class_map[_validc]
#     return mask



# def decode_segmap(temp):
#     #convert grey scale to color
#     temp=temp.numpy()
#     r = temp.copy()
#     g = temp.copy()
#     b = temp.copy()
#     for l in range(0, n_classes):
#         r[temp == l] = label_colours[l][0]
#         g[temp == l] = label_colours[l][1]
#         b[temp == l] = label_colours[l][2]

#     rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
#     rgb[:, :, 0] = r / 255.0
#     rgb[:, :, 1] = g / 255.0
#     rgb[:, :, 2] = b / 255.0
#     return rgb


from torch.utils.data import Dataset
import os
from PIL import Image
from transformers import SegformerFeatureExtractor

class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, root_dir, feature_extractor, train=True):
        """
        Args:
            root_dir (string): Root directory of the dataset containing the images + annotations.
            feature_extractor (SegFormerFeatureExtractor): feature extractor to prepare images + segmentation maps.
            train (bool): Whether to load "training" or "validation" images + annotations.
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.train = train

        sub_path = "training" if self.train else "validation"
        self.img_dir = os.path.join(self.root_dir, "images", sub_path)
        self.ann_dir = os.path.join(self.root_dir, "annotations", sub_path)
        
        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
          image_file_names.extend(files)
        self.images = sorted(image_file_names)
        
        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
          annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        segmentation_map = Image.open(os.path.join(self.ann_dir, self.annotations[idx]))

        # randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k,v in encoded_inputs.items():
          encoded_inputs[k].squeeze_() # remove batch dimension

        return encoded_inputs


# dataset=MyClass('./data/', split='val', mode='fine',
#                      target_type='semantic',transforms=transform)
# img,seg= dataset[20]
# # print(img.shape,seg.shape)



# # fig,ax=plt.subplots(ncols=2,nrows=1,figsize=(16,8))
# # ax[0].imshow(img.permute(1, 2, 0))
# # ax[1].imshow(seg,cmap='gray')


# #class labels before label correction
# print(torch.unique(seg))
# print(len(torch.unique(seg)))


# #class labels after label correction
# encode =  encode_segmap(void_classes = void_classes, valid_classes = valid_classes, class_map = class_map)
# res = encode.Encode_segmap(mask = seg.clone())
# # res=encode_segmap(seg.clone())
# print(res.shape)
# print(torch.unique(res))
# print(len(torch.unique(res)))


# #from grey to color
# decode = decode_segmap(label_colours = label_colours, n_classes = n_classes)
# res1 = decode.Decode_segmap(temp = res.clone())
# # res1=decode_segmap(res.clone())

# fig,ax=plt.subplots(ncols=2,figsize=(12,10))  
# ax[0].imshow(res,cmap='gray')
# ax[1].imshow(res1)






