# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 23:30:27 2022

@author: Clark
"""
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import SegFormer
from utils import decode_segmap, encode_segmap
from dataset import MyClass
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
transform=A.Compose(
[
    A.Resize(256, 512),
    A.HorizontalFlip(),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
]
)


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
train_class =  MyClass ('./data/', split='train', mode='fine',
                     target_type='semantic',transforms=transform)
val_class =  MyClass ('./data/', split='val', mode='fine',
                     target_type='semantic',transforms=transform)




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
encode =  encode_segmap(void_classes = void_classes, valid_classes = valid_classes, class_map = class_map)



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
def main():
    model = SegFormer(
            in_chans = 3,
            embed_dims = [64, 128, 256, 512],
            depths = [2, 2, 2, 2],
            num_heads = [1, 2, 4, 8], # 5?
            overlap_sizes = [4, 2, 2, 2],
            reduction_ratios = [8, 4, 2, 1],
            mlp_ratios = [4, 4, 4, 4],
            decoder_channels = 256,
            sr_ratios  =[8, 4, 2, 1],
            num_classes = 20).to(DEVICE)
    
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader = DataLoader(train_class, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS, pin_memory=True )
    val_loader = DataLoader(val_class, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS, pin_memory=True )



    # check_accuracy(val_loader, model, device=DEVICE)
    # scaler = torch.cuda.amp.GradScaler()
    

    # keeping-track-of-losses 
    train_losses = []
    valid_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        # keep-track-of-training-and-validation-loss
        train_loss = 0.0
        valid_loss = 0.0
        # training-the-model
        model.train()

        for data, segment in train_loader:
            # move-tensors-to-GPU 
            data = data.to(DEVICE)
            encode =  encode_segmap(void_classes = void_classes, valid_classes = valid_classes, class_map = class_map)
            segment = encode.Encode_segmap(mask = segment.clone())
            segment = segment.to(DEVICE)
            # clear-the-gradients-of-all-optimized-variables
            optimizer.zero_grad()
            # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
            output = model(data)
            # calculate-the-batch-loss
            loss = loss_fn(output, segment)
            # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
            loss.backward()
            # perform-a-ingle-optimization-step (parameter-update)
            optimizer.step()
            # update-training-loss
            train_loss += loss.item() * data.size(0)

       

        # validate-the-model
        model.eval()
        for data, target in val_loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)
            output = model(data)
            loss = loss_fn(output, target)
            
        # update-average-validation-loss 
            valid_loss += loss.item() * data.size(0)

        # calculate-average-losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(val_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)    

    # print-training/validation-statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(

        epoch, train_loss, valid_loss))

    # for epoch in range(NUM_EPOCHS):
    #     train_fn(train_loader, model, optimizer, loss_fn, scaler)

    #     # save model
    #     checkpoint = {
    #         "state_dict": model.state_dict(),
    #         "optimizer":optimizer.state_dict(),
    #     }
    #     save_checkpoint(checkpoint)

    #     # check accuracy
    #     # check_accuracy(val_loader, model, device=DEVICE)

    #     # print some examples to a folder
    #     save_predictions_as_imgs(
    #         val_loader, model, folder="saved_images/", device=DEVICE
    #     )
    # test-the-model

    model.eval()  # it-disables-dropout

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of the model: {} %'.format(100 * correct / total))
    # Save 
    torch.save(model.state_dict(), 'model.ckpt')
    


if __name__ == "__main__":
    main()



