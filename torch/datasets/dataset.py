
import glob
import os
import torch
import cv2
import numpy as np
import random
from PIL import Image
from torchvision import transforms
from utils import *

class CamVid(torch.utils.data.Dataset):
    def __init__(self,path,mode = 'train'):
        self.path = path
        self.mode = mode
        self.image_path = sorted(glob.glob(path +'/'+ self.mode +'/*.png'))
        self.label_path = sorted(glob.glob(path +'/'+ self.mode + '_labels/' + '*.png'))
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.csv = get_label_info('./datasets/CamVid/class_dict.csv')


    def __getitem__(self,index):
        random_seed = random.random()
        img = np.array(Image.open(self.image_path[index]))
        label = np.array(Image.open(self.label_path[index]))
        label = one_hot_it_v11_dice(label,self.csv).astype(np.uint8)
        label = np.transpose(label, [2,0,1]).astype(np.float32)
        img = self.to_tensor(img).float()
        return img, label
    def __len__(self):
        return len(self.image_path)
