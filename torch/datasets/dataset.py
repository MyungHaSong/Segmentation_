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
    def __init__(self,path,scale,mode = 'training'):
        self.path = path
        self.image_path = sorted(glob.glob(path + 'img/' + '*.png'))
        self.label_path = sorted(glob.glob(path + 'label/' + '*.png'))
        self.to_tensor = transform.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.csv = get_label_info('./CamVid/CamVid/class_dict.csv')


    def __getitem__(self,index):
        random_seed = random.random()
        img = Image.open(self.image_path[index])
        label = np.array(Image.open(self.label_path[index]))
        label = one_hot_it_v11(label,self.csv)
        return img, label
