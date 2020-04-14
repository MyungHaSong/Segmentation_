import torch
import torch.nn as nn
from backbone.VGG import VGGNet
import torch.nn.functional as F

class FCN32s(nn.Module):
    def __init__(self,n,pretrain):
        super(FCN32s,self).__init__()
        class_n = n
        self.pretrained = pretrain
        self.gate = nn.Sequential(
            nn.Conv2d(512,4096,7),
            nn.ReLU(inplace = True),
            nn.Dropout2d(),
            nn.Conv2d(4096,4096,1),
            nn.ReLU(inplace = True),
            nn.Dropout2d(),
            nn.Conv2d(4096,class_n,1),
        )
    def forward(self,x):
        size= x.size()[2:]
        x = self.pretrained(x)
        x = x['x5']
        x = self.gate(x)
        x = F.upsample_bilinear(x,size)
        return x 

class FCN16s(nn.Module):
    def __init__(self,n,pretrain):
        super(FCN32s,self).__init__()
        class_n = n
        self.pretrained = pretrain
        self.gate = nn.Sequential(
            nn.Conv2d(512,4096,7),
            nn.ReLU(inplace = True),
            nn.Dropout2d(),
            nn.Conv2d(4096,4096,1),
            nn.ReLU(inplace = True),
            nn.Dropout2d(),
            nn.Conv2d(4096,class_n,1),
        )
    def forward(self,x):
        size= x.size()[2:]
        x = self.pretrained(x)
        x5 = x['x5']
        x4 = x['x4']
        size_4 = x4.size()[2:]
        x = self.gate(x)
        x = F.upsample_bilinear(x,size_4)
        x += x4
        x = F.upsample_bilinear(x,size)
        return x
class FCN8s(nn.Module):
    def __init__(self,n,pretrain):
        super(FCN32s,self).__init__()
        class_n = n
        self.pretrained = pretrain
        self.gate = nn.Sequential(
            nn.Conv2d(512,4096,7),
            nn.ReLU(inplace = True),
            nn.Dropout2d(),
            nn.Conv2d(4096,4096,1),
            nn.ReLU(inplace = True),
            nn.Dropout2d(),
            nn.Conv2d(4096,class_n,1),
        )
    def forward(self,x):
        size= x.size()[2:]
        x = self.pretrained(x)
        x5 = x['x5']
        x4 = x['x4']
        x3 = x['x3']
        size_4 = x4.size()[2:]
        size_3 = x3.size()[2:]
        x = self.gate(x)
        x = F.upsample_bilinear(x,size_4)
        x += x4
        x = F.upsample_bilinear(x,size_3)
        x += x3
        x = F.upsample_bilinear(x,size)
        return x
vgg = VGGNet(requires_grad = True)
data = torch.rand([1,3,255,255])
b = FCN32s(21,vgg)(data)






