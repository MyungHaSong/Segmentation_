from backbone.resnet_bise import build_context
import torch
import torch.nn as nn
import torch.nn.functional as F
class Convblock(nn.Module):
    def __init__(self, in_c, out, kernel_size=3,stride=2):
        super(Convblock,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c,out,kernel_size=kernel_size,stride=stride,padding = 1, bias=False),
            nn.BatchNorm2d(out),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.block(x)

class SpatialPath(nn.Module):
    def __init__(self):
        super(SpatialPath,self).__init__()
        self.block1 = Convblock(3,64)
        self.block2 = Convblock(64,128)
        self.block3 = Convblock(128,256)
    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
class AttRefineModule(nn.Module):
    def __init__(self,in_c,out):
        super(AttRefineModule,self).__init__()
        self.conv = nn.Conv2d(in_c,out,kernel_size=1,stride=1, bias=False)
        self.bn = nn.BatchNorm2d(out)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    def forward(self,x):
        out = self.avg_pool(x)
        out = self.conv(x)
        out = self.bn(x)
        out = self.sigmoid(x)
        x = torch.mul(x,out)
        return x
class FeatureFusionModule(nn.Module):
    def __init__(self,num_classes,in_c):
        super(FeatureFusionModule,self).__init__()
        self.block = Convblock(in_c,num_classes,stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.conv1 = nn.Conv2d(num_classes,num_classes,kernel_size=1,stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(num_classes,num_classes,kernel_size=1,stride=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,in1,in2):
        input = torch.cat([in1,in2], dim= 1)
        feature = self.block(input)
        out = self.avg_pool(feature)
        out = self.relu(self.conv1(out))
        out = self.relu(self.conv2(out))
        out = torch.mul(feature,out)
        out = torch.add(out,feature)
        return out
class BiseNet(nn.Module):
    def __init__(self,num_classes,context_model):
        super(BiseNet, self).__init__()
        self.context = build_context(context_model)
        self.spatial = SpatialPath()
        if context_model =='resnet18':
            self.att_refine_module1 = AttRefineModule(256,256)
            self.att_refine_module2 = AttRefineModule(512,512)
            self.super_conv1 = nn.Conv2d(256,num_classes,kernel_size=1)
            self.super_conv2 = nn.Conv2d(512,num_classes,kernel_size=1)
            self.feature_fusion = FeatureFusionModule(num_classes,1024)
        elif context_model =='resnet101':
            self.att_refine_module1 = AttRefineModule(1024,1024)
            self.att_refine_module2 = AttRefineModule(2048,2048)
            self.super_conv1 = nn.Conv2d(1024,num_classes,kernel_size=1)
            self.super_conv2 = nn.Conv2d(2048,num_classes,kernel_size=1)
            self.feature_fusion = FeatureFusionModule(num_classes,3328)
        self.conv = nn.Conv2d(num_classes,num_classes,kernel_size= 1)

    def forward(self,x):
        s_feature = self.spatial(x)
        x_3,x_4,tail = self.context(x)
        x_3 = self.att_refine_module1(x_3)
        x_4 = self.att_refine_module2(x_4)
        x_4 = torch.mul(x_4,tail)

        x_3 = torch.nn.functional.interpolate(x_3, size= s_feature.size()[-2:],mode = 'bilinear')
        x_4 = torch.nn.functional.interpolate(x_4, size= s_feature.size()[-2:],mode = 'bilinear')
        feature = torch.cat([x_3,x_4], dim=1)

        if self.training == True:
            x_3_sup = self.super_conv1(x_3)
            x_4_sup = self.super_conv2(x_4)
            x_3_sup = torch.nn.functional.interpolate(x_3_sup, size = x.size()[-2:], mode = 'bilinear')
            x_4_sup = torch.nn.functional.interpolate(x_4_sup, size = x.size()[-2:], mode = 'bilinear')


        result = self.feature_fusion(s_feature,feature)
        result = torch.nn.functional.interpolate(result, scale_factor=8,mode='bilinear')
        result = self.conv(result)


        if self.training ==True:
            return result, x_3_sup, x_4_sup
        return result
