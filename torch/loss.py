import torch
import torch.nn as nn
import torch.nn.functional as F

def flatten(x):
    c = x.size(1)
    axis_ = (1,0) + tuple(range(2,x.dim()))
    transpose = x.permute(axis)
    return transpose.contiguous().view(c,-1)

def dice_loss(input, target):
    ep = 1e-5
    output = F.softmax(output, dim=1)
    output = flatten(output)
    target = flatten(target)
    inter = (output * target).sum(-1)
    deno = (output+target).sum(-1)
    dice = inter / deno
    return 1-dice

