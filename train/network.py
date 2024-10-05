import math
import torch
import torch.nn as nn
from modules import Conv2D, Linear 

class ConvReLUBatchNorm(nn.Module):
    def __init__(self, in_features, out_features, kernel):
        super().__init__()
        self.weight = self.init_weight(in_features, out_features, kernel).cuda()
        self.weight.retain_grad()
        self.bias = self.init_bias(out_features, norm=True).cuda()
        self.bn = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def init_weight(self, in_features, out_features, kernel):
        weight = torch.randn(out_features, in_features, kernel, kernel, requires_grad=True).to(torch.float32)
        torch.nn.init.xavier_uniform_(weight)
        return weight

    def init_bias(self, out_features, norm=True):
        if norm:
            bias = torch.zeros(out_features, requires_grad=True).to(torch.float32)
        else:
            bias = torch.randn(out_features, require_grad=True).to(torch.float32)
            nn.init.xavier_uniform_(bias)
        return bias
    
    def forward(self, x):
        x = Conv2D.apply(x, self.weight, self.bias, 0, 1, 1, 1, False)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class FeedForward(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = self.init_weight(in_features, out_features).cuda()
        self.weight.retain_grad()
        self.b = torch.zeros(out_features).cuda()
    
    def init_weight(self, in_features, out_features):
        weight = torch.randn(out_features, in_features, requires_grad=True).to(torch.float32)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        return weight
    
    def forward(self, x):
        x = Linear.apply(x, self.weight, self.b, False)
        return x
    

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvReLUBatchNorm(1, 128, 5)
        self.conv2 = ConvReLUBatchNorm(128, 256, 3)
        self.conv3 = ConvReLUBatchNorm(256, 128, 3)
        self.avgpool = nn.AvgPool2d(20).cuda()
        self.ffw1 = FeedForward(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)        
        x = self.avgpool(x).reshape(x.shape[0], 128)
        x = self.ffw1(x)
        return x