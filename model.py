from torchvision import datasets, transforms
from PIL import Image
import os
from functions_for_train import calculate_optimal_size, find_image_sizes
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from collections import Counter
import numpy as np
import pandas as pd
import torch.nn.functional as F

import torch

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
class SSEBlock(nn.Module):
    def __init__(self, channels):
        super(SSEBlock, self).__init__()
        self.conv_spatial = nn.Conv2d(channels, 1, kernel_size=1)  # For spatial squeeze
        self.conv_channel = nn.Conv2d(channels, channels, kernel_size=1)  # For channel excitation
        self.sigmoid = nn.sigmoid()
        self.swish = Swish()

    def forward(self, x):
        # Spatial Squeeze: compress spatial dimensions to a single channel
        spatial_squeeze = self.conv_spatial(x)
        spatial_squeeze = self.sigmoid(spatial_squeeze)

        # Channel Excitation: emphasize important channels
        channel_excitation = self.conv_channel(x)
        channel_excitation = self.sigmoid(channel_excitation)

        # Element-wise multiplication of spatial and channel attention
        out = spatial_squeeze * channel_excitation * x
        return out
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.swish = Swish()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class FEEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FEEBlock, self).__init__()
        # Use smaller kernel sizes to avoid issues with input size
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # Reduced to 3x3
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(self.bn(self.conv2(x)))
        return x

# Define DFM Block
class DFMBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DFMBlock, self).__init__()
        # Use smaller kernel sizes with padding to avoid input size issues
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # Reduced to 3x3
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=2, padding=1)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv1(x)))
        x = self.relu(self.bn(self.conv2(x)))
        x = self.relu(self.bn(self.conv3(x)))
        x = self.relu(self.bn(self.conv4(x)))
        return x


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, dilation=dilation, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
class EnhancedSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv2d(2, 1, kernel_size, padding=kernel_size // 2, dilation=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU()

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = self.gelu(self.fc1(w))
        w = self.gelu(self.fc2(w))
        return x * w



# RASPP (Residual Atrous Spatial Pyramid Pooling)
# class RASPP(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(RASPP, self).__init__()
#         dilation_rates = [1, 3, 6, 9]
#         self.branches = nn.ModuleList([
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False)
#             for rate in dilation_rates
#         ])
#         self.conv1x1 = nn.Conv2d(out_channels * len(dilation_rates), out_channels, kernel_size=1)
#         self.bn = nn.BatchNorm2d(out_channels)

#     def forward(self, x):
#         branches = [branch(x) for branch in self.branches]
#         out = torch.cat(branches, dim=1)
#         out = self.conv1x1(out)
#         out = self.bn(out)
#         return F.relu(out)

# R2 Block (Recursive Residual Block)
class R2Block(nn.Module):
    def __init__(self, in_channels, out_channels, recursion=2):
        super(R2Block, self).__init__()
        self.recursion = recursion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        for _ in range(self.recursion):
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            x += residual  # Add the residual connection
        return F.relu(x)

class EnhancedResNet(nn.Module):#1024
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet50(weights=None)  # Initialize without pre-trained weights
        # Load state_dict from the pre-trained ResNet-50 model
        pre_trained_weights = torch.load('pre_weight/resnet50.pth/resnet50.pth')
        resnet.load_state_dict(pre_trained_weights)

        # Use all layers except the last two (adaptive pooling and fc layer)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # 1x1 convolution to reduce channels from 2048 to 1024
        self.conv1x1 = nn.Conv2d(2048, 1024, kernel_size=1)
        self.r2_block = R2Block(1024, 1024)
        # Adjust the custom layers for 1024 channels
        self.se_block = SEBlock(1024)
        self.channel_attention = ChannelAttention(1024)
        self.spatial_attention = EnhancedSpatialAttention()
        self.residual_block = BasicBlock(1024, 1024)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)  # Feature extraction

        x = self.conv1x1(x)  # Reduce channels from 2048 to 1024

        x = self.se_block(x)  # Squeeze and Excitation block
        x = self.channel_attention(x) * x  # Channel attention
        x = self.spatial_attention(x) * x  # Spatial attention
        x = self.residual_block(x)  # Residual block

        x = F.dropout(x, p=0.3, training=self.training)  # Dropout

        x = self.se_block(x)
        x = self.channel_attention(x) * x  # Channel attention
        x = self.spatial_attention(x) * x  # Spatial attention
        x = self.residual_block(x)  # Residual block

        x = self.pooling(x)  # Adaptive average pooling
        x = torch.flatten(x, 1)  # Flatten for fully connected layer
        x = self.fc(x)  # Output layer
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size // 2))
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Channel-wise average
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Channel-wise max
        out = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(out))
    

# class CustomModel(nn.Module):
#     def __init__(self, num_classes):
#         super(CustomModel, self).__init__()
#         # DenseNet121 Backbone (Feature Extractor)
#         densenet = models.densenet121(pretrained=True)
#         self.backbone = nn.Sequential(*list(densenet.features))  # DenseNet121 features
        
#         # Conv2D Layer (Input channels adjusted for DenseNet121 output)
#         self.conv2d = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        
#         # SEBlock
#         self.se_block = SEBlock(1024)
        
#         # Channel Attention
#         self.channel_attention = ChannelAttention(1024)
        
#         # Global Average Pooling
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
#         # Fully Connected Layer
#         self.fc = nn.Linear(1024 * 2, num_classes)  # For concatenation of branch1 and branch2

#     def forward(self, x):
#         # Backbone
#         x = self.backbone(x)  # Output: [Batch, 1024, 7, 7]
        
#         # Conv2D
#         x = self.conv2d(x)  # Output: [Batch, 1024, 7, 7]
        
#         # Branch 1: SE -> Channel Attention
#         branch1 = self.se_block(self.channel_attention(x))
        
#         # Branch 2: Channel Attention -> SE
#         branch2 = self.channel_attention(self.se_block(x))
        
#         # Concatenate Both Branches
#         x = torch.cat([branch1, branch2], dim=1)  # [Batch, 2048, 7, 7]
        
#         # Global Average Pooling
#         x = self.global_avg_pool(x)  # [Batch, 2048, 1, 1]
        
#         # Flatten
#         x = torch.flatten(x, 1)  # [Batch, 2048]
        
#         # Fully Connected Layer
#         x = self.fc(x)  # [Batch, num_classes]
        
#         return x


        
class CustomModel(nn.Module): #(Original)
    def __init__(self, num_classes):
        super(CustomModel, self).__init__()
        # ResNet50 Backbone (Feature Extractor)
        # densenet = models.densenet121(pretrained=True)

        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove FC layer and AvgPool
        
        # Conv2D Layer
        self.conv2d = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1)
        
        # SEBlock
        self.se_block = SEBlock(1024)
        
        # Channel Attention
        self.channel_attention = ChannelAttention(1024)
        
        # Spatial Attention
        self.spatial_attention = SpatialAttention()
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully Connected Layer
        self.fc = nn.Linear(1024 * 2, num_classes)  # Adjusted for concatenation of branch1 and branch2
    
    def forward(self, x):
        # Backbone (ResNet50)
        x = self.backbone(x)  # Output: [Batch, 2048, 7, 7]
        
        # Conv2D
        x = self.conv2d(x)  # Output: [Batch, 1024, 7, 7]
        
        # Parallel Pathways
        branch1 = self.se_block(self.channel_attention(x))  # SE -> Channel Attention
        branch2 = self.channel_attention(self.se_block(x))  # Channel Attention -> SE
        
        # Concatenate Parallel Pathways
        x = torch.cat([branch1, branch2], dim=1)  # Output: [Batch, 2048, 7, 7] after concat
        
        # Spatial Attention
        x = self.spatial_attention(x)  # Output: [Batch, 2048, 7, 7]
        
        # Global Average Pooling
        x = self.global_avg_pool(x)  # Output: [Batch, 2048, 1, 1]
        
        # Flatten before feeding to FC layer
        x = torch.flatten(x, 1)  # Output: [Batch, 2048]
        
        # Fully Connected Layer
        x = self.fc(x)  # Output: [Batch, num_classes]
        
        return x

# class 3BranchNet(nn.Module):
#     def __init__(self, num_classes):
#         super(CustomModel2, self).__init__()
#         resnet = models.resnet50(pretrained=True)
#         self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
#         self.conv2d = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1)
#         self.se_block = SEBlock(1024)
#         self.channel_attention = ChannelAttention(1024)
#         self.spatial_attention = SpatialAttention()
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
#         # Fully Connected Layer
#         self.fc = nn.Linear(1024 * 3, num_classes)  # 3 branches

#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.conv2d(x)

#         branch1 = self.se_block(self.channel_attention(x))
#         branch2 = self.channel_attention(self.se_block(x))
#         branch3 = self.spatial_attention(self.channel_attention(x))

#         x = torch.cat([branch1, branch2, branch3], dim=1)
#         x = self.global_avg_pool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x


# class EnhancedResNet(nn.Module):#2048
#     def __init__(self, num_classes):
#         super().__init__()
#         resnet = models.resnet50(weights=None)  # Initialize without pre-trained weights
#         # Load state_dict from the pre-trained ResNet-50 model
#         pre_trained_weights = torch.load('pre_weight/resnet50.pth/resnet50.pth')
#         resnet.load_state_dict(pre_trained_weights)
#
#         # Use all layers except the last two (adaptive pooling and fc layer)
#         self.features = nn.Sequential(*list(resnet.children())[:-2])
#
#          # Adding custom layers
#         self.se_block = SEBlock(2048)
#         self.channel_attention = ChannelAttention(2048)
#         self.spatial_attention = EnhancedSpatialAttention()
#         self.residual_block = BasicBlock(2048, 2048)
#         self.pooling = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(2048, num_classes)
#     def forward(self, x):
#         x = self.features(x)  # Feature extraction
#
#         x = self.se_block(x)  # Squeeze and Excitation block
#         x = self.channel_attention(x) * x  # Channel attention
#         x = self.spatial_attention(x) * x  # Spatial attention
#         x = self.residual_block(x)  # Residual block
#
#         x = F.dropout(x, p=0.3, training=self.training)  # Dropout
#
#         x = self.se_block(x)
#         x = self.channel_attention(x) * x  # Channel attention
#         x = self.spatial_attention(x) * x  # Spatial attention
#         x = self.residual_block(x)  # Residual block
#
#         x = self.pooling(x)  # Adaptive average pooling
#         x = torch.flatten(x, 1)  # Flatten for fully connected layer
#         x = self.fc(x)  # Output layer
#         return x



