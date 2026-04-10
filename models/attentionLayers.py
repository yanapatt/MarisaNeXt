# -*- coding: utf-8 -*-
"""attentionLayers.ipynb

Original file is located at
    https://colab.research.google.com/drive/1G7pAdHRur_zny2idEIWxb3Uef0sCqGo8
"""

import torch
import torch.nn as nn
from typing import Type

# Reference: https://arxiv.org/abs/1807.06521
# Reference Code: https://github.com/changzy00/pytorch-attention/blob/master/attention_mechanisms/cbam.py
# CBAM: Convolutional Block Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, min_hidden_channels: int = 1, act_layer: Type[nn.Module] = nn.GELU) -> None:
        super(ChannelAttention, self).__init__()
        hidden_dim = max(min_hidden_channels, in_channels // reduction)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            act_layer(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(torch.mean(x, dim=(2, 3), keepdim=True))
        max_out = self.fc(torch.amax(x, dim=(2, 3), keepdim=True))

        output = avg_out + max_out
        scale = self.sigmoid(output)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super(SpatialAttention, self).__init__()

        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.cat([avg_out, max_out], dim=1)
        scale = self.conv1(scale)
        scale = self.sigmoid(scale)

        return x * scale

class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, min_hidden_channels: int = 1, kernel_size: int = 7, act_layer: Type[nn.Module] = nn.GELU) -> None:
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(
            in_channels,
            reduction=reduction,
            min_hidden_channels=min_hidden_channels,
            act_layer=act_layer
        )
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x

# Reference: https://arxiv.org/abs/1709.01507
# Reference Code: https://github.com/changzy00/pytorch-attention/blob/master/attention_mechanisms/se_module.py
# Squeeze-and-Excitation Networks
class SE(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, min_hidden_channels: int = 1, act_layer: Type[nn.Module] = nn.GELU) -> None:
        super(SE, self).__init__()
        hidden_dim = max(min_hidden_channels, in_channels // reduction)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim, bias=False),
            act_layer(),
            nn.Linear(hidden_dim, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, w, h = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Reference: https://arxiv.org/abs/2103.02907
# Reference Code: https://github.com/changzy00/pytorch-attention/blob/master/attention_mechanisms/coordatten.py
# Coordinate Attention for Efficient Mobile Network Design
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16, min_hidden_channels: int = 1, act_layer: Type[nn.Module] = nn.GELU) -> None:
        super(CoordinateAttention, self).__init__()
        hidden_dim = max(min_hidden_channels, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        self.act = act_layer()

        self.conv_h = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, h, w = x.shape

        # Mean + Max Pooling (pool_h) -> (B, C, H, 1)
        avg_h = torch.mean(x, dim=3, keepdim=True)
        max_h = torch.amax(x, dim=3, keepdim=True)
        x_h = avg_h + max_h

        # Mean + Max Pooling (pool_w) -> (B, C, 1, W)
        avg_w = torch.mean(x, dim=2, keepdim=True)
        max_w = torch.amax(x, dim=2, keepdim=True)
        x_w = (avg_w + max_w).transpose(-1, -2) # (B, C, W, 1)

        # Concatenate and Transform
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split and Generate Attention Maps
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.transpose(-1, -2) # (B, C, 1, W)

        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))

        out = identity * a_h * a_w
        return out

# Reference: http://proceedings.mlr.press/v139/yang21o/yang21o.pdf
# Reference Code: https://github.com/changzy00/pytorch-attention/blob/master/attention_mechanisms/simam.py
# SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks
class SimAM(torch.nn.Module):
    def __init__(self, in_channels: int = None, e_lambda: float = 1e-04):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)