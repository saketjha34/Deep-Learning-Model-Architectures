import torch
import torch.nn as nn
import numpy as np

class GroupedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(GroupedConv2d, self).__init__()
        self.groups = groups
        self.group_in_channels = in_channels // groups
        self.group_out_channels = out_channels // groups

        # Initialize weight for grouped convolution manually using numpy
        self.weight = nn.Parameter(torch.tensor(
            np.random.randn(groups, self.group_out_channels, self.group_in_channels, kernel_size, kernel_size),
            dtype=torch.float32
        ))

        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Split input by groups
        group_inputs = torch.chunk(x, self.groups, dim=1)
        group_outputs = [torch.nn.functional.conv2d(
            group_input, 
            self.weight[group_idx], 
            self.bias[self.group_out_channels * group_idx: self.group_out_channels * (group_idx + 1)]
            if self.bias is not None else None,
            stride=self.stride, 
            padding=self.padding
        ) for group_idx, group_input in enumerate(group_inputs)]

        # Concatenate along the channel dimension
        return torch.cat(group_outputs, dim=1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups, width_per_group):
        super(BottleneckBlock, self).__init__()
        intermediate_channels = groups * width_per_group
        
        # 1x1 reduction
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)

        # Grouped convolution (3x3)
        self.conv2 = GroupedConv2d(intermediate_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)

        # 1x1 expansion
        self.conv3 = nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        # Bottleneck forward
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = torch.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Add shortcut
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return torch.relu(out)

class ResNeXt(nn.Module):
    def __init__(self, layers:list[int]=[3, 4, 6,3], groups:int=32, width_per_group:int=4, num_classes:int=1000):
        super(ResNeXt, self).__init__()
        self.in_channels = 64

        # Initial Convolution Layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNeXt Layers
        self.layer1 = self._make_layer(256, layers[0], stride=1, groups=groups, width_per_group=width_per_group)
        self.layer2 = self._make_layer(512, layers[1], stride=2, groups=groups, width_per_group=width_per_group)
        self.layer3 = self._make_layer(1024, layers[2], stride=2, groups=groups, width_per_group=width_per_group)
        self.layer4 = self._make_layer(2048, layers[3], stride=2, groups=groups, width_per_group=width_per_group)

        # Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, out_channels, blocks, stride, groups, width_per_group):
        layers = []
        layers.append(BottleneckBlock(self.in_channels, out_channels, stride, groups, width_per_group))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BottleneckBlock(self.in_channels, out_channels, 1, groups, width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x