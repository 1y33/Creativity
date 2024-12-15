import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseConv(nn.Module):
    def __init__(self, conv_layer, in_channels, out_channels, kernel=3, stride=1, padding=1, activation=nn.ReLU()):
        super().__init__()
        self.layers = nn.Sequential(
            conv_layer(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class Conv(BaseConv):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, activation=nn.ReLU()):
        super().__init__(nn.Conv2d, in_channels, out_channels, kernel, stride, padding, activation)

class ConvTr(BaseConv):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, padding=1, activation=nn.ReLU()):
        super().__init__(nn.ConvTranspose2d, in_channels, out_channels, kernel, stride, padding, activation)

class ResidualConv(nn.Module):
    def __init__(self, conv_class, in_channels, out_channels, kernel=3, stride=1, padding=1, activation=nn.ReLU()):
        super().__init__()
        self.conv = conv_class(in_channels, out_channels, kernel, stride, padding, activation)

        if in_channels != out_channels or stride != 1:
            self.id = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        else:
            self.id = nn.Identity()

        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.id(x)
        x = self.conv(x)
        x = self.activation(x + skip)
        return x

class NResidual(nn.Module):
    def __init__(self, n_layers, in_channels, out_channels, block_class=Conv, **kwargs):
        super().__init__()
        layers = [block_class(in_channels, in_channels, **kwargs) for _ in range(n_layers - 1)]
        layers.append(block_class(in_channels, out_channels, **kwargs))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
class MaxPool(nn.Module):
    def __init__(self,kernel=2,stride=2,padding=0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel,stride,padding)
    
    def forward(self,x :torch.Tensor)->torch.Tensor:
        return self.pool(x)

class BottleNeck(nn.Module):
    def __init__(self, in_channels,out_channels,expansion=4,kernel=3,stride=1,padding=0):
        super().__init__()

        bottleneck_channels = out_channels // expansion

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)

        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=kernel, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        skip = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv3(x)))
        x = self.bn3(self.conv3(x))
        return x + skip
