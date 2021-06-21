import torch as t
import torch.nn as nn
from torch.nn.modules.activation import ReLU


class BasicBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1, res: bool = False, norm: bool = True, dropout: float = 0.2):
        super().__init__()
        self.res = res
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(c_out) if norm else nn.Identity()
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(c_out) if norm else nn.Identity()
        if res and stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, 0),
                nn.BatchNorm2d(c_out) if norm else nn.Identity()
            )
        else:
            self.downsample = None
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)
        if self.res:
            out += identity
        out = self.relu2(out)
        out = self.dropout2(out)
        return out


class CNN(nn.Module):
    def __init__(self, block_sizes, res: bool = False, norm: bool = True, conv_dropout=0.2, fc_dropout=0.5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),  # 64*64*64
            nn.BatchNorm2d(64) if norm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 64*32*32
        )
        self.block_list = nn.ModuleList()
        for block_size in block_sizes:
            block = BasicBlock(*block_size, res=res, norm=norm, dropout=conv_dropout)
            self.block_list.append(block)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(block_sizes[-1][1], 200)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.block_list:
            x = block(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block_sizes, res: bool = False, norm: bool = True, conv_dropout=0.2, fc_dropout=0.5):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),  # 64*32*32
            nn.BatchNorm2d(64) if norm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)  # 64*16*16
        )
        self.block_list = nn.ModuleList()
        for block_size in block_sizes:
            block = BasicBlock(*block_size, res=res, norm=norm, dropout=conv_dropout)
            self.block_list.append(block)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(block_sizes[-1][1], 200)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.block_list:
            x = block(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
