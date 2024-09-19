import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)        # Batch Normalization
        self.relu1 = nn.ReLU(inplace=True)          
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(out_planes)       # Batch Normalization
        self.relu2 = nn.ReLU(inplace=True)         
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False) 
        self.droprate = dropRate                    # Dropout
        self.equalInOut = (in_planes == out_planes) 
        # 如果输入输出通道数不同，使用1x1卷积匹配维度
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))              # 如果输入输出通道数不同，先做BN和ReLU
        else:
            out = self.relu1(self.bn1(x))           # 如果输入输出通道数相同，直接做BN和ReLU
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))  # 第一个卷积层
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)      # Dropout层
        out = self.conv2(out)                        # 第二个卷积层
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)   # 残差连接

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]  # 每个阶段的通道数
        assert((depth - 4) % 6 == 0)                 # 确保深度(depth)可以被6整除，并且至少为4
        n = (depth - 4) / 6                         # 每个阶段的块数
        block = BasicBlock                          # 使用的基本块类型为BasicBlock

        # 第一个卷积层，输入通道数为3（RGB图像），输出通道数为nChannels[0]，使用3x3的卷积核
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        
        # 三个网络阶段，每个阶段有n个块
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        
        # 全局平均池化和分类器
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)   # 全连接层，输出为num_classes个类别
        self.nChannels = nChannels[3]                    # 最后一个阶段的输出通道数

        # 初始化所有层的权重和偏置
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 卷积层权重初始化
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)            # BN层权重初始化
                m.bias.data.zero_()               # BN层偏置初始化
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()               # 全连接层偏置初始化
    
    def forward(self, x):
        out = self.conv1(x)                     
        out = self.block1(out)                    # 第一个网络阶段
        out = self.block2(out)                    # 第二个网络阶段
        out = self.block3(out)                    # 第三个网络阶段
        out = self.relu(self.bn1(out))            # BN和ReLU
        out = F.avg_pool2d(out, 8)                # 全局平均池化，kernel_size为8
        out = out.view(-1, self.nChannels)        # 展平操作
        return self.fc(out)                       # 分类器，输出分类结果
