import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Recall, Precision, Fbeta, confusion_matrix, ConfusionMatrix, EpochMetric
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.leaky_relu(out)
        if self.is_last:
            return out, preact
        else:
            return out
        
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out
        
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop1 = nn.Dropout(0.4)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.drop2 = nn.Dropout(0.4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out
    

class MergerNet(nn.Module):
    def __init__(self, cutout_size = 200, channels=1, num_classes = 3):
        super(MergerNet, self).__init__()
        self.cutout_size = cutout_size
        self.channels = channels
        self.expected_input_shape = (16, self.channels, self.cutout_size, self.cutout_size)
            
        self.backbone = ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], in_channel=channels, zero_init_residual=True) #Re12snet18 outputs 5
        #self.fc1 = nn.Linear(2048, 1024)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        
        out = self.fc(features)
        
        return out
    
# Simple CNN to see if ResNet architecture performs better.
class MergerCNN(nn.Module): # based on the DRAGON_CNN architecture
    def __init__(self, cutout_size=200, channels=1, num_classes=3):
        super(MergerCNN, self).__init__()
        self.cutout_size = cutout_size
        self.channels = channels
        self.expected_input_shape = (
            16,
            self.channels,
            self.cutout_size,
            self.cutout_size,
        )

        self.num_classes = num_classes

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.channels, 64, kernel_size=(3, 3), padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding='same'),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding='same'),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding='same'),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding='same'),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer8 = nn.Sequential(
            nn.Conv2d(384, 512, kernel_size=3, padding='same'),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(2048, 1024)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Forward pass through the layers
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        # Flatten the output tensor
        out = out.view(out.size(0), -1)

        # Fully connected layers
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)

        return out

    
           
