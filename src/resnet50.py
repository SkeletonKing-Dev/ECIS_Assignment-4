"""
Teacher: ResNet-50 (bottleneck blocks, [3, 4, 6, 3]) adapted to 32x32 images:
3x3 stride-1 stem and no max-pool (the ImageNet 7x7/stride-2 stem + max-pool
would shrink a 32x32 image to 8x8 before the first stage).

Feature maps used for feature distillation:
  layer3 output: 1024 x 8 x 8   (intermediate)
  layer4 output: 2048 x 4 x 4   (final, before pooling)
"""
import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_ch, planes, stride=1):
        super().__init__()
        out_ch = planes * self.expansion
        self.conv1 = nn.Conv2d(in_ch, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                                          nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + self.shortcut(x))


class ResNet50(nn.Module):
    feat_channels = (1024, 2048)

    def __init__(self, num_classes=100):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        in_ch = 64
        stages = []
        for planes, blocks, stride in ((64, 3, 1), (128, 4, 2), (256, 6, 2), (512, 3, 2)):
            layers = []
            for s in [stride] + [1] * (blocks - 1):
                layers.append(Bottleneck(in_ch, planes, s))
                in_ch = planes * Bottleneck.expansion
            stages.append(nn.Sequential(*layers))
        self.layer1, self.layer2, self.layer3, self.layer4 = stages
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x, return_feats=False):
        x = self.layer2(self.layer1(self.stem(x)))
        f_mid = self.layer3(x)
        f_final = self.layer4(f_mid)
        logits = self.fc(torch.flatten(self.pool(f_final), 1))
        return (logits, [f_mid, f_final]) if return_feats else logits
