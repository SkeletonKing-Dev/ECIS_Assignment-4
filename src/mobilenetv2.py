"""
Student 1: MobileNetV2 (Sandler et al., 2018), width 1.0.

Building block: inverted residual = 1x1 expansion (ReLU6) -> 3x3 depthwise
(ReLU6) -> 1x1 linear projection, with a skip connection when the stride is 1
and the channel count is unchanged.

CIFAR adaptation (32x32 input): stem stride 1 instead of 2, and the second
stage stride 1 instead of 2, so the final feature map is 4x4 (an unchanged
ImageNet network would end at 1x1).

Feature maps used for distillation:
  after the 96-channel stage: 96 x 8 x 8
  after the 1x1 head conv:   1280 x 4 x 4
"""
import torch
import torch.nn as nn


def conv_bn_act(in_ch, out_ch, k=3, stride=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, k, stride, k // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(out_ch), nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, out_ch, stride, expand):
        super().__init__()
        hidden = in_ch * expand
        layers = [] if expand == 1 else [conv_bn_act(in_ch, hidden, 1)]
        layers += [conv_bn_act(hidden, hidden, 3, stride, groups=hidden),          # depthwise
                   nn.Conv2d(hidden, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]  # linear bottleneck
        self.block = nn.Sequential(*layers)
        self.use_res = stride == 1 and in_ch == out_ch

    def forward(self, x):
        return x + self.block(x) if self.use_res else self.block(x)


class MobileNetV2(nn.Module):
    # (expansion t, output channels c, repeats n, stride s); stride of stage 2 changed 2 -> 1 for CIFAR
    cfg = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
           (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]
    feat_channels = (96, 1280)

    def __init__(self, num_classes=100, dropout=0.2):
        super().__init__()
        self.stem = conv_bn_act(3, 32, 3, 1)               # stride 1 for 32x32 input
        in_ch = 32
        stages = []
        for t, c, n, s in self.cfg:
            blocks = []
            for i in range(n):
                blocks.append(InvertedResidual(in_ch, c, s if i == 0 else 1, t))
                in_ch = c
            stages.append(nn.Sequential(*blocks))
        self.stages_mid = nn.Sequential(*stages[:5])       # up to the 96-channel stage (8x8)
        self.stages_end = nn.Sequential(*stages[5:])       # 160, 320 channels (4x4)
        self.head = conv_bn_act(320, 1280, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))

    def forward(self, x, return_feats=False):
        f_mid = self.stages_mid(self.stem(x))
        f_final = self.head(self.stages_end(f_mid))
        logits = self.classifier(torch.flatten(self.pool(f_final), 1))
        return (logits, [f_mid, f_final]) if return_feats else logits
