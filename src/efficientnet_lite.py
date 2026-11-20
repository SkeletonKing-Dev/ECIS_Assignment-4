"""
Student 3: EfficientNet-Lite0 (Google, 2020): EfficientNet-B0 made friendly to
mobile accelerators and INT8 quantization:
  * no squeeze-and-excitation blocks
  * ReLU6 instead of Swish
  * stem (32 ch) and head (1280 ch) are fixed, not scaled
Lite0 uses width = depth = 1.0, so the block layout equals B0.

Building block: MBConv = 1x1 expansion -> kxk depthwise -> 1x1 projection,
residual when stride is 1 and channels match.

CIFAR adaptation (32x32 input): stem stride 1 and the second stage stride 1,
so the final feature map is 4x4.

Feature maps used for distillation:
  after the 112-channel stage: 112 x 8 x 8
  after the 1x1 head conv:    1280 x 4 x 4
"""
import torch
import torch.nn as nn


def conv_bn_relu6(in_ch, out_ch, k=1, stride=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, k, stride, k // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(out_ch), nn.ReLU6(inplace=True))


class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride, expand):
        super().__init__()
        hidden = in_ch * expand
        layers = [] if expand == 1 else [conv_bn_relu6(in_ch, hidden, 1)]
        layers += [conv_bn_relu6(hidden, hidden, k, stride, groups=hidden),         # depthwise
                   nn.Conv2d(hidden, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]
        self.block = nn.Sequential(*layers)
        self.use_res = stride == 1 and in_ch == out_ch

    def forward(self, x):
        return x + self.block(x) if self.use_res else self.block(x)


class EfficientNetLite0(nn.Module):
    # (expansion, kernel, stride, out channels, repeats); stage-2 stride changed 2 -> 1 for CIFAR
    cfg = [(1, 3, 1, 16, 1), (6, 3, 1, 24, 2), (6, 5, 2, 40, 2), (6, 3, 2, 80, 3),
           (6, 5, 1, 112, 3), (6, 5, 2, 192, 4), (6, 3, 1, 320, 1)]
    feat_channels = (112, 1280)

    def __init__(self, num_classes=100, dropout=0.2):
        super().__init__()
        self.stem = conv_bn_relu6(3, 32, 3, 1)
        in_ch = 32
        stages = []
        for e, k, s, c, n in self.cfg:
            blocks = []
            for i in range(n):
                blocks.append(MBConv(in_ch, c, k, s if i == 0 else 1, e))
                in_ch = c
            stages.append(nn.Sequential(*blocks))
        self.stages_mid = nn.Sequential(*stages[:5])      # ... 112 channels, 8x8
        self.stages_end = nn.Sequential(*stages[5:])      # 192, 320 channels, 4x4
        self.head = conv_bn_relu6(320, 1280, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, num_classes))

    def forward(self, x, return_feats=False):
        f_mid = self.stages_mid(self.stem(x))
        f_final = self.head(self.stages_end(f_mid))
        logits = self.classifier(torch.flatten(self.pool(f_final), 1))
        return (logits, [f_mid, f_final]) if return_feats else logits
