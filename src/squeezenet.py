"""
Student 2: SqueezeNet 1.1 (Iandola et al., 2016).

Building block: Fire module = 1x1 "squeeze" conv (few channels) followed by
parallel 1x1 and 3x3 "expand" convs whose outputs are concatenated.

CIFAR adaptation / modifications (documented in the report):
  * first conv is 3x3 stride 1 (instead of 3x3 stride 2) for 32x32 input
  * BatchNorm after every conv: the original has none, and training it from
    scratch at a high learning rate is unstable without it
  * the classifier is a 1x1 conv to num_classes + global average pooling, as
    in the original, but without the final ReLU (which would force all logits
    to be >= 0 and hurt the softmax / distillation targets)

Feature maps used for distillation:
  after fire5: 256 x 8 x 8
  after fire9: 512 x 4 x 4
"""
import torch
import torch.nn as nn


def conv_bn_relu(in_ch, out_ch, k, stride=1, pad=0):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, k, stride, pad, bias=False),
                         nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))


class Fire(nn.Module):
    def __init__(self, in_ch, squeeze, expand):
        super().__init__()
        self.squeeze = conv_bn_relu(in_ch, squeeze, 1)
        self.expand1 = conv_bn_relu(squeeze, expand, 1)
        self.expand3 = conv_bn_relu(squeeze, expand, 3, pad=1)

    def forward(self, x):
        s = self.squeeze(x)
        return torch.cat([self.expand1(s), self.expand3(s)], 1)


class SqueezeNet(nn.Module):
    feat_channels = (256, 512)

    def __init__(self, num_classes=100, dropout=0.5):
        super().__init__()
        self.stem = conv_bn_relu(3, 64, 3, 1, 1)                              # 32x32
        self.block_mid = nn.Sequential(
            nn.MaxPool2d(3, 2, 1), Fire(64, 16, 64), Fire(128, 16, 64),       # 16x16
            nn.MaxPool2d(3, 2, 1), Fire(128, 32, 128), Fire(256, 32, 128))    # 8x8 -> 256 ch
        self.block_end = nn.Sequential(
            nn.MaxPool2d(3, 2, 1), Fire(256, 48, 192), Fire(384, 48, 192),    # 4x4
            Fire(384, 64, 256), Fire(512, 64, 256))                           # 512 ch
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Conv2d(512, num_classes, 1))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, return_feats=False):
        f_mid = self.block_mid(self.stem(x))
        f_final = self.block_end(f_mid)
        logits = torch.flatten(self.pool(self.classifier(f_final)), 1)
        return (logits, [f_mid, f_final]) if return_feats else logits
