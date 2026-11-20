"""
Q2: searchable CNN architecture space and the architecture generator.

An architecture is a JSON-serialisable config ("genome"):

{
  "resolution": 28,              # input resolution (images are resized to RxR)
  "stem_channels": 24,           # 3x3 stem conv
  "head_channels": 256,          # 1x1 conv before pooling (0 = no head conv)
  "stages": [                    # 4 stages, strides 1, 2, 2, 2
    {"channels": 32, "depth": 2, "activation": "hswish",
     "blocks": [{"type": "mbconv6", "kernel": 5}, {"type": "dwsep", "kernel": 3},
                {"type": "conv", "kernel": 3}]},     # only the first `depth` blocks are used
    ...
  ]
}

Searchable dimensions (as required):
  variable depth         -> depth per stage (1-3 blocks, 4-12 blocks in total)
  variable channels      -> stem, per-stage and head channels
  kernel sizes           -> 3, 5, 7 per block
  depthwise separable    -> block types 'dwsep' (depthwise kxk + pointwise 1x1) and
                            'mbconv3' / 'mbconv6' (inverted residual with depthwise conv)
                            besides a standard 'conv' block
  activation functions   -> relu, relu6, hardswish, silu per stage
  input resolution       -> 24, 28, 32

build_from_config() is the architecture generator: it turns any config into
an nn.Module. sample_config() and mutate() are used by the search.
"""
import copy
import json
import math
import random

import torch
import torch.nn as nn

SPACE = {
    "resolution": [24, 28, 32],
    "stem_channels": [16, 24, 32],
    "head_channels": [0, 256, 512],
    "stage_channels": [[16, 24, 32], [32, 48, 64], [64, 96, 128], [128, 160, 192, 256]],
    "depth": [1, 2, 3],
    "activation": ["relu", "relu6", "hswish", "silu"],
    "block_type": ["conv", "dwsep", "mbconv3", "mbconv6"],
    "kernel": [3, 5, 7],
}
STAGE_STRIDES = [1, 2, 2, 2]
MAX_DEPTH = max(SPACE["depth"])

ACTIVATIONS = {"relu": nn.ReLU, "relu6": nn.ReLU6, "hswish": nn.Hardswish, "silu": nn.SiLU}


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------
def conv_bn(in_ch, out_ch, k, stride=1, groups=1, act=None):
    layers = [nn.Conv2d(in_ch, out_ch, k, stride, k // 2, groups=groups, bias=False), nn.BatchNorm2d(out_ch)]
    if act is not None:
        layers.append(ACTIVATIONS[act]())
    return nn.Sequential(*layers)


class Block(nn.Module):
    """One searchable block. A residual connection is added whenever the
    stride is 1 and the channel count does not change."""

    def __init__(self, btype, in_ch, out_ch, k, stride, act):
        super().__init__()
        if btype == "conv":                         # standard kxk convolution
            self.body = conv_bn(in_ch, out_ch, k, stride, act=act)
        elif btype == "dwsep":                      # depthwise separable (MobileNetV1)
            self.body = nn.Sequential(conv_bn(in_ch, in_ch, k, stride, groups=in_ch, act=act),
                                      conv_bn(in_ch, out_ch, 1, act=act))
        elif btype.startswith("mbconv"):            # inverted residual (MobileNetV2), expansion 3 or 6
            hidden = in_ch * int(btype[-1])
            self.body = nn.Sequential(conv_bn(in_ch, hidden, 1, act=act),
                                      conv_bn(hidden, hidden, k, stride, groups=hidden, act=act),
                                      conv_bn(hidden, out_ch, 1))          # linear projection
        else:
            raise ValueError(btype)
        self.residual = stride == 1 and in_ch == out_ch

    def forward(self, x):
        return x + self.body(x) if self.residual else self.body(x)


class NASNet(nn.Module):
    def __init__(self, config, num_classes=10):
        super().__init__()
        self.config = copy.deepcopy(config)
        self.resolution = config["resolution"]
        act0 = config["stages"][0]["activation"]
        layers = [conv_bn(3, config["stem_channels"], 3, 1, act=act0)]
        in_ch = config["stem_channels"]
        for stage, stride in zip(config["stages"], STAGE_STRIDES):
            for i, b in enumerate(stage["blocks"][:stage["depth"]]):
                layers.append(Block(b["type"], in_ch, stage["channels"], b["kernel"],
                                    stride if i == 0 else 1, stage["activation"]))
                in_ch = stage["channels"]
        if config["head_channels"]:
            layers.append(conv_bn(in_ch, config["head_channels"], 1, act=config["stages"][-1]["activation"]))
            in_ch = config["head_channels"]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(in_ch, num_classes))

    def forward(self, x):
        return self.classifier(torch.flatten(self.pool(self.features(x)), 1))


def build_from_config(config, num_classes=10):
    """Architecture generator: config -> PyTorch model."""
    return NASNet(config, num_classes)


# ----------------------------------------------------------------------------
# Sampling and mutation
# ----------------------------------------------------------------------------
def random_block(rng):
    return {"type": rng.choice(SPACE["block_type"]), "kernel": rng.choice(SPACE["kernel"])}


def sample_config(rng=random):
    return {
        "resolution": rng.choice(SPACE["resolution"]),
        "stem_channels": rng.choice(SPACE["stem_channels"]),
        "head_channels": rng.choice(SPACE["head_channels"]),
        "stages": [{"channels": rng.choice(SPACE["stage_channels"][i]),
                    "depth": rng.choice(SPACE["depth"]),
                    "activation": rng.choice(SPACE["activation"]),
                    "blocks": [random_block(rng) for _ in range(MAX_DEPTH)]}
                   for i in range(len(STAGE_STRIDES))],
    }


def mutate(config, rng=random):
    """Changes exactly one gene to a different value. Returns (child, description)."""
    child = copy.deepcopy(config)

    def resample(options, current):
        return rng.choice([o for o in options if o != current])

    gene = rng.choice(["resolution", "stem", "head", "channels", "depth", "activation",
                       "block_type", "kernel", "block_type", "kernel"])   # block genes more often
    s = rng.randrange(len(child["stages"]))
    st = child["stages"][s]
    if gene == "resolution":
        child["resolution"] = resample(SPACE["resolution"], child["resolution"])
        return child, f"resolution -> {child['resolution']}"
    if gene == "stem":
        child["stem_channels"] = resample(SPACE["stem_channels"], child["stem_channels"])
        return child, f"stem -> {child['stem_channels']}"
    if gene == "head":
        child["head_channels"] = resample(SPACE["head_channels"], child["head_channels"])
        return child, f"head -> {child['head_channels']}"
    if gene == "channels":
        st["channels"] = resample(SPACE["stage_channels"][s], st["channels"])
        return child, f"stage{s + 1}.channels -> {st['channels']}"
    if gene == "depth":
        st["depth"] = resample(SPACE["depth"], st["depth"])
        return child, f"stage{s + 1}.depth -> {st['depth']}"
    if gene == "activation":
        st["activation"] = resample(SPACE["activation"], st["activation"])
        return child, f"stage{s + 1}.activation -> {st['activation']}"
    b = rng.randrange(st["depth"])                  # mutate an active block
    if gene == "block_type":
        st["blocks"][b]["type"] = resample(SPACE["block_type"], st["blocks"][b]["type"])
        return child, f"stage{s + 1}.block{b + 1}.type -> {st['blocks'][b]['type']}"
    st["blocks"][b]["kernel"] = resample(SPACE["kernel"], st["blocks"][b]["kernel"])
    return child, f"stage{s + 1}.block{b + 1}.kernel -> {st['blocks'][b]['kernel']}"


def active_config(config):
    """Config with unused blocks removed (canonical form: two configs that
    build the same network get the same key)."""
    c = copy.deepcopy(config)
    for st in c["stages"]:
        st["blocks"] = st["blocks"][:st["depth"]]
    return c


def config_key(config):
    return json.dumps(active_config(config), sort_keys=True)


def describe(config):
    """Compact one-line description, e.g. 'R28 s24 | 32x2 hswish [mb6k5,ds3] | ... | h256'."""
    short = {"conv": "c", "dwsep": "ds", "mbconv3": "mb3", "mbconv6": "mb6"}
    parts = [f"R{config['resolution']} s{config['stem_channels']}"]
    for st in config["stages"]:
        blocks = ",".join(f"{short[b['type']]}k{b['kernel']}" for b in st["blocks"][:st["depth"]])
        parts.append(f"{st['channels']}x{st['depth']} {st['activation']} [{blocks}]")
    parts.append(f"h{config['head_channels']}")
    return " | ".join(parts)


def space_size():
    """Number of distinct networks in the search space."""
    n_blocks = len(SPACE["block_type"]) * len(SPACE["kernel"])
    total = len(SPACE["resolution"]) * len(SPACE["stem_channels"]) * len(SPACE["head_channels"])
    for chans in SPACE["stage_channels"]:
        total *= len(chans) * len(SPACE["activation"]) * sum(n_blocks ** d for d in SPACE["depth"])
    return total


if __name__ == "__main__":
    print(f"Search space size: {space_size():.3e} architectures (log10 = {math.log10(space_size()):.1f})")
    cfg = sample_config(random.Random(0))
    print(describe(cfg))
    m = build_from_config(cfg)
    print(m(torch.randn(2, 3, cfg["resolution"], cfg["resolution"])).shape)
