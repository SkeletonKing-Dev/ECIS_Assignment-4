"""
Shared code for Assignment 4: paths, datasets (CIFAR-100 for knowledge
distillation, CIFAR-10 for NAS), a fast on-device data loader, training /
evaluation helpers and checkpoint utilities.
"""
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")   # hide library deprecation warnings in the logs

# ----------------------------------------------------------------------------
# Paths (relative to the project root, so scripts run from any directory)
# ----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
ONNX_DIR = os.path.join(ROOT, "onnx_models")
RESULT_DIR = os.path.join(ROOT, "results")
LOG_DIR = os.path.join(ROOT, "logs")

STUDENTS = ["mobilenetv2", "squeezenet", "efficientnet_lite0"]
TEACHER_PATH = os.path.join(MODEL_DIR, "teacher_resnet50_cifar100.pth")
NAS_BEST_PATH = os.path.join(MODEL_DIR, "nas_best_cifar10.pth")
NAS_BEST_JSON = os.path.join(RESULT_DIR, "nas_best_arch.json")


def student_path(arch, kind):
    """kind: 'baseline' (trained alone) or 'kd' (distilled)."""
    return os.path.join(MODEL_DIR, f"student_{arch}_{kind}_cifar100.pth")


def ensure_dirs():
    for d in (MODEL_DIR, ONNX_DIR, RESULT_DIR, LOG_DIR):
        os.makedirs(d, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------
DATASETS = {
    # name: (torchvision class name, folder created by torchvision, mean, std)
    "cifar10": ("CIFAR10", "cifar-10-batches-py",
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ("CIFAR100", "cifar-100-python",
                 (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


def find_data_root(name):
    """Reuse an existing copy if present (torchvision then verifies it and skips
    the download). Order: $CIFAR10_ROOT / $CIFAR100_ROOT, data/<name>, data."""
    folder = DATASETS[name][1]
    candidates = [os.environ.get(f"{name.upper()}_ROOT"),
                  os.path.join(ROOT, "data", name),
                  os.path.join(ROOT, "data")]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, folder)):
            return c
    return os.path.join(ROOT, "data", name)


def load_arrays(name, train, fake=False):
    """Returns (images uint8 [N,32,32,3], labels int64 [N], class names)."""
    if fake:   # random data so the pipeline can be smoke-tested offline
        n_cls = 100 if name == "cifar100" else 10
        g = np.random.default_rng(0 if train else 1)
        n = 2000 if train else 500
        return (g.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8),
                g.integers(0, n_cls, n).astype(np.int64),
                [f"class_{i}" for i in range(n_cls)])
    import torchvision
    cls = getattr(torchvision.datasets, DATASETS[name][0])
    ds = cls(find_data_root(name), train=train, download=True)
    return ds.data, np.asarray(ds.targets, dtype=np.int64), list(ds.classes)


def normalize_stats(name):
    return DATASETS[name][2], DATASETS[name][3]


class DeviceLoader:
    """Keeps the whole uint8 dataset on the device (GPU if available) and does
    augmentation (random crop with 4-pixel padding + horizontal flip) and
    normalisation on the device in batches. Much faster than a PIL/DataLoader
    pipeline for 32x32 images, which matters with many short trainings (NAS).
    resolution: if set, batches are resized to resolution x resolution."""

    def __init__(self, images, labels, dataset, batch_size, train, device,
                 resolution=None, drop_last=None):
        self.x = torch.from_numpy(np.ascontiguousarray(images)).to(device).permute(0, 3, 1, 2).contiguous()
        self.y = torch.as_tensor(labels, dtype=torch.long, device=device)
        mean, std = normalize_stats(dataset)
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.bs, self.train, self.device = batch_size, train, device
        self.resolution = resolution
        self.drop_last = train if drop_last is None else drop_last

    def __len__(self):
        n = len(self.y)
        return n // self.bs if self.drop_last else (n + self.bs - 1) // self.bs

    @staticmethod
    def random_crop_flip(x, pad=4):
        b, c, h, w = x.shape
        p = F.pad(x, (pad, pad, pad, pad))
        oy = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
        ox = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
        rows = (oy[:, None] + torch.arange(h, device=x.device))[:, :, None]       # [B,H,1]
        cols = (ox[:, None] + torch.arange(w, device=x.device))[:, None, :]       # [B,1,W]
        bidx = torch.arange(b, device=x.device)[:, None, None]
        out = p[bidx, :, rows, cols].permute(0, 3, 1, 2)                         # [B,C,H,W]
        flip = torch.rand(b, device=x.device) < 0.5
        return torch.where(flip[:, None, None, None], out.flip(3), out)

    def __iter__(self):
        n = len(self.y)
        idx = torch.randperm(n, device=self.device) if self.train else torch.arange(n, device=self.device)
        for i in range(len(self)):
            j = idx[i * self.bs:(i + 1) * self.bs]
            xb = self.x[j].float().div_(255)
            if self.train:
                xb = self.random_crop_flip(xb)
            xb = (xb - self.mean) / self.std
            if self.resolution and self.resolution != xb.shape[-1]:
                xb = resize(xb, self.resolution)
            yield xb, self.y[j]


def resize(x, res):
    return F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=True)


def get_loaders(dataset, batch_size=128, device="cpu", fake=False, resolution=None):
    """Full train set + test set for the final trainings."""
    xtr, ytr, classes = load_arrays(dataset, True, fake)
    xte, yte, _ = load_arrays(dataset, False, fake)
    train = DeviceLoader(xtr, ytr, dataset, batch_size, True, device, resolution)
    test = DeviceLoader(xte, yte, dataset, 500, False, device, resolution)
    return train, test, classes


def preprocess_numpy(images_uint8, dataset, resolution=None):
    """uint8 [N,32,32,3] -> normalised float tensor [N,3,R,R] (CPU), same as eval."""
    x = torch.from_numpy(np.ascontiguousarray(images_uint8)).permute(0, 3, 1, 2).float() / 255
    mean, std = normalize_stats(dataset)
    x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    if resolution and resolution != x.shape[-1]:
        x = resize(x, resolution)
    return x


# ----------------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------------
def make_optimizer(model_or_params, lr, epochs, steps_per_epoch, wd=5e-4):
    params = model_or_params.parameters() if isinstance(model_or_params, nn.Module) else model_or_params
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs,
                                                steps_per_epoch=steps_per_epoch, pct_start=0.25)
    return opt, sched


def autocast(device):
    """Mixed precision on GPU (faster training), plain FP32 on CPU."""
    return torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda")


def train_epoch(model, loader, opt, sched, device, scaler, loss_fn=None):
    """One epoch of supervised training. loss_fn(model, x, y) -> (loss, logits)
    lets the KD script reuse this loop with its own loss."""
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        opt.zero_grad(set_to_none=True)
        with autocast(device):
            if loss_fn is None:
                out = model(x)
                loss = F.cross_entropy(out, y)
            else:
                loss, out = loss_fn(model, x, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device=None, return_preds=False):
    model.eval()
    correct, total, preds = 0, 0, []
    for x, y in loader:
        if device is not None:
            x, y = x.to(device), y.to(device)
        p = model(x).argmax(1)
        correct += (p == y).sum().item()
        total += y.size(0)
        if return_preds:
            preds.append(p.cpu())
    acc = 100.0 * correct / total
    return (acc, torch.cat(preds).numpy()) if return_preds else acc


def fit(model, train_loader, test_loader, device, epochs, lr, tag="", wd=5e-4, loss_fn=None, params=None,
        epoch_hook=None):
    """Fixed-schedule training (One-Cycle LR). The weights after the LAST epoch
    are kept: no checkpoint is selected on the test set. Test accuracy is
    printed for monitoring only."""
    opt, sched = make_optimizer(params if params is not None else model, lr, epochs, len(train_loader), wd)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    t0 = time.time()
    acc = None
    for ep in range(epochs):
        loss, tr_acc = train_epoch(model, train_loader, opt, sched, device, scaler, loss_fn)
        acc = evaluate(model, test_loader)
        extra = f"  {epoch_hook()}" if epoch_hook else ""
        print(f"[{tag}] epoch {ep + 1:3d}/{epochs}  loss {loss:.4f}  train {tr_acc:.2f}%  "
              f"test {acc:.2f}%  ({(time.time() - t0) / 60:.1f} min){extra}", flush=True)
    return acc


# ----------------------------------------------------------------------------
# Checkpoints: every checkpoint stores what is needed to rebuild the model
# ----------------------------------------------------------------------------
def save_checkpoint(path, model, arch, dataset, classes, **extra):
    torch.save({"arch": arch, "dataset": dataset, "num_classes": len(classes), "classes": classes,
                "config": getattr(model, "config", None), "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path, map_location="cpu"):
    """Returns (model in eval mode, checkpoint dict)."""
    from networks import build_model
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if ckpt["arch"] == "nas":
        from nas_space import build_from_config
        model = build_from_config(ckpt["config"], ckpt["num_classes"])
    else:
        model = build_model(ckpt["arch"], ckpt["num_classes"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt


def file_size_mb(path):
    return os.path.getsize(path) / 2 ** 20


# ----------------------------------------------------------------------------
# Catalog of deployable models (name -> checkpoint path)
# ----------------------------------------------------------------------------
def model_catalog(include_teacher=True, only_existing=True):
    cat = {}
    if include_teacher:
        cat["teacher_resnet50"] = TEACHER_PATH
    for arch in STUDENTS:
        for kind in ("baseline", "kd"):
            cat[f"{arch}_{kind}"] = student_path(arch, kind)
    cat["nas_best"] = NAS_BEST_PATH
    if only_existing:
        cat = {k: v for k, v in cat.items() if os.path.exists(v)}
    return cat


def onnx_path(name):
    return os.path.join(ONNX_DIR, f"{name}.onnx")
