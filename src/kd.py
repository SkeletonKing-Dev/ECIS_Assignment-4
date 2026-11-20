"""
Knowledge distillation framework.

Total loss for a batch (x, y):

    L = (1 - alpha) * CE(z_s, y)                                 hard labels
      + alpha * T^2 * KL( softmax(z_t / T) || softmax(z_s / T) )  soft targets
      + beta * sum_i FeatLoss( A_i(f_s_i), f_t_i )                intermediate features

  z_s, z_t   student / teacher logits
  T          temperature: T > 1 softens the teacher distribution, exposing how
             similar the teacher thinks the wrong classes are ("dark knowledge")
  T^2        keeps the soft-target gradient magnitude independent of T
             (gradients of the softened softmax scale with 1/T^2)
  f_s, f_t   student / teacher feature maps at matching depths (8x8 and 4x4)
  A_i        adapter: 1x1 conv + BN mapping student channels to teacher
             channels (FitNets-style "regressor"); spatial size is matched by
             adaptive average pooling if needed. Adapters are trained jointly
             and thrown away after training, so the deployed student is unchanged.
  FeatLoss   squared distance between the L2-normalised flattened maps
             (range 0..4). Normalising makes the loss independent of the very
             different activation scales of teacher and student.

Modes: 'logit' (beta = 0), 'feature' (alpha = 0), 'both'.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_target_loss(student_logits, teacher_logits, T):
    """Hinton et al. 2015: KL between temperature-softened distributions, times T^2."""
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * T * T


def feature_loss(f_s, f_t):
    """Squared L2 distance between normalised feature maps, averaged over the batch."""
    if f_s.shape[-2:] != f_t.shape[-2:]:
        f_s = F.adaptive_avg_pool2d(f_s, f_t.shape[-2:])
    s = F.normalize(f_s.flatten(1), dim=1)
    t = F.normalize(f_t.flatten(1), dim=1)
    return (s - t).pow(2).sum(1).mean()


class Adapter(nn.Module):
    """1x1 conv + BN: student channels -> teacher channels."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x):
        return self.proj(x)


class Distiller(nn.Module):
    def __init__(self, student, teacher, T=4.0, alpha=0.9, beta=2.0, mode="both"):
        super().__init__()
        self.student, self.teacher = student, teacher
        self.T = T
        self.alpha = 0.0 if mode == "feature" else alpha
        self.beta = 0.0 if mode == "logit" else beta
        self.mode = mode
        for p in self.teacher.parameters():          # teacher is frozen
            p.requires_grad_(False)
        self.adapters = nn.ModuleList(Adapter(cs, ct) for cs, ct in
                                      zip(student.feat_channels, teacher.feat_channels))
        self.reset_stats()

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()                          # teacher BN stats never change
        return self

    def trainable_parameters(self):
        return list(self.student.parameters()) + list(self.adapters.parameters())

    def forward(self, x):                            # inference = student only
        return self.student(x)

    def kd_loss(self, x, y):
        with torch.no_grad():
            t_logits, t_feats = self.teacher(x, return_feats=True)
        s_logits, s_feats = self.student(x, return_feats=True)
        s32, t32 = s_logits.float(), t_logits.float()

        ce = F.cross_entropy(s32, y)
        soft = soft_target_loss(s32, t32, self.T) if self.alpha > 0 else s32.new_zeros(())
        feat = s32.new_zeros(())
        if self.beta > 0:
            for adapter, fs, ft in zip(self.adapters, s_feats, t_feats):
                feat = feat + feature_loss(adapter(fs).float(), ft.float())
        loss = (1 - self.alpha) * ce + self.alpha * soft + self.beta * feat

        n = y.size(0)
        self.stats["n"] += n
        for k, v in (("ce", ce), ("soft", soft), ("feat", feat)):
            self.stats[k] += v.item() * n
        self.stats["agree"] += (s32.argmax(1) == t32.argmax(1)).sum().item()
        return loss, s_logits

    def reset_stats(self):
        self.stats = {"n": 0, "ce": 0.0, "soft": 0.0, "feat": 0.0, "agree": 0}

    def epoch_summary(self):
        n = max(1, self.stats["n"])
        s = (f"| CE {self.stats['ce'] / n:.3f}  KD {self.stats['soft'] / n:.3f}  "
             f"feat {self.stats['feat'] / n:.3f}  agree-w-teacher {100 * self.stats['agree'] / n:.1f}%")
        self.reset_stats()
        return s
