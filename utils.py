import os
import random
import time
from typing import Dict

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.total = 0.0

    def update(self, val: float, n: int = 1):
        self.total += float(val) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_device(prefer_mps: bool = False) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def structure_loss(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-6)

    pred_sigmoid = torch.sigmoid(pred)
    inter = ((pred_sigmoid * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sigmoid + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def weighted_bce_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    eps = 1e-10
    count_pos = torch.sum(gt) * 1.0 + eps
    count_neg = torch.sum(1.0 - gt) * 1.0
    beta = count_neg / count_pos
    beta_back = count_pos / (count_pos + count_neg)

    bce = torch.nn.BCEWithLogitsLoss(pos_weight=beta)
    return beta_back * bce(pred, gt)


def dice_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred_prob = torch.sigmoid(pred)
    inter = (pred_prob * gt).sum(dim=(1, 2, 3))
    denom = pred_prob.sum(dim=(1, 2, 3)) + gt.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + 1e-8) / (denom + 1e-8)).mean()


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

    probs = torch.sigmoid(logits)
    preds_bin = (probs >= 0.5).float()

    inter = (preds_bin * targets).sum(dim=(1, 2, 3))
    union = preds_bin.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - inter
    iou = (inter + 1e-8) / (union + 1e-8)

    dice = (2 * inter + 1e-8) / (preds_bin.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-8)
    mae = torch.abs(probs - targets).mean(dim=(1, 2, 3))

    return {
        "iou": iou.mean().item(),
        "dice": dice.mean().item(),
        "mae": mae.mean().item(),
    }


def save_checkpoint(state: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def time_text(sec: float) -> str:
    if sec >= 3600:
        return f"{sec / 3600:.1f}h"
    if sec >= 60:
        return f"{sec / 60:.1f}m"
    return f"{sec:.1f}s"


def tensor_to_uint8_map(t: torch.Tensor) -> np.ndarray:
    if t.ndim == 3:
        t = t[0]
    arr = t.detach().cpu().float()
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-8)
    return (arr.numpy() * 255.0).astype(np.uint8)


def save_gray_map(t: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(tensor_to_uint8_map(t), mode="L").save(path)
