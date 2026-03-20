import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode


_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
_MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
_RGB_MEAN = [0.485, 0.456, 0.406]
_RGB_STD = [0.229, 0.224, 0.225]


@dataclass
class SamplePaths:
    image: str
    depth: str
    mask: str
    edge: Optional[str] = None


def _list_images(image_dir: str) -> List[str]:
    files = [f for f in os.listdir(image_dir) if f.lower().endswith(_IMG_EXTS)]
    return sorted(files)


def _find_by_stem(root: str, stem: str, exts: Tuple[str, ...]) -> Optional[str]:
    for ext in exts:
        p = os.path.join(root, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def _resize_rgb(img: Image.Image, size_hw: Tuple[int, int]) -> torch.Tensor:
    img = TF.resize(img, size_hw, interpolation=InterpolationMode.BILINEAR)
    t = TF.to_tensor(img)
    return TF.normalize(t, mean=_RGB_MEAN, std=_RGB_STD)


def _resize_gray(img: Image.Image, size_hw: Tuple[int, int], nearest: bool = False) -> torch.Tensor:
    mode = InterpolationMode.NEAREST if nearest else InterpolationMode.BILINEAR
    img = TF.resize(img, size_hw, interpolation=mode)
    return TF.to_tensor(img)


def _normalize_depth(depth_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    depth_raw = depth_raw.clamp(0.0, 1.0)
    d_min = depth_raw.amin(dim=(1, 2), keepdim=True)
    d_max = depth_raw.amax(dim=(1, 2), keepdim=True)
    depth_raw = (depth_raw - d_min) / (d_max - d_min + 1e-6)
    depth_norm = (depth_raw - 0.5) / 0.5
    return depth_norm, depth_raw


def _mask_to_edge(mask: torch.Tensor) -> torch.Tensor:
    mask_b = mask.unsqueeze(0)
    dilated = F.max_pool2d(mask_b, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask_b, kernel_size=3, stride=1, padding=1)
    edge = (dilated - eroded > 0).float()
    return edge.squeeze(0)


def _maybe_apply_color_jitter(img: Image.Image):
    if random.random() < 0.5:
        img = TF.adjust_brightness(img, random.uniform(0.9, 1.1))
    if random.random() < 0.4:
        img = TF.adjust_contrast(img, random.uniform(0.9, 1.1))
    if random.random() < 0.3:
        img = TF.adjust_saturation(img, random.uniform(0.9, 1.1))
    return img


def _apply_depth_robustness(
    depth_raw: torch.Tensor,
    drop_prob: float = 0.0,
    noise_std: float = 0.0,
    blur_prob: float = 0.0,
) -> torch.Tensor:
    out = depth_raw
    if drop_prob > 0 and random.random() < drop_prob:
        if random.random() < 0.5:
            out = torch.zeros_like(out)
        else:
            out = out * random.uniform(0.0, 0.25)

    if blur_prob > 0 and random.random() < blur_prob:
        out = F.avg_pool2d(out.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)

    if noise_std > 0:
        out = torch.clamp(out + torch.randn_like(out) * noise_std, 0.0, 1.0)
    return out


class RGBDTrainDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        depth_dir: str,
        mask_dir: str,
        edge_dir: Optional[str] = None,
        image_size: int = 384,
        augment: bool = True,
        depth_drop_prob: float = 0.0,
        depth_noise_std: float = 0.0,
        depth_blur_prob: float = 0.0,
    ):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.mask_dir = mask_dir
        self.edge_dir = edge_dir
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.depth_drop_prob = float(depth_drop_prob)
        self.depth_noise_std = float(depth_noise_std)
        self.depth_blur_prob = float(depth_blur_prob)

        self.samples: List[SamplePaths] = []
        for img_name in _list_images(image_dir):
            stem = os.path.splitext(img_name)[0]
            img_path = os.path.join(image_dir, img_name)
            depth_path = _find_by_stem(depth_dir, stem, _MASK_EXTS)
            mask_path = _find_by_stem(mask_dir, stem, _MASK_EXTS)
            edge_path = _find_by_stem(edge_dir, stem, _MASK_EXTS) if edge_dir else None
            if depth_path is None or mask_path is None:
                continue
            self.samples.append(SamplePaths(image=img_path, depth=depth_path, mask=mask_path, edge=edge_path))

        if not self.samples:
            raise RuntimeError(f"未在 {image_dir} 找到可用 RGB-D 训练样本")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        rgb = Image.open(sample.image).convert("RGB")
        depth = Image.open(sample.depth).convert("L")
        mask = Image.open(sample.mask).convert("L")
        edge = Image.open(sample.edge).convert("L") if sample.edge else None

        if self.augment and random.random() < 0.5:
            rgb = TF.hflip(rgb)
            depth = TF.hflip(depth)
            mask = TF.hflip(mask)
            if edge is not None:
                edge = TF.hflip(edge)

        if self.augment:
            rgb = _maybe_apply_color_jitter(rgb)

        size_hw = (self.image_size, self.image_size)
        rgb_t = _resize_rgb(rgb, size_hw)
        depth_raw = _resize_gray(depth, size_hw, nearest=False)
        mask_t = (_resize_gray(mask, size_hw, nearest=True) > 0.5).float()

        depth_raw = _apply_depth_robustness(
            depth_raw=depth_raw,
            drop_prob=self.depth_drop_prob,
            noise_std=self.depth_noise_std,
            blur_prob=self.depth_blur_prob,
        )
        depth_t, depth_raw = _normalize_depth(depth_raw)

        if edge is not None:
            edge_t = (_resize_gray(edge, size_hw, nearest=True) > 0.5).float()
        else:
            edge_t = _mask_to_edge(mask_t)

        name = os.path.splitext(os.path.basename(sample.image))[0]
        return {
            "image": rgb_t,
            "depth": depth_t,
            "depth_raw": depth_raw,
            "mask": mask_t,
            "edge": edge_t,
            "name": name,
        }


class RGBDEvalDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        depth_dir: str,
        mask_dir: str,
        image_size: int = 384,
    ):
        self.image_dir = image_dir
        self.depth_dir = depth_dir
        self.mask_dir = mask_dir
        self.image_size = int(image_size)

        self.samples: List[SamplePaths] = []
        for img_name in _list_images(image_dir):
            stem = os.path.splitext(img_name)[0]
            img_path = os.path.join(image_dir, img_name)
            depth_path = _find_by_stem(depth_dir, stem, _MASK_EXTS)
            mask_path = _find_by_stem(mask_dir, stem, _MASK_EXTS)
            if depth_path is None or mask_path is None:
                continue
            self.samples.append(SamplePaths(image=img_path, depth=depth_path, mask=mask_path))

        if not self.samples:
            raise RuntimeError(f"未在 {image_dir} 找到可用 RGB-D 评测样本")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        rgb = Image.open(sample.image).convert("RGB")
        depth = Image.open(sample.depth).convert("L")
        mask = Image.open(sample.mask).convert("L")

        size_hw = (self.image_size, self.image_size)
        rgb_t = _resize_rgb(rgb, size_hw)
        depth_resized = _resize_gray(depth, size_hw, nearest=False)
        depth_t, depth_raw = _normalize_depth(depth_resized)

        mask_t = TF.to_tensor(mask)
        mask_t = (mask_t > 0.5).float()

        name = os.path.splitext(os.path.basename(sample.image))[0]
        return {
            "image": rgb_t,
            "depth": depth_t,
            "depth_raw": depth_raw,
            "mask": mask_t,
            "name": name,
        }
