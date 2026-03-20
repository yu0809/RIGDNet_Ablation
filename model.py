import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


_BACKBONE_DIMS = {
    "resnet18": [64, 128, 256, 512],
    "resnet34": [64, 128, 256, 512],
    "resnet50": [256, 512, 1024, 2048],
}

_TV_WEIGHTS = {
    "resnet18": models.ResNet18_Weights.DEFAULT,
    "resnet34": models.ResNet34_Weights.DEFAULT,
    "resnet50": models.ResNet50_Weights.DEFAULT,
}


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        if key.startswith("backbone."):
            key = key[len("backbone.") :]
        out[key] = value
    return out


def _load_resnet_reference_state(backbone_name: str, weights_path: Optional[str]) -> Optional[Dict[str, torch.Tensor]]:
    if backbone_name not in _BACKBONE_DIMS:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    if weights_path:
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"预训练权重不存在: {weights_path}")
        raw = torch.load(weights_path, map_location="cpu")
        if isinstance(raw, dict):
            if "state_dict" in raw:
                raw = raw["state_dict"]
            elif "model" in raw:
                raw = raw["model"]
        return _clean_state_dict(raw)

    try:
        model = getattr(models, backbone_name)(weights=_TV_WEIGHTS[backbone_name])
        return model.state_dict()
    except Exception as exc:
        print(f"[RIGDNet][WARN] load pretrained backbone failed: {exc}")
        return None


def _adapt_first_conv(weight: torch.Tensor, in_channels: int) -> torch.Tensor:
    if weight.shape[1] == in_channels:
        return weight
    if in_channels == 1:
        return weight.mean(dim=1, keepdim=True)

    repeat = (in_channels + weight.shape[1] - 1) // weight.shape[1]
    weight = weight.repeat(1, repeat, 1, 1)[:, :in_channels]
    return weight / float(repeat)


def _constant_map_like(x: torch.Tensor, value: float) -> torch.Tensor:
    return torch.full(
        (x.shape[0], 1, x.shape[2], x.shape[3]),
        float(value),
        device=x.device,
        dtype=x.dtype,
    )


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualRefineBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, 3)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        return self.relu(x + identity)


class EvidenceProjector(nn.Module):
    def __init__(self, in_channels: int = 4, hidden_channels: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TaskDrivenDepthRectifier(nn.Module):
    def __init__(self, hidden_channels: int = 32, max_delta: float = 0.25):
        super().__init__()
        self.max_delta = float(max_delta)
        self.rgb_stem = nn.Sequential(
            ConvBNAct(3, hidden_channels, kernel_size=3),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3),
        )
        self.depth_stem = nn.Sequential(
            ConvBNAct(1, hidden_channels, kernel_size=3),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3),
        )
        self.guidance_mixer = nn.Sequential(
            ConvBNAct(hidden_channels * 2 + 4, hidden_channels, kernel_size=3),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3),
        )
        self.rectify_gate_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.rectify_residual_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)

    def forward(
        self,
        rgb: torch.Tensor,
        depth_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        rgb_gray = rgb.mean(dim=1, keepdim=True)
        rgb_edge = self._gradient_magnitude(rgb_gray)
        depth_edge = self._gradient_magnitude(depth_raw)

        rgb_feat = self.rgb_stem(rgb)
        depth_feat = self.depth_stem(depth_raw)
        guidance = self.guidance_mixer(torch.cat([rgb_feat, depth_feat, rgb_gray, depth_raw, rgb_edge, depth_edge], dim=1))

        rectify_gate = self.rectify_gate_head(guidance)
        rectify_residual = self.max_delta * torch.tanh(self.rectify_residual_head(guidance))
        rectified_depth_raw = torch.clamp(depth_raw + rectify_gate * rectify_residual, 0.0, 1.0)
        rectified_depth = (rectified_depth_raw - 0.5) / 0.5
        rectify_strength = torch.abs(rectify_gate * rectify_residual) / max(self.max_delta, 1e-6)
        rectified_depth_edge = self._gradient_magnitude(rectified_depth_raw)

        explain = {
            "rectify_gate": rectify_gate,
            "rectify_strength": rectify_strength,
            "rectify_residual": rectify_residual,
            "rectified_depth_edge": rectified_depth_edge,
        }
        return rectified_depth, rectified_depth_raw, explain


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        in_channels: int = 3,
        pretrained_state: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        if backbone_name not in _BACKBONE_DIMS:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        base = getattr(models, backbone_name)(weights=None)
        if in_channels != 3:
            old_conv = base.conv1
            base.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

        if pretrained_state is not None:
            adapted = dict(pretrained_state)
            if "conv1.weight" in adapted:
                adapted["conv1.weight"] = _adapt_first_conv(adapted["conv1.weight"], in_channels)
            missing, unexpected = base.load_state_dict(adapted, strict=False)
            if unexpected:
                print(f"[RIGDNet][WARN] unexpected pretrained keys: {len(unexpected)}")
            if missing:
                print(f"[RIGDNet][INFO] missing pretrained keys: {len(missing)}")

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]


class InterpretableFusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.rgb_proj = ConvBNAct(in_channels, out_channels, kernel_size=1, padding=0)
        self.depth_proj = ConvBNAct(in_channels, out_channels, kernel_size=1, padding=0)

        self.rgb_confidence_head = nn.Sequential(
            ConvBNAct(out_channels * 2, out_channels, kernel_size=3),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.depth_confidence_head = nn.Sequential(
            ConvBNAct(out_channels * 2 + 1, out_channels, kernel_size=3),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.consistency_head = nn.Sequential(
            ConvBNAct(out_channels * 4, out_channels, kernel_size=3),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.boundary_head = nn.Sequential(
            ConvBNAct(4, max(out_channels // 2, 16), kernel_size=3),
            nn.Conv2d(max(out_channels // 2, 16), 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.context_prior = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 4, max(out_channels // 2, 16), kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(out_channels // 2, 16), 3, kernel_size=1),
        )
        self.rgb_evidence = EvidenceProjector()
        self.depth_evidence = EvidenceProjector()
        self.shared_evidence = EvidenceProjector()
        self.shared_context = nn.Sequential(
            ConvBNAct(out_channels * 4, out_channels, kernel_size=3),
            ConvBNAct(out_channels, out_channels, kernel_size=3),
        )
        self.refine = ResidualRefineBlock(out_channels)

        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3), persistent=False)

    def _gradient_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)

    def forward(
        self,
        rgb_feat: torch.Tensor,
        depth_feat: torch.Tensor,
        depth_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rgb_proj = self.rgb_proj(rgb_feat)
        depth_proj = self.depth_proj(depth_feat)
        depth_raw = F.interpolate(depth_raw, size=rgb_proj.shape[-2:], mode="bilinear", align_corners=False)
        discrepancy = torch.abs(rgb_proj - depth_proj)
        interaction = rgb_proj * depth_proj

        rgb_confidence = self.rgb_confidence_head(torch.cat([rgb_proj, discrepancy], dim=1))
        depth_confidence = self.depth_confidence_head(torch.cat([depth_proj, discrepancy, depth_raw], dim=1))
        cross_modal_consistency = self.consistency_head(
            torch.cat([rgb_proj, depth_proj, interaction, discrepancy], dim=1)
        )

        discrepancy_energy = discrepancy.mean(dim=1, keepdim=True)
        depth_edge = self._gradient_magnitude(depth_raw)
        boundary_uncertainty = self.boundary_head(
            torch.cat(
                [
                    depth_edge,
                    discrepancy_energy,
                    torch.abs(rgb_confidence - depth_confidence),
                    1.0 - cross_modal_consistency,
                ],
                dim=1,
            )
        )

        context_prior = self.context_prior(torch.cat([rgb_proj, depth_proj, discrepancy, interaction], dim=1))
        rgb_prior, depth_prior, shared_prior = torch.chunk(context_prior, 3, dim=1)

        rgb_route_factors = torch.cat(
            [
                rgb_confidence,
                1.0 - depth_confidence,
                boundary_uncertainty,
                1.0 - cross_modal_consistency,
            ],
            dim=1,
        )
        depth_route_factors = torch.cat(
            [
                depth_confidence,
                cross_modal_consistency,
                1.0 - boundary_uncertainty,
                depth_confidence * cross_modal_consistency,
            ],
            dim=1,
        )
        shared_route_factors = torch.cat(
            [
                cross_modal_consistency,
                0.5 * (rgb_confidence + depth_confidence),
                1.0 - boundary_uncertainty,
                rgb_confidence * depth_confidence,
            ],
            dim=1,
        )

        rgb_evidence = F.softplus(self.rgb_evidence(rgb_route_factors) + rgb_prior)
        depth_evidence = F.softplus(self.depth_evidence(depth_route_factors) + depth_prior)
        shared_evidence = F.softplus(self.shared_evidence(shared_route_factors) + shared_prior)
        evidence = torch.cat([rgb_evidence, depth_evidence, shared_evidence], dim=1)
        gates = evidence / (evidence.sum(dim=1, keepdim=True) + 1e-6)
        rgb_gate, depth_gate, shared_gate = torch.chunk(gates, 3, dim=1)

        shared_feature = self.shared_context(torch.cat([rgb_proj, depth_proj, interaction, discrepancy], dim=1))
        fused = rgb_gate * rgb_proj + depth_gate * depth_proj + shared_gate * shared_feature
        fused = self.refine(fused)

        gate_entropy = -(gates * torch.log(gates + 1e-6)).sum(dim=1, keepdim=True) / math.log(3.0)

        explain = {
            "rgb_gate": rgb_gate,
            "depth_gate": depth_gate,
            "shared_gate": shared_gate,
            "rgb_confidence": rgb_confidence,
            "depth_confidence": depth_confidence,
            "cross_modal_consistency": cross_modal_consistency,
            "boundary_uncertainty": boundary_uncertainty,
            "gate_entropy": gate_entropy,
            "rgb_evidence": rgb_evidence,
            "depth_evidence": depth_evidence,
            "shared_evidence": shared_evidence,
        }
        return fused, explain


class SimpleGateFusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.rgb_proj = ConvBNAct(in_channels, out_channels, kernel_size=1, padding=0)
        self.depth_proj = ConvBNAct(in_channels, out_channels, kernel_size=1, padding=0)
        self.gate_head = nn.Sequential(
            ConvBNAct(out_channels * 3, out_channels, kernel_size=3),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.refine = ResidualRefineBlock(out_channels)

    def forward(
        self,
        rgb_feat: torch.Tensor,
        depth_feat: torch.Tensor,
        depth_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rgb_proj = self.rgb_proj(rgb_feat)
        depth_proj = self.depth_proj(depth_feat)
        discrepancy = torch.abs(rgb_proj - depth_proj)

        rgb_gate = self.gate_head(torch.cat([rgb_proj, depth_proj, discrepancy], dim=1))
        depth_gate = 1.0 - rgb_gate
        shared_gate = _constant_map_like(rgb_gate, 1e-6)

        gates = torch.cat([rgb_gate, depth_gate, shared_gate], dim=1)
        gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-6)
        rgb_gate, depth_gate, shared_gate = torch.chunk(gates, 3, dim=1)

        fused = self.refine(rgb_gate * rgb_proj + depth_gate * depth_proj)

        discrepancy_energy = torch.tanh(discrepancy.mean(dim=1, keepdim=True))
        cross_modal_consistency = 1.0 - discrepancy_energy
        boundary_uncertainty = discrepancy_energy
        gate_entropy = -(gates * torch.log(gates + 1e-6)).sum(dim=1, keepdim=True) / math.log(3.0)

        explain = {
            "rgb_gate": rgb_gate,
            "depth_gate": depth_gate,
            "shared_gate": shared_gate,
            "rgb_confidence": rgb_gate,
            "depth_confidence": depth_gate,
            "cross_modal_consistency": cross_modal_consistency,
            "boundary_uncertainty": boundary_uncertainty,
            "gate_entropy": gate_entropy,
            "rgb_evidence": rgb_gate,
            "depth_evidence": depth_gate,
            "shared_evidence": shared_gate,
        }
        return fused, explain


class ConcatFusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.rgb_proj = ConvBNAct(in_channels, out_channels, kernel_size=1, padding=0)
        self.depth_proj = ConvBNAct(in_channels, out_channels, kernel_size=1, padding=0)
        self.mixer = nn.Sequential(
            ConvBNAct(out_channels * 2, out_channels, kernel_size=3),
            ConvBNAct(out_channels, out_channels, kernel_size=3),
        )
        self.refine = ResidualRefineBlock(out_channels)

    def forward(
        self,
        rgb_feat: torch.Tensor,
        depth_feat: torch.Tensor,
        depth_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rgb_proj = self.rgb_proj(rgb_feat)
        depth_proj = self.depth_proj(depth_feat)
        discrepancy = torch.abs(rgb_proj - depth_proj)
        discrepancy_energy = torch.tanh(discrepancy.mean(dim=1, keepdim=True))

        fused = self.refine(self.mixer(torch.cat([rgb_proj, depth_proj], dim=1)))

        rgb_gate = _constant_map_like(rgb_proj, 0.5)
        depth_gate = _constant_map_like(rgb_proj, 0.5)
        shared_gate = _constant_map_like(rgb_proj, 1e-6)
        gates = torch.cat([rgb_gate, depth_gate, shared_gate], dim=1)
        gates = gates / (gates.sum(dim=1, keepdim=True) + 1e-6)
        rgb_gate, depth_gate, shared_gate = torch.chunk(gates, 3, dim=1)
        gate_entropy = -(gates * torch.log(gates + 1e-6)).sum(dim=1, keepdim=True) / math.log(3.0)

        explain = {
            "rgb_gate": rgb_gate,
            "depth_gate": depth_gate,
            "shared_gate": shared_gate,
            "rgb_confidence": _constant_map_like(rgb_proj, 0.5),
            "depth_confidence": _constant_map_like(rgb_proj, 0.5),
            "cross_modal_consistency": 1.0 - discrepancy_energy,
            "boundary_uncertainty": discrepancy_energy,
            "gate_entropy": gate_entropy,
            "rgb_evidence": rgb_gate,
            "depth_evidence": depth_gate,
            "shared_evidence": shared_gate,
        }
        return fused, explain


class RGBOnlyFusionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.rgb_proj = ConvBNAct(in_channels, out_channels, kernel_size=1, padding=0)
        self.refine = ResidualRefineBlock(out_channels)

    def forward(
        self,
        rgb_feat: torch.Tensor,
        depth_feat: Optional[torch.Tensor] = None,
        depth_raw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        rgb_proj = self.rgb_proj(rgb_feat)
        fused = self.refine(rgb_proj)
        zero = _constant_map_like(rgb_proj, 0.0)
        one = _constant_map_like(rgb_proj, 1.0)
        explain = {
            "rgb_gate": one,
            "depth_gate": zero,
            "shared_gate": zero,
            "rgb_confidence": one,
            "depth_confidence": zero,
            "cross_modal_consistency": zero,
            "boundary_uncertainty": zero,
            "gate_entropy": zero,
            "rgb_evidence": one,
            "depth_evidence": zero,
            "shared_evidence": zero,
        }
        return fused, explain


class DecoderBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3),
            ConvBNAct(channels, channels, kernel_size=3),
            ResidualRefineBlock(channels),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + skip
        return self.block(x)


class DisagreementAwareRefinement(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.context_branch = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, kernel_size=3),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3),
        )
        self.local_branch = nn.Sequential(
            ConvBNAct(in_channels + 1, hidden_channels, kernel_size=3),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3),
        )
        self.residual_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, feature: torch.Tensor, disagreement_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        context_feature = self.context_branch(feature)
        local_feature = self.local_branch(torch.cat([feature, disagreement_map], dim=1))
        refined_feature = context_feature + disagreement_map * local_feature
        residual_logits = self.residual_head(refined_feature)
        return residual_logits, refined_feature


class RIGDNet(nn.Module):
    """
    Reliability-aware Interpretable Gated Dual-branch Network.

    RGB branch and depth branch share the same backbone family.
    Depth branch is optionally initialized from RGB ImageNet-1K weights by
    averaging the first convolution over channels, avoiding extra depth-only
    pretraining and keeping the control variable explicit.
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        decoder_channels: int = 96,
        pretrained_backbone: bool = True,
        rgb_weights_path: Optional[str] = None,
        depth_init_mode: str = "rgb_average",
        use_depth_branch: bool = True,
        fusion_mode: str = "evidence",
        use_rectifier: bool = True,
        use_disagreement_refinement: bool = True,
        use_edge_branch: bool = True,
    ):
        super().__init__()
        if backbone_name not in _BACKBONE_DIMS:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        if depth_init_mode not in {"rgb_average", "random"}:
            raise ValueError(f"Unsupported depth_init_mode: {depth_init_mode}")
        if fusion_mode not in {"evidence", "simple_gate", "concat"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        self.use_depth_branch = bool(use_depth_branch)
        self.fusion_mode = fusion_mode
        self.use_rectifier = bool(use_rectifier) and self.use_depth_branch
        self.use_disagreement_refinement = bool(use_disagreement_refinement)
        self.use_edge_branch = bool(use_edge_branch)

        pretrained_state = None
        if pretrained_backbone:
            pretrained_state = _load_resnet_reference_state(backbone_name, rgb_weights_path)

        depth_state = pretrained_state if self.use_depth_branch and depth_init_mode == "rgb_average" else None

        self.rgb_encoder = ResNetEncoder(
            backbone_name=backbone_name,
            in_channels=3,
            pretrained_state=pretrained_state,
        )
        self.depth_encoder = None
        if self.use_depth_branch:
            self.depth_encoder = ResNetEncoder(
                backbone_name=backbone_name,
                in_channels=1,
                pretrained_state=depth_state,
            )

        rectifier_channels = max(32, decoder_channels // 2)
        self.depth_rectifier = None
        if self.use_rectifier:
            self.depth_rectifier = TaskDrivenDepthRectifier(hidden_channels=rectifier_channels)

        feat_dims = _BACKBONE_DIMS[backbone_name]
        if not self.use_depth_branch:
            fusion_block_cls = RGBOnlyFusionBlock
        elif fusion_mode == "evidence":
            fusion_block_cls = InterpretableFusionBlock
        elif fusion_mode == "simple_gate":
            fusion_block_cls = SimpleGateFusionBlock
        else:
            fusion_block_cls = ConcatFusionBlock
        self.fusion_blocks = nn.ModuleList([fusion_block_cls(in_dim, decoder_channels) for in_dim in feat_dims])

        self.decoder4 = DecoderBlock(decoder_channels)
        self.decoder3 = DecoderBlock(decoder_channels)
        self.decoder2 = DecoderBlock(decoder_channels)
        self.decoder1 = DecoderBlock(decoder_channels)

        self.edge_encoder = None
        self.edge_head = None
        if self.use_edge_branch:
            self.edge_encoder = nn.Sequential(
                ConvBNAct(decoder_channels * 2, decoder_channels, kernel_size=3),
                ConvBNAct(decoder_channels, decoder_channels, kernel_size=3),
            )
            self.edge_head = nn.Conv2d(decoder_channels, 1, kernel_size=1)

        self.aux_heads = nn.ModuleList([nn.Conv2d(decoder_channels, 1, kernel_size=1) for _ in range(4)])
        self.base_head = nn.Sequential(
            ConvBNAct(decoder_channels * 5, decoder_channels, kernel_size=3),
            nn.Conv2d(decoder_channels, 1, kernel_size=1),
        )
        self.disagreement_aggregator = None
        self.disagreement_refiner = None
        if self.use_disagreement_refinement:
            self.disagreement_aggregator = nn.Sequential(
                ConvBNAct(len(feat_dims) + 1, rectifier_channels, kernel_size=3),
                ConvBNAct(rectifier_channels, rectifier_channels, kernel_size=3),
                nn.Conv2d(rectifier_channels, 1, kernel_size=1),
            )
            self.disagreement_refiner = DisagreementAwareRefinement(
                in_channels=decoder_channels * 5,
                hidden_channels=decoder_channels,
            )

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        depth_raw: Optional[torch.Tensor] = None,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        if depth_raw is None:
            depth_raw = (depth + 1.0) * 0.5

        rgb_feats = self.rgb_encoder(rgb)
        if self.use_depth_branch:
            if self.depth_rectifier is not None:
                rectified_depth, rectified_depth_raw, rectifier_explain = self.depth_rectifier(rgb, depth_raw)
            else:
                rectified_depth = depth
                rectified_depth_raw = depth_raw
                rectifier_explain = {
                    "rectify_gate": None,
                    "rectify_strength": None,
                    "rectify_residual": None,
                    "rectified_depth_edge": None,
                }
            depth_feats = self.depth_encoder(rectified_depth)
        else:
            rectified_depth_raw = torch.zeros_like(depth_raw)
            rectifier_explain = {
                "rectify_gate": None,
                "rectify_strength": None,
                "rectify_residual": None,
                "rectified_depth_edge": None,
            }
            depth_feats = [None for _ in rgb_feats]

        fused_feats: List[torch.Tensor] = []
        explain: Dict[str, List[torch.Tensor]] = {
            "rgb_gate": [],
            "depth_gate": [],
            "shared_gate": [],
            "rgb_confidence": [],
            "depth_confidence": [],
            "cross_modal_consistency": [],
            "boundary_uncertainty": [],
            "gate_entropy": [],
            "rgb_evidence": [],
            "depth_evidence": [],
            "shared_evidence": [],
            "stage_disagreement": [],
        }

        for rgb_feat, depth_feat, block in zip(rgb_feats, depth_feats, self.fusion_blocks):
            fused, block_explain = block(rgb_feat, depth_feat, rectified_depth_raw)
            stage_disagreement = torch.clamp(
                (
                    block_explain["boundary_uncertainty"]
                    + (1.0 - block_explain["cross_modal_consistency"])
                    + block_explain["gate_entropy"]
                )
                / 3.0,
                0.0,
                1.0,
            )
            block_explain["stage_disagreement"] = stage_disagreement
            fused_feats.append(fused)
            for key in explain:
                explain[key].append(block_explain[key])

        f1, f2, f3, f4 = fused_feats
        d4 = self.decoder4(f4)
        d3 = self.decoder3(d4, f3)
        d2 = self.decoder2(d3, f2)
        d1 = self.decoder1(d2, f1)

        if self.edge_encoder is not None and self.edge_head is not None:
            edge_feat = self.edge_encoder(torch.cat([d1, f1], dim=1))
        else:
            edge_feat = torch.zeros_like(d1)

        if output_size is None:
            output_size = rgb.shape[-2:]

        aux_feats = [d1, d2, d3, d4]
        aux_logits = [
            F.interpolate(head(feat), size=output_size, mode="bilinear", align_corners=False)
            for head, feat in zip(self.aux_heads, aux_feats)
        ]

        edge_logits = None
        if self.edge_head is not None:
            edge_logits = F.interpolate(self.edge_head(edge_feat), size=output_size, mode="bilinear", align_corners=False)

        final_feature = torch.cat(
            [
                d1,
                F.interpolate(d2, size=d1.shape[-2:], mode="bilinear", align_corners=False),
                F.interpolate(d3, size=d1.shape[-2:], mode="bilinear", align_corners=False),
                F.interpolate(d4, size=d1.shape[-2:], mode="bilinear", align_corners=False),
                edge_feat,
            ],
            dim=1,
        )
        base_logits = self.base_head(final_feature)
        disagreement_map = None
        refined_feature_energy = None
        logits = base_logits
        if self.disagreement_aggregator is not None and self.disagreement_refiner is not None:
            disagreement_inputs = [
                F.interpolate(stage_map, size=d1.shape[-2:], mode="bilinear", align_corners=False)
                for stage_map in explain["stage_disagreement"]
            ]
            rectify_strength = rectifier_explain.get("rectify_strength")
            if rectify_strength is None:
                rectify_strength = torch.zeros(
                    (rgb.shape[0], 1, d1.shape[2], d1.shape[3]),
                    device=d1.device,
                    dtype=d1.dtype,
                )
            else:
                rectify_strength = F.interpolate(rectify_strength, size=d1.shape[-2:], mode="bilinear", align_corners=False)
            disagreement_inputs.append(rectify_strength)
            disagreement_map = torch.sigmoid(self.disagreement_aggregator(torch.cat(disagreement_inputs, dim=1)))

            refine_residual, refined_feature = self.disagreement_refiner(final_feature, disagreement_map)
            logits = base_logits + disagreement_map * refine_residual
            refined_feature_energy = refined_feature.norm(dim=1, keepdim=True)

        base_logits = F.interpolate(base_logits, size=output_size, mode="bilinear", align_corners=False)
        logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

        explain.update(
            {
                "rectify_gate": rectifier_explain["rectify_gate"],
                "rectify_strength": rectifier_explain["rectify_strength"],
                "rectify_residual": rectifier_explain["rectify_residual"],
                "rectified_depth_edge": rectifier_explain["rectified_depth_edge"],
                "rectified_depth_raw": rectified_depth_raw,
                "disagreement_map": disagreement_map,
                "refined_feature_energy": refined_feature_energy,
            }
        )

        return {
            "logits": logits,
            "base_logits": base_logits,
            "edge_logits": edge_logits,
            "aux_logits": aux_logits,
            "rectified_depth_raw": rectified_depth_raw,
            "explain": explain,
        }
