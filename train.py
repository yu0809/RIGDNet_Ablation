import argparse
from contextlib import nullcontext
import os
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from dataset import RGBDEvalDataset, RGBDTrainDataset
from model import RIGDNet
import utils

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    p = os.path.expanduser(path)
    if not os.path.isabs(p):
        p = os.path.normpath(os.path.join(PROJECT_DIR, p))
    return p


def _build_optimizer(model: RIGDNet, train_cfg: Dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-4)),
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )


def _build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: Dict):
    scheduler_name = str(train_cfg.get("scheduler", "cosine")).lower()
    epoch_max = int(train_cfg.get("epoch_max", 10))
    min_lr = float(train_cfg.get("min_lr", 1e-6))
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_max, eta_min=min_lr)
    return None


def _normalize_gate_target(target: torch.Tensor) -> torch.Tensor:
    return target / (target.sum(dim=1, keepdim=True) + 1e-6)


def _compute_losses(outputs: Dict, mask: torch.Tensor, edge: Optional[torch.Tensor], train_cfg: Dict) -> Dict[str, torch.Tensor]:
    main_loss = utils.structure_loss(outputs["logits"], mask)
    base_logits = outputs.get("base_logits", None)
    if base_logits is not None:
        base_loss = utils.structure_loss(base_logits, mask)
    else:
        base_loss = mask.new_tensor(0.0)

    aux_logits = outputs.get("aux_logits", [])
    if aux_logits:
        aux_loss = torch.stack([utils.structure_loss(aux_logit, mask) for aux_logit in aux_logits]).mean()
    else:
        aux_loss = mask.new_tensor(0.0)

    edge_logits = outputs.get("edge_logits", None)
    if edge is not None and edge_logits is not None:
        edge_loss = utils.weighted_bce_loss(edge_logits, edge) + 0.5 * utils.dice_loss(edge_logits, edge)
    else:
        edge_loss = mask.new_tensor(0.0)

    gate_loss = mask.new_tensor(0.0)
    rectify_loss = mask.new_tensor(0.0)
    disagreement_loss = mask.new_tensor(0.0)
    uncertainty_loss = mask.new_tensor(0.0)
    entropy_loss = mask.new_tensor(0.0)
    explain = outputs.get("explain", {})
    rgb_gates = explain.get("rgb_gate", [])
    depth_gates = explain.get("depth_gate", [])
    shared_gates = explain.get("shared_gate", [])
    rgb_confidences = explain.get("rgb_confidence", [])
    depth_confidences = explain.get("depth_confidence", [])
    consistencies = explain.get("cross_modal_consistency", [])
    boundary_uncertainties = explain.get("boundary_uncertainty", [])
    gate_entropies = explain.get("gate_entropy", [])
    rectify_gate = explain.get("rectify_gate", None)
    rectify_strength = explain.get("rectify_strength", None)
    rectified_depth_edge = explain.get("rectified_depth_edge", None)
    disagreement_map = explain.get("disagreement_map", None)

    if (
        rgb_gates
        and depth_gates
        and shared_gates
        and rgb_confidences
        and depth_confidences
        and consistencies
        and boundary_uncertainties
    ):
        gate_loss = torch.stack(
            [
                F.l1_loss(
                    torch.cat([rgb_gate, depth_gate, shared_gate], dim=1),
                    _normalize_gate_target(
                        torch.cat(
                            [
                                rgb_conf.detach() * (1.0 - depth_conf.detach() + boundary.detach()),
                                depth_conf.detach() * consistency.detach(),
                                0.5
                                * (rgb_conf.detach() + depth_conf.detach())
                                * consistency.detach()
                                * (1.0 - boundary.detach()),
                            ],
                            dim=1,
                        )
                    ),
                )
                for rgb_gate, depth_gate, shared_gate, rgb_conf, depth_conf, consistency, boundary in zip(
                    rgb_gates,
                    depth_gates,
                    shared_gates,
                    rgb_confidences,
                    depth_confidences,
                    consistencies,
                    boundary_uncertainties,
                )
            ]
        ).mean()

    if edge is not None and boundary_uncertainties:
        edge_target = F.avg_pool2d(edge, kernel_size=5, stride=1, padding=2)
        uncertainty_loss = torch.stack(
            [
                F.binary_cross_entropy(
                    boundary.clamp(1e-6, 1.0 - 1e-6),
                    F.interpolate(edge_target, size=boundary.shape[-2:], mode="bilinear", align_corners=False),
                )
                for boundary in boundary_uncertainties
            ]
        ).mean()

    if edge is not None and gate_entropies:
        entropy_target = F.avg_pool2d(edge, kernel_size=9, stride=1, padding=4)
        entropy_loss = torch.stack(
            [
                F.l1_loss(
                    entropy,
                    F.interpolate(entropy_target, size=entropy.shape[-2:], mode="bilinear", align_corners=False),
                )
                for entropy in gate_entropies
            ]
        ).mean()

    if edge is not None and rectify_gate is not None and rectify_strength is not None:
        focus_target = F.avg_pool2d(edge, kernel_size=7, stride=1, padding=3)
        focus_target = F.interpolate(focus_target, size=rectify_gate.shape[-2:], mode="bilinear", align_corners=False)
        rectify_loss = F.binary_cross_entropy(rectify_gate.clamp(1e-6, 1.0 - 1e-6), focus_target)
        rectify_loss = rectify_loss + 0.5 * (rectify_strength * (1.0 - focus_target)).mean()
        if rectified_depth_edge is not None:
            normalized_rectified_edge = rectified_depth_edge / (
                rectified_depth_edge.amax(dim=(2, 3), keepdim=True) + 1e-6
            )
            rectify_loss = rectify_loss + 0.25 * F.l1_loss(normalized_rectified_edge, focus_target)

    if edge is not None and disagreement_map is not None:
        disagreement_target = F.avg_pool2d(edge, kernel_size=9, stride=1, padding=4)
        if rectify_strength is not None:
            disagreement_target = torch.maximum(
                disagreement_target,
                F.interpolate(rectify_strength.detach(), size=disagreement_target.shape[-2:], mode="bilinear", align_corners=False),
            )
        disagreement_loss = F.binary_cross_entropy(
            F.interpolate(disagreement_map, size=disagreement_target.shape[-2:], mode="bilinear", align_corners=False).clamp(
                1e-6, 1.0 - 1e-6
            ),
            disagreement_target,
        )

    total = (
        float(train_cfg.get("seg_loss_weight", 1.0)) * main_loss
        + float(train_cfg.get("base_loss_weight", 0.25)) * base_loss
        + float(train_cfg.get("aux_loss_weight", 0.3)) * aux_loss
        + float(train_cfg.get("edge_loss_weight", 0.2)) * edge_loss
        + float(train_cfg.get("gate_loss_weight", 0.05)) * gate_loss
        + float(train_cfg.get("rectify_loss_weight", 0.05)) * rectify_loss
        + float(train_cfg.get("disagreement_loss_weight", 0.05)) * disagreement_loss
        + float(train_cfg.get("uncertainty_loss_weight", 0.03)) * uncertainty_loss
        + float(train_cfg.get("entropy_loss_weight", 0.02)) * entropy_loss
    )

    return {
        "total": total,
        "main": main_loss.detach(),
        "base": base_loss.detach(),
        "aux": aux_loss.detach(),
        "edge": edge_loss.detach(),
        "gate": gate_loss.detach(),
        "rectify": rectify_loss.detach(),
        "disagreement": disagreement_loss.detach(),
        "uncertainty": uncertainty_loss.detach(),
        "entropy": entropy_loss.detach(),
    }


def train_one_epoch(model, loader, optimizer, scaler, device, train_cfg: Dict):
    model.train()
    total_meter = utils.AverageMeter()
    main_meter = utils.AverageMeter()
    base_meter = utils.AverageMeter()
    aux_meter = utils.AverageMeter()
    edge_meter = utils.AverageMeter()
    gate_meter = utils.AverageMeter()
    rectify_meter = utils.AverageMeter()
    disagreement_meter = utils.AverageMeter()
    uncertainty_meter = utils.AverageMeter()
    entropy_meter = utils.AverageMeter()

    use_amp = bool(train_cfg.get("use_amp", True)) and device.type == "cuda"
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    log_interval = int(train_cfg.get("log_interval", 10))

    pbar = tqdm(enumerate(loader, 1), total=len(loader), desc="train", leave=False)
    for step, batch in pbar:
        image = batch["image"].to(device)
        depth = batch["depth"].to(device)
        depth_raw = batch["depth_raw"].to(device)
        mask = batch["mask"].to(device)
        edge = batch.get("edge", None)
        edge = edge.to(device) if edge is not None else None

        optimizer.zero_grad(set_to_none=True)
        amp_ctx = torch.amp.autocast(device_type="cuda", enabled=True) if use_amp else nullcontext()
        with amp_ctx:
            outputs = model(image, depth, depth_raw=depth_raw, output_size=tuple(mask.shape[-2:]))
            losses = _compute_losses(outputs, mask, edge, train_cfg)

        scaler.scale(losses["total"]).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = image.size(0)
        total_meter.update(losses["total"].item(), bs)
        main_meter.update(losses["main"].item(), bs)
        base_meter.update(losses["base"].item(), bs)
        aux_meter.update(losses["aux"].item(), bs)
        edge_meter.update(losses["edge"].item(), bs)
        gate_meter.update(losses["gate"].item(), bs)
        rectify_meter.update(losses["rectify"].item(), bs)
        disagreement_meter.update(losses["disagreement"].item(), bs)
        uncertainty_meter.update(losses["uncertainty"].item(), bs)
        entropy_meter.update(losses["entropy"].item(), bs)

        if step % max(1, log_interval) == 0 or step == len(loader):
            pbar.set_postfix(
                loss=f"{total_meter.avg:.4f}",
                seg=f"{main_meter.avg:.4f}",
                base=f"{base_meter.avg:.4f}",
                edge=f"{edge_meter.avg:.4f}",
                gate=f"{gate_meter.avg:.4f}",
            )

    return {
        "loss": total_meter.avg,
        "seg": main_meter.avg,
        "base": base_meter.avg,
        "aux": aux_meter.avg,
        "edge": edge_meter.avg,
        "gate": gate_meter.avg,
        "rectify": rectify_meter.avg,
        "disagreement": disagreement_meter.avg,
        "uncertainty": uncertainty_meter.avg,
        "entropy": entropy_meter.avg,
    }


@torch.no_grad()
def evaluate(model, loader, device, train_cfg: Dict):
    model.eval()
    loss_meter = utils.AverageMeter()
    metric_meter = {"iou": 0.0, "dice": 0.0, "mae": 0.0, "n": 0}

    for batch in tqdm(loader, desc="val", leave=False):
        image = batch["image"].to(device)
        depth = batch["depth"].to(device)
        depth_raw = batch["depth_raw"].to(device)
        mask = batch["mask"].to(device)

        outputs = model(image, depth, depth_raw=depth_raw, output_size=tuple(mask.shape[-2:]))
        main_loss = utils.structure_loss(outputs["logits"], mask)
        loss_meter.update(main_loss.item(), image.size(0))

        metrics = utils.compute_metrics(outputs["logits"], mask)
        bs = image.size(0)
        metric_meter["iou"] += metrics["iou"] * bs
        metric_meter["dice"] += metrics["dice"] * bs
        metric_meter["mae"] += metrics["mae"] * bs
        metric_meter["n"] += bs

    n = max(1, metric_meter["n"])
    return {
        "loss": loss_meter.avg,
        "iou": metric_meter["iou"] / n,
        "dice": metric_meter["dice"] / n,
        "mae": metric_meter["mae"] / n,
    }


def main():
    parser = argparse.ArgumentParser(description="RIGDNet Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--save_dir", type=str, default=None, help="覆盖 checkpoint 保存目录")
    parser.add_argument("--resume", type=str, default=None, help="恢复训练 checkpoint")
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    train_data_cfg = cfg.get("train_dataset", {})
    val_data_cfg = cfg.get("val_dataset", {})

    utils.set_seed(int(train_cfg.get("seed", 42)))
    device = utils.prepare_device(prefer_mps=bool(train_cfg.get("allow_mps", False)))
    print(f"使用设备: {device}")

    save_dir = _resolve_path(args.save_dir or train_cfg.get("save_dir", "./checkpoints"))
    os.makedirs(save_dir, exist_ok=True)
    print(f"保存目录: {save_dir}")

    train_ds = RGBDTrainDataset(
        image_dir=_resolve_path(train_data_cfg["img_dir"]),
        depth_dir=_resolve_path(train_data_cfg["depth_dir"]),
        mask_dir=_resolve_path(train_data_cfg["mask_dir"]),
        edge_dir=_resolve_path(train_data_cfg.get("edge_dir")),
        image_size=int(train_data_cfg.get("image_size", 384)),
        augment=bool(train_data_cfg.get("augment", True)),
        depth_drop_prob=float(train_data_cfg.get("depth_drop_prob", 0.0)),
        depth_noise_std=float(train_data_cfg.get("depth_noise_std", 0.0)),
        depth_blur_prob=float(train_data_cfg.get("depth_blur_prob", 0.0)),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    val_loader = None
    if bool(val_data_cfg.get("enabled", False)):
        val_ds = RGBDEvalDataset(
            image_dir=_resolve_path(val_data_cfg["img_dir"]),
            depth_dir=_resolve_path(val_data_cfg["depth_dir"]),
            mask_dir=_resolve_path(val_data_cfg["mask_dir"]),
            image_size=int(val_data_cfg.get("image_size", train_data_cfg.get("image_size", 384))),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=max(0, int(train_cfg.get("num_workers", 0)) // 2),
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    model = RIGDNet(
        backbone_name=str(model_cfg.get("backbone_name", "resnet50")),
        decoder_channels=int(model_cfg.get("decoder_channels", 96)),
        pretrained_backbone=bool(model_cfg.get("pretrained_backbone", True)),
        rgb_weights_path=_resolve_path(model_cfg.get("rgb_weights_path")),
        depth_init_mode=str(model_cfg.get("depth_init_mode", "rgb_average")),
        use_depth_branch=bool(model_cfg.get("use_depth_branch", True)),
        fusion_mode=str(model_cfg.get("fusion_mode", "evidence")),
        use_rectifier=bool(model_cfg.get("use_rectifier", True)),
        use_disagreement_refinement=bool(model_cfg.get("use_disagreement_refinement", True)),
        use_edge_branch=bool(model_cfg.get("use_edge_branch", True)),
    ).to(device)

    optimizer = _build_optimizer(model, train_cfg)
    scheduler = _build_scheduler(optimizer, train_cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("use_amp", True)) and device.type == "cuda")

    start_epoch = 1
    best_iou = -1.0
    resume_path = args.resume or train_cfg.get("resume")
    if resume_path:
        resume_path = _resolve_path(resume_path)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"恢复权重不存在: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict, strict=False)
        if isinstance(ckpt, dict) and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None and "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_iou = float(ckpt.get("best_iou", -1.0))
        print(f"恢复训练: {resume_path}, start_epoch={start_epoch}")

    epoch_max = int(train_cfg.get("epoch_max", 10))
    save_every = int(train_cfg.get("save_every", 0))

    latest_ckpt = os.path.join(save_dir, "checkpoint_latest.pth")
    best_ckpt = os.path.join(save_dir, "checkpoint_best.pth")

    for epoch in range(start_epoch, epoch_max + 1):
        tic = time.time()

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            train_cfg=train_cfg,
        )

        msg = (
            f"[Epoch {epoch}/{epoch_max}] "
            f"loss={train_stats['loss']:.4f} seg={train_stats['seg']:.4f} base={train_stats['base']:.4f} "
            f"aux={train_stats['aux']:.4f} edge={train_stats['edge']:.4f} "
            f"gate={train_stats['gate']:.4f} rect={train_stats['rectify']:.4f} "
            f"dis={train_stats['disagreement']:.4f} unc={train_stats['uncertainty']:.4f} "
            f"ent={train_stats['entropy']:.4f}"
        )

        val_stats = None
        if val_loader is not None:
            val_stats = evaluate(model=model, loader=val_loader, device=device, train_cfg=train_cfg)
            msg += (
                f" val_loss={val_stats['loss']:.4f}"
                f" iou={val_stats['iou']:.4f}"
                f" dice={val_stats['dice']:.4f}"
                f" mae={val_stats['mae']:.4f}"
            )

            if val_stats["iou"] > best_iou:
                best_iou = val_stats["iou"]
                best_payload = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "best_iou": best_iou,
                    "metric_name": "iou",
                    "config": cfg,
                }
                utils.save_checkpoint(best_payload, best_ckpt)

        latest_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_iou": best_iou,
            "config": cfg,
        }
        utils.save_checkpoint(latest_payload, latest_ckpt)

        if save_every > 0 and epoch % save_every == 0:
            utils.save_checkpoint(latest_payload, os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"))

        if scheduler is not None:
            scheduler.step()

        msg += f" time={utils.time_text(time.time() - tic)}"
        print(msg)

    print("训练结束。")


if __name__ == "__main__":
    main()
