import argparse
import json
import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from dataset import RGBDEvalDataset
from model import RIGDNet
import utils

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(path: str) -> str:
    p = os.path.expanduser(path)
    if not os.path.isabs(p):
        p = os.path.normpath(os.path.join(PROJECT_DIR, p))
    return p


def _save_prediction(logits: torch.Tensor, path: str):
    prob = torch.sigmoid(logits[0, 0]).detach().cpu().numpy()
    prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
    Image.fromarray((prob * 255).astype(np.uint8), mode="L").save(path)


@torch.no_grad()
def run_test(model, loader, device, save_pred: bool = False, save_explain: bool = False, output_dir: str = "./results"):
    model.eval()
    metric_meter = {"iou": 0.0, "dice": 0.0, "mae": 0.0, "n": 0}

    if save_pred or save_explain:
        os.makedirs(output_dir, exist_ok=True)

    for batch in tqdm(loader, desc="test"):
        image = batch["image"].to(device)
        depth = batch["depth"].to(device)
        depth_raw = batch["depth_raw"].to(device)
        mask = batch["mask"].to(device)
        name = batch["name"][0]

        outputs = model(image, depth, depth_raw=depth_raw, output_size=tuple(mask.shape[-2:]))
        logits = outputs["logits"]

        metrics = utils.compute_metrics(logits, mask)
        bs = image.size(0)
        metric_meter["iou"] += metrics["iou"] * bs
        metric_meter["dice"] += metrics["dice"] * bs
        metric_meter["mae"] += metrics["mae"] * bs
        metric_meter["n"] += bs

        if save_pred:
            _save_prediction(logits, os.path.join(output_dir, f"{name}_pred.png"))
            base_logits = outputs.get("base_logits", None)
            if base_logits is not None:
                _save_prediction(base_logits, os.path.join(output_dir, f"{name}_base_pred.png"))

        if save_explain:
            edge_logits = outputs.get("edge_logits", None)
            if edge_logits is not None:
                utils.save_gray_map(edge_logits[0], os.path.join(output_dir, f"{name}_edge.png"))
            explain = outputs["explain"]
            explain_keys = [
                ("rgb_gate", "rgb_gate"),
                ("depth_gate", "depth_gate"),
                ("shared_gate", "shared_gate"),
                ("rgb_confidence", "rgb_conf"),
                ("depth_confidence", "depth_conf"),
                ("cross_modal_consistency", "consistency"),
                ("boundary_uncertainty", "boundary_unc"),
                ("gate_entropy", "gate_entropy"),
                ("stage_disagreement", "stage_disagreement"),
            ]
            for explain_key, filename_key in explain_keys:
                for stage_idx, explain_map in enumerate(explain.get(explain_key, []), start=1):
                    explain_map = torch.nn.functional.interpolate(
                        explain_map,
                        size=mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    utils.save_gray_map(
                        explain_map[0],
                        os.path.join(output_dir, f"{name}_{filename_key}_s{stage_idx}.png"),
                    )

            single_map_keys = [
                ("rectify_gate", "rectify_gate"),
                ("rectify_strength", "rectify_strength"),
                ("rectify_residual", "rectify_residual"),
                ("rectified_depth_edge", "rectified_depth_edge"),
                ("disagreement_map", "disagreement_map"),
                ("refined_feature_energy", "refined_energy"),
                ("rectified_depth_raw", "rectified_depth"),
            ]
            for explain_key, filename_key in single_map_keys:
                explain_map = explain.get(explain_key, None)
                if explain_map is None:
                    continue
                explain_map = torch.nn.functional.interpolate(
                    explain_map,
                    size=mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                utils.save_gray_map(
                    explain_map[0],
                    os.path.join(output_dir, f"{name}_{filename_key}.png"),
                )

    n = max(1, metric_meter["n"])
    return {
        "iou": metric_meter["iou"] / n,
        "dice": metric_meter["dice"] / n,
        "mae": metric_meter["mae"] / n,
    }


def main():
    parser = argparse.ArgumentParser(description="RIGDNet Testing")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--model", type=str, default=None, help="模型权重路径")
    parser.add_argument("--output_dir", type=str, default=None, help="预测图保存目录")
    parser.add_argument("--metrics_out", type=str, default=None, help="评测指标 JSON 输出路径")
    parser.add_argument("--save_pred", action="store_true", help="保存预测图")
    parser.add_argument("--save_explain", action="store_true", help="保存解释性图")
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    test_cfg = cfg.get("test_dataset", {})
    train_cfg = cfg.get("training", {})
    testing_cfg = cfg.get("testing", {})
    model_cfg = cfg.get("model", {})

    model_path = args.model or testing_cfg.get("model_path")
    if not model_path:
        raise ValueError("请通过 --model 或 config.testing.model_path 提供模型权重路径")
    model_path = _resolve_path(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"模型权重不存在: {model_path}")

    strict_load = bool(testing_cfg.get("strict_load", True))
    output_dir = _resolve_path(args.output_dir or testing_cfg.get("output_dir", "./results"))

    device = utils.prepare_device(prefer_mps=bool(train_cfg.get("allow_mps", False)))
    print(f"使用设备: {device}")

    test_ds = RGBDEvalDataset(
        image_dir=_resolve_path(test_cfg["img_dir"]),
        depth_dir=_resolve_path(test_cfg["depth_dir"]),
        mask_dir=_resolve_path(test_cfg["mask_dir"]),
        image_size=int(test_cfg.get("image_size", 384)),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = RIGDNet(
        backbone_name=str(model_cfg.get("backbone_name", "resnet50")),
        decoder_channels=int(model_cfg.get("decoder_channels", 96)),
        pretrained_backbone=False,
        rgb_weights_path=None,
        depth_init_mode=str(model_cfg.get("depth_init_mode", "rgb_average")),
        use_depth_branch=bool(model_cfg.get("use_depth_branch", True)),
        fusion_mode=str(model_cfg.get("fusion_mode", "evidence")),
        use_rectifier=bool(model_cfg.get("use_rectifier", True)),
        use_disagreement_refinement=bool(model_cfg.get("use_disagreement_refinement", True)),
        use_edge_branch=bool(model_cfg.get("use_edge_branch", True)),
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    if strict_load:
        model.load_state_dict(state_dict, strict=True)
    else:
        incompat = model.load_state_dict(state_dict, strict=False)
        if incompat.missing_keys or incompat.unexpected_keys:
            print(
                f"[WARN] strict_load=False, missing={len(incompat.missing_keys)}, "
                f"unexpected={len(incompat.unexpected_keys)}"
            )

    print(f"已加载模型: {model_path}")
    checkpoint_meta = {}
    if isinstance(ckpt, dict):
        checkpoint_meta = {
            "checkpoint_epoch": ckpt.get("epoch"),
            "best_iou": ckpt.get("best_iou"),
            "metric_name": ckpt.get("metric_name"),
        }

    metrics = run_test(
        model=model,
        loader=test_loader,
        device=device,
        save_pred=args.save_pred,
        save_explain=args.save_explain or bool(testing_cfg.get("save_explain", False)),
        output_dir=output_dir,
    )

    print("\n========== 测试结果 ==========")
    print(f"IoU:  {metrics['iou']:.4f}")
    print(f"Dice: {metrics['dice']:.4f}")
    print(f"MAE:  {metrics['mae']:.4f}")
    if args.metrics_out:
        metrics_out = _resolve_path(args.metrics_out)
        metrics_dir = os.path.dirname(metrics_out)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(metrics_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "iou": metrics["iou"],
                    "dice": metrics["dice"],
                    "mae": metrics["mae"],
                    "model_path": model_path,
                    "output_dir": output_dir,
                    **checkpoint_meta,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    if args.save_pred or args.save_explain or bool(testing_cfg.get("save_explain", False)):
        print(f"结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
