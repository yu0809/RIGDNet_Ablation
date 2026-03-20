import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List

import yaml

from experiments import EXPERIMENTS, GROUPS


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_ENTRY = PROJECT_DIR / "train.py"
TEST_ENTRY = PROJECT_DIR / "test.py"
GENERATED_CONFIG_DIR = PROJECT_DIR / "generated_configs"


def _deep_update(base: Dict, override: Dict) -> Dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_yaml(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _select_experiments(group: str, explicit: Iterable[str]) -> List[str]:
    if explicit:
        names = list(explicit)
    else:
        if group not in GROUPS:
            raise ValueError(f"未知实验组: {group}")
        names = list(GROUPS[group])
    unknown = [name for name in names if name not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"未知实验名: {unknown}")
    return names


def _resolve_output_root(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


def _build_run_config(base_cfg: Dict, exp_name: str, seed: int, output_root: Path) -> Dict:
    cfg = copy.deepcopy(base_cfg)
    _deep_update(cfg, EXPERIMENTS[exp_name]["overrides"])

    run_root = output_root / exp_name / f"seed_{seed}"
    checkpoint_dir = run_root / "checkpoints"
    result_dir = run_root / "results"

    cfg.setdefault("training", {})
    cfg.setdefault("testing", {})
    cfg["training"]["seed"] = int(seed)
    cfg["training"]["save_dir"] = str(checkpoint_dir)
    cfg["testing"]["model_path"] = str(checkpoint_dir / "checkpoint_best.pth")
    cfg["testing"]["output_dir"] = str(result_dir)
    return cfg


def _run_command(cmd: List[str]) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=PROJECT_DIR, check=True)


def _pick_checkpoint(checkpoint_dir: Path) -> Path:
    best_path = checkpoint_dir / "checkpoint_best.pth"
    latest_path = checkpoint_dir / "checkpoint_latest.pth"
    if best_path.is_file():
        return best_path
    if latest_path.is_file():
        return latest_path
    raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_dir}")


def _write_summary(per_run_rows: List[Dict], summary_rows: List[Dict], output_root: Path) -> None:
    per_run_csv = output_root / "per_run_metrics.csv"
    summary_csv = output_root / "summary_metrics.csv"
    summary_json = output_root / "summary_metrics.json"

    if per_run_rows:
        with open(per_run_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_run_rows)

    if summary_rows:
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, ensure_ascii=False, indent=2)


def _paper_fields(cfg: Dict, metrics: Dict) -> Dict:
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    return {
        "best_epoch": metrics.get("checkpoint_epoch"),
        "best_val_iou": metrics.get("best_iou"),
        "metric_name": metrics.get("metric_name", "iou"),
        "epoch_max": train_cfg.get("epoch_max"),
        "lr": train_cfg.get("lr"),
        "weight_decay": train_cfg.get("weight_decay"),
        "seg_loss_weight": train_cfg.get("seg_loss_weight"),
        "base_loss_weight": train_cfg.get("base_loss_weight"),
        "aux_loss_weight": train_cfg.get("aux_loss_weight"),
        "edge_loss_weight": train_cfg.get("edge_loss_weight"),
        "gate_loss_weight": train_cfg.get("gate_loss_weight"),
        "rectify_loss_weight": train_cfg.get("rectify_loss_weight"),
        "disagreement_loss_weight": train_cfg.get("disagreement_loss_weight"),
        "uncertainty_loss_weight": train_cfg.get("uncertainty_loss_weight"),
        "entropy_loss_weight": train_cfg.get("entropy_loss_weight"),
        "fusion_mode": model_cfg.get("fusion_mode"),
        "use_depth_branch": model_cfg.get("use_depth_branch"),
        "use_rectifier": model_cfg.get("use_rectifier"),
        "use_disagreement_refinement": model_cfg.get("use_disagreement_refinement"),
        "use_edge_branch": model_cfg.get("use_edge_branch"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RIGDNet Ablation Runner")
    parser.add_argument("--base_config", type=str, default="configs/base.yaml", help="基础配置文件")
    parser.add_argument("--group", type=str, default="core", choices=sorted(GROUPS.keys()), help="实验组")
    parser.add_argument("--experiments", nargs="*", default=None, help="显式指定实验名，优先级高于 --group")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="随机种子列表")
    parser.add_argument("--output_root", type=str, default="runs", help="所有实验输出目录")
    parser.add_argument("--save_explain", action="store_true", help="测试时保存解释性图")
    parser.add_argument("--skip_existing", action="store_true", help="若 metrics 已存在则跳过该实验")
    parser.add_argument("--train_only", action="store_true", help="只训练，不测试")
    parser.add_argument("--test_only", action="store_true", help="只测试，要求 checkpoint 已存在")
    args = parser.parse_args()

    if args.train_only and args.test_only:
        raise ValueError("--train_only 与 --test_only 不能同时使用")

    base_config_path = Path(args.base_config)
    if not base_config_path.is_absolute():
        base_config_path = PROJECT_DIR / base_config_path
    base_cfg = _load_yaml(base_config_path.resolve())

    exp_names = _select_experiments(args.group, args.experiments)
    output_root = _resolve_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    per_run_rows: List[Dict] = []
    metrics_by_experiment: Dict[str, List[Dict]] = {name: [] for name in exp_names}

    for exp_name in exp_names:
        print(f"\n========== {exp_name} ==========")
        print(EXPERIMENTS[exp_name]["description"])
        for seed in args.seeds:
            cfg = _build_run_config(base_cfg, exp_name, seed, output_root)
            cfg_path = GENERATED_CONFIG_DIR / f"{exp_name}_seed{seed}.yaml"
            _save_yaml(cfg_path, cfg)

            run_root = output_root / exp_name / f"seed_{seed}"
            checkpoint_dir = run_root / "checkpoints"
            metrics_path = run_root / "metrics.json"

            if args.skip_existing and metrics_path.is_file():
                print(f"[SKIP] {exp_name} seed={seed} 已存在 metrics: {metrics_path}")
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            else:
                if not args.test_only:
                    _run_command([sys.executable, str(TRAIN_ENTRY), "--config", str(cfg_path)])

                if args.train_only:
                    continue

                checkpoint_path = _pick_checkpoint(checkpoint_dir)
                test_cmd = [
                    sys.executable,
                    str(TEST_ENTRY),
                    "--config",
                    str(cfg_path),
                    "--model",
                    str(checkpoint_path),
                    "--metrics_out",
                    str(metrics_path),
                ]
                if args.save_explain:
                    test_cmd.append("--save_explain")
                _run_command(test_cmd)
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)

            row = {
                "experiment": exp_name,
                "seed": seed,
                "iou": metrics["iou"],
                "dice": metrics["dice"],
                "mae": metrics["mae"],
                "checkpoint": metrics.get("model_path", ""),
            }
            row.update(_paper_fields(cfg, metrics))
            per_run_rows.append(row)
            metrics_by_experiment[exp_name].append(row)

    summary_rows: List[Dict] = []
    if not args.train_only:
        for exp_name in exp_names:
            rows = metrics_by_experiment[exp_name]
            if not rows:
                continue
            ious = [row["iou"] for row in rows]
            dices = [row["dice"] for row in rows]
            maes = [row["mae"] for row in rows]
            summary_rows.append(
                {
                    "experiment": exp_name,
                    "runs": len(rows),
                    "iou_mean": mean(ious),
                    "iou_std": stdev(ious) if len(ious) > 1 else 0.0,
                    "dice_mean": mean(dices),
                    "dice_std": stdev(dices) if len(dices) > 1 else 0.0,
                    "mae_mean": mean(maes),
                    "mae_std": stdev(maes) if len(maes) > 1 else 0.0,
                    "best_epoch_mean": mean([float(row["best_epoch"]) for row in rows if row.get("best_epoch") is not None])
                    if any(row.get("best_epoch") is not None for row in rows)
                    else None,
                    "best_val_iou_mean": mean([float(row["best_val_iou"]) for row in rows if row.get("best_val_iou") is not None])
                    if any(row.get("best_val_iou") is not None for row in rows)
                    else None,
                }
            )

    _write_summary(per_run_rows, summary_rows, output_root)
    print(f"\n输出目录: {output_root}")
    if not args.train_only:
        print(f"汇总文件: {output_root / 'summary_metrics.csv'}")


if __name__ == "__main__":
    main()
