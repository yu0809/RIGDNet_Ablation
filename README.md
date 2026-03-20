# RIGDNet_Ablation

这套目录是给论文实验单独准备的副本，原始 `RIGDNet` 目录不会被修改。

## 目录作用

- `model.py / train.py / test.py`：独立版训练与测试代码，支持消融开关。
- `configs/base.yaml`：正式实验基础配置。
- `configs/smoke.yaml`：快速冒烟配置。
- `experiments.py`：实验清单与覆盖项。
- `run_ablations.py`：批量生成配置、训练、测试、汇总结果。
- `run_all.sh`：服务器上一键运行入口。

## 已内置的实验

- `full`
- `rgb_only`
- `naive_fusion`
- `simple_gate`
- `wo_rectification`
- `wo_evidence`
- `wo_refinement`
- `wo_edge`
- `wo_depth_aug`
- `depth_init_random`
- `wo_gate_supervision`
- `wo_rectify_loss`
- `wo_disagreement_loss`

默认实验组：

- `quick`：最小排错集
- `core`：论文主消融推荐集
- `all`：包含结构消融和损失消融

## 一键运行

先进入你服务器上的 Python 环境，再执行：

```bash
cd /path/to/human_detection/RIGDNet_Ablation
bash run_all.sh --group core --seeds 42 43 44
```

如果你只想先检查流程：

```bash
cd /path/to/human_detection/RIGDNet_Ablation
bash run_all.sh --base_config configs/smoke.yaml --group quick --seeds 42
```

## 输出内容

运行结束后会在 `runs/` 下生成：

- 每个实验各自的 `checkpoints/`
- 每个实验各自的 `results/`
- 每次运行的 `metrics.json`
- 总表 `summary_metrics.csv`
- 明细表 `per_run_metrics.csv`

## 常用参数

- `--group core`
- `--experiments full wo_rectification wo_evidence`
- `--seeds 42 43 44`
- `--output_root /your/output/path`
- `--save_explain`
- `--skip_existing`
- `--train_only`
- `--test_only`
