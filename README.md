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

## 通过 Kaggle CLI 上传为 Dataset（与官方 API 文档一致）

官方说明入口：[Kaggle Public API](https://www.kaggle.com/docs/api)（CLI 为 [kaggle-cli](https://github.com/Kaggle/kaggle-cli)）；数据集元数据格式见仓库内 [Dataset Metadata](https://github.com/Kaggle/kaggle-api/wiki/Dataset-Metadata)。

### 1. 安装并登录

```bash
pip install kaggle
```

在 Kaggle 网页 **Account → API → Create New API Token** 下载 `kaggle.json`，放到本机：

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 2. 准备「只含代码」的上传文件夹（推荐）

训练数据体积大，建议 **代码与数据分成两个 Dataset**。代码包可用本目录脚本生成干净副本：

```bash
cd /path/to/RIGDNet_Ablation
bash pack_for_kaggle_dataset.sh
# 默认输出到与 RIGDNet_Ablation 同级的 RIGDNet_Ablation_kaggle_upload/
```

### 3. 生成并编辑 `dataset-metadata.json`

在上传目录内执行（路径与 [官方文档](https://github.com/Kaggle/kaggle-api/wiki/Dataset-Metadata) 一致）：

```bash
cd /path/to/RIGDNet_Ablation_kaggle_upload
kaggle datasets init -p .
```

用编辑器打开生成的 `dataset-metadata.json`，至少改好：

- **`id`**：`你的Kaggle用户名/数据集短名`（全站唯一）
- **`title`**：6～50 字符
- **`subtitle`**：20～80 字符
- **`licenses`**：与数据/代码许可一致（如 `apache-2.0` 等，见文档列表）

### 4. 创建新数据集 / 发新版本

```bash
# 首次创建
kaggle datasets create -p .

# 之后同一路径更新版本
kaggle datasets version -p . -m "更新说明"
```

上传完成后，在 Kaggle Notebook 里 **Add Data** 挂载该 Dataset；Notebook 中再 `git clone` 你的 GitHub 或直接使用 Dataset 内代码均可（二选一即可，避免重复维护）。

### 5. 训练数据单独上传

将 `train/`、`validation/`、`test/`（或你实际使用的目录）打成 **另一个 Dataset**，同样在文件夹里放 `dataset-metadata.json` 后执行 `kaggle datasets create`。Notebook 里通过 `/kaggle/input/<dataset-name>/` 访问，并与本仓库 `configs/base.yaml` 中的相对路径对齐（见先前 Kaggle 目录布局说明）。
