# 复现指南 / Reproduction Guide

本项目已完成训练和评估。你**不需要重新训练**，只需要下载数据即可复现推理和评估。

## 快速开始

```bash
# 1. 克隆项目 (已包含 best_model.pth 和所有结果)
git clone git@github.com:qianxu05172019/brats-biohub.git
cd brats-biohub

# 2. 安装依赖
pip install -r requirements.txt
# 或手动安装: pip install monai torch nibabel numpy pandas matplotlib seaborn pyyaml wandb tqdm

# 3. 下载数据 (二选一, 见下方详细说明)
bash scripts/setup_data.sh kaggle   # 方式1: Kaggle (~13GB)
bash scripts/setup_data.sh monai    # 方式2: MONAI (~4.5GB, 需转换)

# 4. 直接推理 (不需要重新训练!)
python train.py --config configs/infer_sliding_window.yaml

# 5. 评估
python -m src.eval --pred_dir outputs/predictions --gt_dir data/BraTS2021_Training_Data
```

---

## 数据下载详细说明

### 方式 1: Kaggle (推荐)

数据集: [BRaTS 2021 Task 1 Dataset](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)

```bash
# 安装 Kaggle CLI
pip install kaggle

# 配置 API (去 https://www.kaggle.com/settings -> API -> Create New Token)
mkdir -p ~/.kaggle
# 把下载的 kaggle.json 放到 ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 下载 (~13GB)
kaggle datasets download -d dschettler8845/brats-2021-task1 -p data/
unzip data/brats-2021-task1.zip -d data/BraTS2021_Training_Data
rm data/brats-2021-task1.zip
```

### 方式 2: MONAI DecathlonDataset

这会下载 Medical Segmentation Decathlon (MSD) 的 Task01_BrainTumour，然后转换为 BraTS 格式。

```bash
# MSD 数据只有 484 例 (BraTS 2021 有 731 例)，但足够复现流程
python3 -c "
from monai.apps import DecathlonDataset
from pathlib import Path
Path('data').mkdir(exist_ok=True)
ds = DecathlonDataset(root_dir='data', task='Task01_BrainTumour',
                      section='training', download=True, cache_rate=0.0)
print(f'Downloaded {len(ds)} cases')
"

# 转换为 BraTS 格式
python scripts/convert_msd_to_brats.py all
```

### 方式 3: 手动下载

1. 去 [Synapse 平台](https://www.synapse.org/#!Synapse:syn25829067) 注册并申请访问
2. 下载 `BraTS2021_Training_Data.tar` (~23GB)
3. 解压到 `data/BraTS2021_Training_Data/`

---

## 数据格式说明

下载后的目录结构应该是:

```
data/BraTS2021_Training_Data/
├── BraTS2021_00000/
│   ├── BraTS2021_00000_flair.nii.gz   # FLAIR 模态
│   ├── BraTS2021_00000_t1.nii.gz      # T1 模态
│   ├── BraTS2021_00000_t1ce.nii.gz    # T1 对比增强
│   ├── BraTS2021_00000_t2.nii.gz      # T2 模态
│   └── BraTS2021_00000_seg.nii.gz     # 分割标签 (0/1/2/4)
├── BraTS2021_00002/
│   └── ...
└── ... (共 731 例)
```

每个病例 5 个 NIfTI 文件，约 30MB/例，总计约 23GB。

---

## 路径配置

下载数据后，确认 `configs/train_3d_unet.yaml` 中的路径正确:

```yaml
data:
  root_dir: data/BraTS2021_Training_Data  # 相对路径或绝对路径
```

推理配置 `configs/infer_sliding_window.yaml` 也需要确认路径。

---

## 已包含的文件 (不需要重新生成)

| 文件 | 说明 |
|------|------|
| `checkpoints/best_model.pth` | 最佳模型权重 (epoch 277, Dice=0.9215) |
| `checkpoints/latest.pth` | 最后一个 epoch 的完整状态 |
| `reports/results.csv` | 147 例验证集的完整评估结果 |
| `reports/training_history.json` | 287 个 epoch 的训练日志 |
| `reports/figures/` | 4 张预生成的分析图 |
| `outputs/predictions/` | 验证集预测的 NIfTI 文件 |
| `notebooks/01_project_presentation.ipynb` | 项目展示 notebook (本地可运行) |

---

## 如果你想重新训练

```bash
# 完整训练 (~21 小时, 需要 GPU >= 16GB VRAM)
python train.py --config configs/train_3d_unet.yaml

# 断点续训 (从 latest.pth 继续)
python train.py --config configs/train_3d_unet.yaml --resume
```

---

## 环境要求

- Python 3.8+
- PyTorch 1.12+ (with CUDA for training/inference)
- MONAI 1.0+
- GPU: >= 16GB VRAM (训练), >= 8GB VRAM (推理)
- 磁盘: ~25GB (数据) + ~500MB (项目文件)

原始训练环境: RunPod, NVIDIA A40 (48GB), CUDA 12.x, Python 3.11
