#!/bin/bash
# ============================================================
# BraTS 2021 数据下载与转换脚本
# Download and convert BraTS 2021 brain tumor segmentation data
# ============================================================
#
# 两种下载方式 (二选一):
#   方式 1: Kaggle (推荐, 直接就是 BraTS 格式)
#   方式 2: MONAI DecathlonDataset (MSD 格式, 需要转换)
#
# 用法:
#   bash scripts/setup_data.sh kaggle    # 方式1
#   bash scripts/setup_data.sh monai     # 方式2
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_ROOT}/data/BraTS2021_Training_Data"

echo "============================================================"
echo "  BraTS 2021 Data Setup"
echo "  Target: ${DATA_DIR}"
echo "============================================================"

METHOD="${1:-kaggle}"

# ------ 方式 1: Kaggle ------
if [ "$METHOD" = "kaggle" ]; then
    echo ""
    echo "[Method 1] Downloading from Kaggle..."
    echo ""

    # 检查 kaggle CLI
    if ! command -v kaggle &> /dev/null; then
        echo "ERROR: kaggle CLI not found. Install it:"
        echo "  pip install kaggle"
        echo "  Then put your API token at ~/.kaggle/kaggle.json"
        echo "  Get token from: https://www.kaggle.com/settings -> API -> Create New Token"
        exit 1
    fi

    # 下载
    DOWNLOAD_DIR="${PROJECT_ROOT}/data"
    mkdir -p "$DOWNLOAD_DIR"
    echo "Downloading dschettler8845/brats-2021-task1 (~13GB)..."
    kaggle datasets download -d dschettler8845/brats-2021-task1 -p "$DOWNLOAD_DIR"

    # 解压
    echo "Extracting..."
    cd "$DOWNLOAD_DIR"
    unzip -q brats-2021-task1.zip -d BraTS2021_Training_Data 2>/dev/null || \
    unzip -q brats-2021-task1.zip 2>/dev/null

    # 清理 zip
    rm -f brats-2021-task1.zip
    echo "Done! Data at: ${DATA_DIR}"

# ------ 方式 2: MONAI DecathlonDataset ------
elif [ "$METHOD" = "monai" ]; then
    echo ""
    echo "[Method 2] Downloading via MONAI DecathlonDataset..."
    echo "This downloads MSD Task01_BrainTumour, then converts to BraTS format."
    echo ""

    # 检查 MONAI
    python3 -c "import monai" 2>/dev/null || {
        echo "ERROR: MONAI not installed. Run: pip install monai"
        exit 1
    }

    # Step 1: 下载 MSD 数据
    python3 -c "
from monai.apps import DecathlonDataset
from pathlib import Path
data_dir = Path('${PROJECT_ROOT}/data')
data_dir.mkdir(parents=True, exist_ok=True)
print('Downloading MSD Task01_BrainTumour (~4.5GB)...')
ds = DecathlonDataset(root_dir=str(data_dir), task='Task01_BrainTumour',
                      section='training', download=True, cache_rate=0.0)
print(f'Downloaded {len(ds)} cases')
"

    # Step 2: 转换为 BraTS 格式
    echo "Converting MSD format -> BraTS2021 format..."
    python3 "${SCRIPT_DIR}/convert_msd_to_brats.py" all

    echo "Done! Data at: ${DATA_DIR}"

else
    echo "ERROR: Unknown method '$METHOD'"
    echo "Usage: bash scripts/setup_data.sh [kaggle|monai]"
    exit 1
fi

# ------ 验证 ------
echo ""
echo "============================================================"
echo "  Verification"
echo "============================================================"
N_CASES=$(ls -d "${DATA_DIR}"/BraTS2021_* 2>/dev/null | wc -l)
echo "Total cases found: ${N_CASES}"

if [ "$N_CASES" -gt 0 ]; then
    SAMPLE=$(ls -d "${DATA_DIR}"/BraTS2021_* | head -1)
    echo "Sample case: $(basename $SAMPLE)"
    ls "$SAMPLE"
    echo ""
    echo "Expected: 5 files per case (_flair, _t1, _t1ce, _t2, _seg)"
fi

echo ""
echo "============================================================"
echo "  Next steps:"
echo "  1. Update configs/train_3d_unet.yaml -> data.root_dir"
echo "     to point to: ${DATA_DIR}"
echo "  2. For inference only (no retraining needed):"
echo "     python train.py --config configs/infer_sliding_window.yaml"
echo "============================================================"
