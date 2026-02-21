# BraTS 2021 脑肿瘤分割项目 — 详解 Walkthrough（中英文版）

> 本文档面向**需要深入理解每个技术细节**的读者，适合准备面试、答辩、或论文写作。
> 每个章节先讲"是什么、为什么"，再讲"怎么做"，最后附模拟面试 Q&A。

---

## 目录 / Table of Contents

1. [问题定义 Problem Definition](#1-问题定义-problem-definition)
2. [数据集 Dataset](#2-数据集-dataset)
3. [预处理 Preprocessing](#3-预处理-preprocessing)
4. [数据加载与增强 Data Loading & Augmentation](#4-数据加载与增强-data-loading--augmentation)
5. [模型架构 Model Architecture](#5-模型架构-model-architecture)
6. [损失函数 Loss Function](#6-损失函数-loss-function)
7. [训练策略 Training Strategy](#7-训练策略-training-strategy)
8. [推理 Inference](#8-推理-inference)
9. [评估指标 Evaluation Metrics](#9-评估指标-evaluation-metrics)
10. [实验结果 Results](#10-实验结果-results)
11. [可视化 Visualization](#11-可视化-visualization)
12. [模拟面试 Q&A Mock Interview](#12-模拟面试-qa-mock-interview)

---

## 1. 问题定义 Problem Definition

### 1.1 任务是什么？What is the task?

**脑肿瘤语义分割 (Brain Tumor Semantic Segmentation)**：给定一个病人的 3D 脑部 MRI 扫描（4种模态），模型需要对每一个体素 (voxel，即3D像素) 预测它属于哪种肿瘤区域。

用大白话说：**让 AI 在脑部 MRI 里自动画出肿瘤的边界**，帮助医生快速定位肿瘤位置、大小和类型。

### 1.2 为什么重要？Why does it matter?

- 手动标注一个病例需要神经外科医生 **30-60分钟**，自动分割只需 **几秒钟**
- 准确的分割直接影响：手术规划、放疗靶区、治疗效果评估
- BraTS 是医学图像分割领域**最权威的 benchmark** 之一

### 1.3 挑战在哪？What makes it hard?

- 肿瘤形状高度不规则 (irregular shapes)，没有固定模式
- 不同子区域（水肿、坏死核心、增强肿瘤）边界模糊
- 3D 数据量大（每个 volume 约 240×240×155 体素），对 GPU 显存要求高
- 类别不平衡：背景体素远多于肿瘤体素

---

## 2. 数据集 Dataset

### 2.1 BraTS 2021 概览

| 项目 | 详情 |
|------|------|
| 数据路径 | `/workspace/DataChallenge/data/BraTS2021_Training_Data/` |
| 总样本数 | 731 个病例（每个一个文件夹） |
| 文件格式 | NIfTI (`.nii.gz`)，医学影像标准格式 |
| 体素大小 | 约 240 × 240 × 155，各向同性 1mm spacing |

### 2.2 四种 MRI 模态 — 为什么需要4种？

每种模态对不同组织的**对比度**不同，就像不同颜色的滤镜能看到不同东西：

| 模态 | 英文全称 | 对什么敏感 | 在代码中的 key |
|------|----------|------------|----------------|
| **FLAIR** | Fluid Attenuated Inversion Recovery | **水肿区域**最亮（抑制了脑脊液信号） | `flair` |
| **T1** | T1-weighted | 正常解剖结构清晰，脂肪亮、水暗 | `t1` |
| **T1ce** | T1 contrast-enhanced (注射造影剂后) | **增强肿瘤**（血脑屏障被破坏处）亮 | `t1ce` |
| **T2** | T2-weighted | 水亮、脂肪暗，**水肿和肿瘤**都亮 | `t2` |

**关键理解**：模型把这 4 种模态当作 4 个**通道 (channels)** 拼在一起输入，类似 RGB 图像有 3 个通道，这里有 4 个。

```
代码位置: src/train.py:250-252
inputs = torch.cat([batch[mod].to(device) for mod in modalities], dim=1)
# 把 4 个 (B, 1, H, W, D) 沿 channel 维度拼成 (B, 4, H, W, D)
```

### 2.3 标签 Labels — BraTS 的独特标注方式

每个病例有一个 `*_seg.nii.gz`，每个体素取值为 0/1/2/4：

| 标签值 | 结构 | 中文 | 体素颜色(可视化中) |
|--------|------|------|-----|
| 0 | Background | 背景（正常脑组织） | 无色 |
| 1 | NCR/NET (Necrotic & Non-Enhancing Tumor) | 坏死核心 | 蓝色 |
| 2 | ED (Peritumoral Edematous tissue) | 瘤周水肿 | 绿色 |
| 4 | ET (GD-Enhancing Tumor) | 增强肿瘤 | 红色 |

> **注意**：没有标签 3！这是 BraTS 数据集的历史遗留问题（早期版本有 label 3 但后来合并了）。

### 2.4 评估区域 — 为什么不直接评估 4 个标签？

BraTS 的评估不是按单个标签算的，而是按**临床有意义的嵌套区域**：

```
               ┌────────────────────────────────────────┐
               │           Whole Tumor (WT)              │
               │   ┌──────────────────────────────┐     │
               │   │      Tumor Core (TC)          │     │
               │   │   ┌──────────────────┐       │     │
               │   │   │  Enhancing (ET)  │       │     │
               │   │   │    label = 4      │       │     │
               │   │   └──────────────────┘       │     │
               │   │     label = 1 (NCR/NET)       │     │
               │   └──────────────────────────────┘     │
               │       label = 2 (Edema)                 │
               └────────────────────────────────────────┘
```

| 区域 | 包含的标签 | 临床意义 |
|------|-----------|---------|
| **WT** (Whole Tumor) | 1 + 2 + 4 | 整个病变范围，指导手术切除边界 |
| **TC** (Tumor Core) | 1 + 4 | 实质性肿瘤（不含水肿），用于评估肿瘤大小 |
| **ET** (Enhancing Tumor) | 4 | 活跃增强肿瘤，反映肿瘤恶性程度 |

**关键理解**：这三个区域是**嵌套的、有重叠的** (ET ⊂ TC ⊂ WT)，所以不能用简单的 softmax 互斥分类。

### 2.5 数据划分 Train/Val Split

```python
# 代码位置: src/data/brats_dataset.py:101-107
train_cases, val_cases = train_test_split(
    all_cases,
    test_size=0.2,       # 20% 作为验证集
    random_state=42,     # 固定 seed 保证可复现
    shuffle=True,
)
# 结果: 约 584 训练 + 147 验证
```

为什么用 80/20？这是医学影像领域常用比例，因为：
- 数据量有限（731例），不能拿太多做验证
- 20% ≈ 147例，统计上已经足够稳定

---

## 3. 预处理 Preprocessing

### 3.1 为什么需要预处理？

原始 MRI 数据有很多"脏"的地方：
- **Bias field (偏置场)**：MRI 磁场不均匀导致同一种组织在不同位置亮度不同
- **噪声 (Noise)**：所有成像系统都有热噪声
- **方向/间距不一致**：不同医院的扫描仪设置不同

### 3.2 N4ITK 偏置场校正 (Bias Field Correction)

**什么是偏置场？** 想象你拍照时灯光不均匀，照片一边亮一边暗。MRI 也有类似问题——同一种脑组织在不同位置的信号强度不一致。

```python
# 代码位置: src/preprocess/bias_field.py:23-97

# 核心流程:
# 1. 用 Otsu 阈值自动生成脑部 mask（区分脑组织和背景）
mask = sitk.OtsuThreshold(image, 0, 1, 200)

# 2. 缩小图像加速计算（shrink_factor=4，即长宽高各缩4倍）
shrunken_image = sitk.Shrink(image, [4, 4, 4])

# 3. N4 算法在缩小的图像上估计偏置场
corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrector.Execute(shrunken_image, shrunken_mask)

# 4. 把估计的偏置场应用回原始分辨率
log_bias_field = corrector.GetLogBiasFieldAsImage(image)
corrected_image = image / sitk.Exp(log_bias_field)  # 除以偏置场=校正
```

**N4 算法原理**：用 B-spline 拟合一个光滑的低频场（偏置场），然后把原图除以这个场，使得同一组织的亮度均匀化。

### 3.3 强度归一化 (Intensity Normalization)

MRI 没有像 CT 那样的标准化单位（HU），不同扫描的绝对亮度值可能差很多。归一化让不同病例的值域统一。

**Z-score 归一化**（本项目使用的方法）：

```python
# 代码位置: src/preprocess/normalize.py:21-66

# 1. 只在非零体素上计算（排除背景空气）
mask = volume != 0
roi = volume[mask]

# 2. 先用百分位数裁剪极端值（去掉最亮/最暗的0.5%）
lo, hi = np.percentile(roi, [0.5, 99.5])
volume = np.clip(volume, lo, hi)

# 3. 标准化: (x - mean) / std
mean = roi.mean()
std = roi.std()
out = (volume - mean) / std
```

**为什么 nonzero_only?** 因为脑部 MRI 外面是空气（值=0），如果把空气算进去，mean 和 std 会被严重稀释。

### 3.4 重采样 (Resampling)

确保所有体素的物理尺寸一致（各向同性 1mm）：

```python
# 代码位置: src/preprocess/resample.py:24-64
# 图像用 B-spline 插值（平滑），标签用最近邻插值（保持整数值）
```

### 3.5 Notebook 演示的额外技术

`notebooks/00_preprocessing_demo.ipynb` 还演示了两个实际训练中没用到的技术：

1. **曲率流去噪 (Curvature Flow Denoising)**：各向异性扩散，平滑噪声但保留边缘
2. **刚体配准 (Rigid Registration)**：T1→T1ce 的 Euler3D 配准，使用 Mattes 互信息作为相似性度量

> **注意**：这些预处理模块 (`src/preprocess/`) 是独立的工具。实际训练流水线用的是 MONAI 的 on-the-fly transforms（详见第4章）。

---

## 4. 数据加载与增强 Data Loading & Augmentation

### 4.1 MONAI CacheDataset — 解决 I/O 瓶颈

3D 医学影像体积很大（每个 ~50MB），如果每个 epoch 都从磁盘读取会很慢。MONAI 提供了 `CacheDataset`：

```python
# 代码位置: src/data/brats_dataset.py:116-127
train_ds = CacheDataset(
    data=train_cases,        # 584 个 case 的路径 dict 列表
    transform=train_tf,      # 完整的 transform pipeline
    cache_rate=0.3,          # 缓存 30% 的数据到 RAM
    num_workers=4,           # 4 个进程并行处理
)
```

**cache_rate=0.3 是什么意思？**
- 30% 的样本（~175个）在第一次加载后会缓存到内存中
- 下次访问这些样本时直接从内存读取，跳过磁盘 I/O 和解压
- 剩下 70% 每次都从磁盘加载
- 如果 RAM 足够大，可以设 1.0（全部缓存），训练会更快

### 4.2 标签转换 — 核心的 ConvertToMultiChannelBasedOnBratsClassesd

这是本项目最重要的自定义 transform，决定了模型的输出方式：

```python
# 代码位置: src/data/transforms.py:52-93

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    def converter(self, img):
        # 输入: (1, H, W, D)，值为 0/1/2/4
        # 输出: (3, H, W, D)，每个通道是 0/1 的二值 mask

        tc = (img == 1) | (img == 4)     # TC: 标签1和4
        wt = (img == 1) | (img == 2) | (img == 4)  # WT: 标签1/2/4
        et = (img == 4)                   # ET: 只有标签4

        return torch.cat([tc, wt, et], dim=0)  # (3, H, W, D)
```

**为什么要这样转换？** 因为 WT、TC、ET 三个区域有重叠：
- 一个标签为 4 的体素，在 TC、WT、ET 三个通道上**都应该是 1**
- 这意味着不能用 softmax（互斥），必须用 **sigmoid**（每个通道独立判断是/否）

### 4.3 训练 Transform Pipeline — 14步详解

每个训练样本在送入模型前，都会经过以下处理。**理解顺序很重要**：

```
代码位置: src/data/transforms.py:108-207
```

#### Step 1-3: I/O 和类型转换
```
LoadImaged          ← 读取 5 个 NIfTI 文件 (4模态+1标签)，得到 numpy arrays
EnsureChannelFirstd ← 确保 shape 为 (C, H, W, D)，C=1
EnsureTyped         ← 图像转 float32，标签转 uint8
```

#### Step 4: 空间标准化
```
Orientationd(axcodes="RAS")  ← 统一方向到 RAS 坐标系
```
**RAS 是什么？** Right-Anterior-Superior，即 x轴指向右、y轴指向前、z轴指向上。不同医院的扫描可能用不同的方向约定，这一步保证统一。

#### Step 5: 标签转换
```
ConvertToMultiChannelBasedOnBratsClassesd  ← (1,H,W,D) → (3,H,W,D)
```
**必须在 Orientationd 之后**，否则方向翻转可能打乱标签值。

#### Step 6: 重采样
```
Spacingd(pixdim=(1.0, 1.0, 1.0))
  ← 图像: bilinear 插值
  ← 标签: nearest 插值（保持二值）
```

#### Step 7-9: 裁剪
```
CropForegroundd   ← 裁掉全是空气的区域，减小体积（用 FLAIR 作参考，margin=10）
SpatialPadd       ← 如果裁后小于 128³，padding 补到 128³
RandSpatialCropd  ← 随机裁一个 128×128×128 的 patch（每次训练不同位置！）
```

**为什么要裁到 128³？** 完整 volume (~240×240×155) 太大，一个 batch 放不下 GPU（batch_size=2 就需要 ~10GB 显存）。随机裁剪也是一种数据增强。

#### Step 10-11: 空间增强
```
RandFlipd    ← 随机翻转（3个轴各 50% 概率）
RandAffined  ← 随机仿射变换: 旋转±15°, 缩放 0.9-1.1x (概率50%)
```

**为什么要翻转和旋转？** 脑肿瘤可以出现在左脑或右脑，翻转让模型不偏向某一侧。旋转增加几何多样性。

#### Step 12-14: 强度增强
```
NormalizeIntensityd    ← z-score 归一化（每个通道独立，只在非零体素上）
RandScaleIntensityd   ← 乘以一个随机因子 1±0.1 (模拟不同扫描仪的亮度差异)
RandShiftIntensityd   ← 加一个随机偏移 ±0.1 (模拟基线偏移)
```

### 4.4 验证 Transform Pipeline

验证时**不做任何随机增强**，只做确定性的预处理：

```
Load → ChannelFirst → Type → Orient(RAS) → LabelConvert → Spacing(1mm) → CropForeground → Normalize
```

**关键区别**：验证时**不做随机裁剪**，使用完整 volume，配合 sliding-window inference。

---

## 5. 模型架构 Model Architecture

### 5.1 3D U-Net 是什么？

U-Net 是一个**编码器-解码器 (Encoder-Decoder)** 结构，名字来自它 U 形的网络拓扑：

```
输入 (4, 128, 128, 128)                        输出 (3, 128, 128, 128)
    │                                               ▲
    ▼                                               │
  ┌──────┐  skip connection  ┌──────┐
  │ 32ch │ ───────────────→ │ 32ch │    ← 128³
  │ 128³ │                   │ 128³ │
  └──┬───┘                   └──▲───┘
     │ stride-2 conv            │ upsample + concat
     ▼                          │
  ┌──────┐  skip connection  ┌──────┐
  │ 64ch │ ───────────────→ │ 64ch │    ← 64³
  │  64³ │                   │  64³ │
  └──┬───┘                   └──▲───┘
     │                          │
     ▼                          │
  ┌──────┐  skip connection  ┌──────┐
  │128ch │ ───────────────→ │128ch │    ← 32³
  │  32³ │                   │  32³ │
  └──┬───┘                   └──▲───┘
     │                          │
     ▼                          │
  ┌──────┐  skip connection  ┌──────┐
  │256ch │ ───────────────→ │256ch │    ← 16³
  │  16³ │                   │  16³ │
  └──┬───┘                   └──▲───┘
     │                          │
     ▼                          │
  ┌──────────────────────────────┐
  │   Bottleneck: 512ch, 8³     │    ← 最底层：空间分辨率最小，通道数最多
  └──────────────────────────────┘
```

**核心思想**：
- **编码器 (左边下行)**：逐步缩小空间尺寸、增加通道数 → 提取高层语义特征（"这是肿瘤"）
- **解码器 (右边上行)**：逐步恢复空间尺寸 → 精确定位（"肿瘤边界在哪"）
- **Skip connections (跳跃连接)**：把编码器的细节特征直接传给解码器 → 结合高层语义和底层细节

### 5.2 具体参数

```python
# 代码位置: src/train.py:364-372
model = UNet(
    spatial_dims=3,            # 3D 卷积（不是 2D！）
    in_channels=4,             # 输入: 4种MRI模态
    out_channels=3,            # 输出: 3个区域 (TC, WT, ET)
    channels=[32, 64, 128, 256, 512],  # 每层通道数
    strides=[2, 2, 2, 2],     # 每次下采样步长=2（空间尺寸减半）
    num_res_units=2,           # 每个 block 有 2 个残差单元
    norm="instance",           # Instance Normalization
)
```

| 属性 | 值 |
|------|------|
| 总参数量 | **19,223,978** (~19.2M) |
| 模型大小 | ~73.3 MB (FP32) / ~221 MB (checkpoint 含 optimizer state) |
| 编码器层数 | 5 层 (32→64→128→256→512) |
| 解码器层数 | 4 层 (512→256→128→64→32) |
| 最终输出层 | 1×1×1 Conv: 32ch → 3ch |

### 5.3 为什么选 Instance Normalization 而不是 Batch Normalization？

- **Batch Norm** 在 batch 维度上统计均值/方差 → batch_size 很小时(本项目=2)统计不稳定
- **Instance Norm** 在每个样本的每个通道内独立统计 → 不受 batch_size 影响
- 医学影像由于 GPU 显存限制，batch_size 通常很小，**Instance Norm 几乎是标配**

### 5.4 残差单元 (Residual Units)

每个 encoder/decoder block 内有 2 个残差单元：`output = Conv(BN(ReLU(x))) + x`

**好处**：缓解梯度消失，允许更深的网络，让训练更稳定。

---

## 6. 损失函数 Loss Function

### 6.1 Dice Loss — 直接优化评估指标

```python
# 代码位置: src/train.py:375-382
loss_fn = DiceLoss(
    sigmoid=True,           # 模型输出 logits，loss 内部做 sigmoid
    include_background=True,  # 计算所有 3 个通道
    to_onehot_y=False,      # 标签已经是 multi-channel 了
    smooth_nr=1e-5,         # 分子平滑项（避免 0/0）
    smooth_dr=1e-5,         # 分母平滑项
    batch=True,             # 在 batch 维度上求平均
)
```

### 6.2 Dice Loss 数学原理

对于单个通道的预测 p 和标签 g：

```
Dice Loss = 1 - (2 * Σ(p * g) + ε) / (Σp² + Σg² + ε)
```

其中 ε = 1e-5 是平滑项。

**为什么用 Dice Loss 而不是 Cross Entropy？**

| 特性 | Dice Loss | Cross Entropy |
|------|-----------|---------------|
| 类别不平衡处理 | 天然处理（关注重叠比例） | 需要权重或 focal loss |
| 与评估指标的关系 | **直接优化 Dice score** | 间接 |
| 梯度特性 | 预测全错时梯度可能很小 | 梯度更稳定 |

实际上很多 BraTS 参赛方案用 **Dice + CE 组合**（配置文件写的 DiceCELoss），但本项目最终实现只用了 **纯 Dice Loss**——这在实践中效果也很好，因为 Dice 直接优化我们关心的指标。

### 6.3 sigmoid=True 的含义

模型输出的是 raw logits（未归一化的实数），Dice Loss 内部先做 sigmoid 把它映射到 [0, 1]，然后再计算 Dice：

```
logits → sigmoid → soft predictions (0~1) → Dice Loss
```

这比先手动 sigmoid 再算 loss 更数值稳定。

---

## 7. 训练策略 Training Strategy

### 7.1 优化器: AdamW

```python
# 代码位置: src/train.py:384-388
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,              # 初始学习率
    weight_decay=1e-5,    # L2 正则化
)
```

**AdamW vs Adam**：AdamW 把 weight decay 从梯度更新中解耦出来（decoupled weight decay），在有 learning rate schedule 的情况下正则化效果更好。

### 7.2 学习率调度: CosineAnnealingLR

```python
# 代码位置: src/train.py:389-393
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=300,        # 完整余弦周期 = 300 epochs
    eta_min=1e-7,     # 最低学习率
)
```

**学习率变化曲线**：从 1e-4 → 按余弦曲线平滑下降 → 1e-7

```
LR
1e-4 ─╮
      │ ╲
      │   ╲
      │     ╲
      │       ╲
      │         ╲__
1e-7 ─┤              ───────
      └──────────────────── epoch
      0        150       300
```

**为什么用 Cosine 而不是 StepLR？** 余弦退火更平滑，避免了阶梯式下降导致的训练不稳定。

### 7.3 混合精度训练 (AMP — Automatic Mixed Precision)

```python
# 代码位置: src/train.py:256-261
scaler = GradScaler("cuda")

with autocast("cuda"):        # 前向传播用 FP16
    loss = loss_fn(model(inputs), labels)

scaler.scale(loss).backward()  # 反向传播（自动缩放防止下溢）
scaler.step(optimizer)         # 更新权重
scaler.update()               # 调整 scale factor
```

**FP16 的好处**：
- 显存减半 → 可以用更大 batch 或更大模型
- 计算速度提升 ~2x（Tensor Core 加速）
- GradScaler 自动处理数值下溢问题

### 7.4 Early Stopping

```python
# 代码位置: src/train.py:114-141
class EarlyStopping:
    patience = 50          # 连续 50 epoch 无改善就停止
    min_delta = 1e-4       # "改善"的最小阈值
```

**工作流程**：
1. 每个 epoch 结束后，检查 val mean Dice 是否比历史最佳提高了至少 0.0001
2. 如果提高了：重置计数器为 0
3. 如果没提高：计数器 +1
4. 计数器 ≥ 50：停止训练

**实际发生了什么**：最佳 Dice 在 epoch 277 达到 0.9215，之后 50 个 epoch 无改善，在 epoch 289（实际 epoch 277+12，但 early stopping 状态是在 epoch 277 后就有了一些微小改善，直到 289 才真正达到 50 次无改善）触发 early stopping。

### 7.5 Checkpoint 策略

```python
# 代码位置: src/train.py:526-552

# 三种检查点，三种用途：
save_checkpoint(save_dir / "best_model.pth", ...)    # val Dice 最佳时覆盖
save_checkpoint(save_dir / f"epoch_{epoch:04d}.pth")  # 每 10 epoch 留一个
save_checkpoint(save_dir / "latest.pth", ...)         # 每 epoch 覆盖（crash保护）
```

每个 checkpoint 保存：

```python
{
    "epoch": epoch,
    "model_state_dict": ...,      # 模型权重
    "optimizer_state_dict": ...,  # 优化器状态（含动量）
    "scheduler_state_dict": ...,  # 学习率调度器状态
    "scaler_state_dict": ...,    # AMP scaler 状态
    "best_dice": best_dice,
    "es_counter": ...,            # early stopping 计数器
    "es_best_score": ...,         # early stopping 历史最佳
    "wandb_run_id": ...,          # wandb run ID（续训时复用）
}
```

**为什么保存这么多？** 为了**完美断点续训 (resume)**：不只是恢复模型权重，还要恢复优化器动量、学习率位置、early stopping 状态，甚至 wandb 的 run 也能接上，曲线不会断。

### 7.6 训练历程

| Epoch | Train Loss | Val Loss | Mean Dice | WT Dice | TC Dice | ET Dice | LR |
|-------|-----------|---------|-----------|---------|---------|---------|-----|
| 2 | 0.890 | 0.933 | 0.291 | 0.610 | 0.068 | 0.194 | 1.00e-4 |
| 12 | 0.199 | 0.259 | 0.828 | 0.894 | 0.827 | 0.764 | 9.96e-5 |
| 52 | 0.090 | 0.104 | 0.900 | 0.916 | 0.907 | 0.878 | 9.28e-5 |
| 102 | 0.077 | 0.098 | 0.904 | 0.919 | 0.911 | 0.883 | 7.41e-5 |
| 152 | 0.071 | 0.087 | 0.915 | 0.928 | 0.925 | 0.891 | 4.90e-5 |
| 202 | 0.063 | 0.086 | 0.916 | 0.932 | 0.923 | 0.892 | 2.42e-5 |
| 252 | 0.062 | 0.081 | 0.921 | 0.932 | 0.930 | 0.900 | 6.28e-6 |
| **277 (best)** | **0.060** | **0.080** | **0.922** | **0.932** | **0.932** | **0.901** | **1.54e-6** |
| 289 (stop) | 0.061 | 0.080 | 0.921 | 0.932 | 0.931 | 0.900 | 4.31e-7 |

**训练总耗时**：~21.4 小时 (77,063 秒)，平均每 epoch ~267-339 秒

### 7.7 wandb 日志

训练分两段：
1. **Run 1** (`logs/train.log`): Epochs 1-6, wandb offline 模式（没联网）
2. **Run 2** (`logs/train_run2.log`): 从 epoch 7 resume，wandb online，run name = `rosy-haze-1`

wandb URL: `https://wandb.ai/qianxu0517-vitra-labs/brats-biohub`

---

## 8. 推理 Inference

### 8.1 Sliding Window Inference — 为什么不能直接塞整个 volume？

训练时裁剪 128³ patch，但推理时要对**完整 volume** 产生预测。如果直接塞进去：
- 显存不够（完整 volume 约 240×240×155，模型激活值会爆显存）
- 训练时只见过 128³ 的输入，推理时给更大的输入可能性能下降

**解决方案：滑动窗口 (Sliding Window)**

```python
# 代码位置: src/infer.py:161-168
outputs = sliding_window_inference(
    inputs=inputs,           # 完整 volume (1, 4, H, W, D)
    roi_size=[128, 128, 128], # 窗口大小
    sw_batch_size=4,         # 同时处理 4 个窗口
    predictor=model,
    overlap=0.5,             # 50% 重叠
    mode="gaussian",         # 高斯加权混合
)
```

**工作原理**：

```
完整 volume (如 160×200×160)
┌─────────────────────────────┐
│  ┌───────┐                  │
│  │ win 1 │                  │
│  │ 128³  │                  │
│  └───┬───┘                  │
│      │  ┌───────┐           │
│      │  │ win 2 │  ← 50% 重叠│
│      │  │ 128³  │           │
│      │  └───────┘           │
│         ...更多窗口...        │
└─────────────────────────────┘

每个窗口独立推理 → 重叠区域用高斯加权平均 → 拼成完整预测
```

**高斯加权 (Gaussian mode) 是什么？**
窗口边缘的预测不如中心可靠（边缘缺少上下文），所以给窗口中心更大的权重、边缘更小的权重（高斯分布），这样重叠区域的融合更平滑，避免拼接痕迹。

### 8.2 后处理：从模型输出到 BraTS 标签

```python
# 代码位置: src/infer.py:171-177

# 1. Sigmoid + 阈值 0.5
preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()[0]  # (3, H, W, D)

# 2. 拆分三个通道
tc, wt, et = preds[0], preds[1], preds[2]

# 3. 按优先级转回 BraTS 单标签格式
label_map = np.zeros_like(tc, dtype=np.uint8)
label_map[wt == 1] = 2    # ED (水肿) — 先填最大的区域
label_map[tc == 1] = 1    # NCR/NET — 覆盖水肿中属于核心的部分
label_map[et == 1] = 4    # ET — 最后填增强肿瘤（最小区域，最高优先级）
```

**为什么这个顺序？** 因为 WT ⊃ TC ⊃ ET：
- 先把所有 WT 区域标为 2 (ED)
- 然后把 TC 内的覆盖为 1 (NCR/NET)
- 最后把 ET 覆盖为 4

这样 label=2 就只剩下 "在 WT 中但不在 TC 中" 的部分，即纯水肿区。

### 8.3 空间逆变换

推理是在 RAS 方向、1mm spacing 的标准空间里做的，但最终预测要映射回**原始图像空间**：

```python
# 代码位置: src/infer.py:180-191

# 1. 从 RAS 转回原始方向
label_map = _reorient_ras_to_orig(label_map, orig_affine)

# 2. 如果 shape 不匹配，用最近邻插值 resize 回原始尺寸
if label_map.shape != orig_shape:
    scale = np.array(orig_shape) / np.array(label_map.shape)
    label_map = zoom(label_map, scale, order=0)  # order=0 = nearest neighbor

# 3. 保存为 NIfTI，使用原始的 affine 矩阵
nii = nib.Nifti1Image(label_map, affine=orig_affine)
```

---

## 9. 评估指标 Evaluation Metrics

### 9.1 四种指标详解

#### Dice Coefficient (Dice 系数)

```
Dice = 2|P ∩ G| / (|P| + |G|)
```

- 衡量两个集合的**重叠程度**
- 范围 [0, 1]，1 = 完全重叠
- **直觉**：如果预测和真值完全一样，Dice = 1；如果完全不重叠，Dice = 0
- BraTS 比赛的**主要评估指标**

#### IoU (Intersection over Union)

```
IoU = |P ∩ G| / |P ∪ G|
```

- 比 Dice 更严格（分母更大）
- Dice 和 IoU 的关系：`IoU = Dice / (2 - Dice)`
- 如果 Dice = 0.9，则 IoU ≈ 0.818

#### HD95 (95th Percentile Hausdorff Distance)

```python
# 代码位置: src/eval.py:61-81
# 计算预测表面到真值表面 + 真值表面到预测表面的距离，取 95% 分位数
```

- 衡量**边界的最大偏差**（单位: mm）
- 用 95% 而不是 100%，是为了对少量离群点鲁棒
- **越小越好**，理想值 = 0
- **直觉**：即使大部分边界很准，但如果有一小块预测跑偏了几厘米，HD95 会很大

#### Sensitivity (灵敏度/召回率)

```
Sensitivity = TP / (TP + FN)
```

- "真正有肿瘤的体素，模型漏掉了多少？"
- 范围 [0, 1]，1 = 没有遗漏
- 在医学中**漏诊比误诊更危险**，所以 sensitivity 很重要

### 9.2 区域提取 — eval 脚本如何工作

```python
# 代码位置: src/eval.py:90-96
def extract_regions(label_map):
    return {
        "TC": ((label_map == 1) | (label_map == 4)).astype(np.uint8),
        "WT": ((label_map == 1) | (label_map == 2) | (label_map == 4)).astype(np.uint8),
        "ET": (label_map == 4).astype(np.uint8),
    }
```

对**预测**和**真值**分别提取 3 个二值 mask，然后逐对计算 4 种指标。

---

## 10. 实验结果 Results

### 10.1 汇总表

| 区域 | Dice (mean±std) | IoU | HD95 (mm) | Sensitivity | Dice<0.7 的例数 |
|------|-----------------|-----|-----------|-------------|---------|
| WT | 0.925 ± 0.071 | 0.867 | 6.80 | 0.921 | 2/147 |
| TC | 0.921 ± 0.098 | 0.867 | 5.31 | 0.935 | 6/147 |
| ET | 0.891 ± 0.093 | 0.815 | 3.73 | 0.909 | 8/147 |
| **Overall** | **0.913** | **0.850** | **5.28** | **0.922** | — |

### 10.2 分布统计

| 区域 | Min | Q1 (25%) | Median | Q3 (75%) | Max |
|------|-----|----------|--------|----------|-----|
| WT Dice | 0.496 | 0.908 | 0.948 | 0.969 | 0.986 |
| TC Dice | 0.399 | 0.923 | 0.959 | 0.977 | 0.991 |
| ET Dice | 0.397 | 0.867 | 0.925 | 0.948 | 0.982 |

**观察**：中位数都比均值高，说明少数难例拉低了均值。绝大多数病例 Dice > 0.90。

### 10.3 典型案例分析

**Best case: BraTS2021_00816** — Mean Dice = 0.983
- 肿瘤较大、边界清晰、各子区域分明
- 模型在这种典型胶质母细胞瘤上表现最好

**Worst case: BraTS2021_00116** — ET Dice = 0.397
- 增强肿瘤区域非常小（可能只有几十个体素）
- 小目标检测是分割模型的普遍弱点

---

## 11. 可视化 Visualization

所有图表由 `reports/generate_figures.py` 生成，位于 `reports/figures/`：

| 图表 | 文件名 | 说明 |
|------|--------|------|
| 训练曲线 | `training_curves.png` | 左图: train/val loss 随 epoch 下降；右图: 三区域 Dice 随 epoch 上升 |
| 分割对比 | `segmentation_comparison.png` | 4个案例 (best/75th/median/25th)：FLAIR原图 vs 真值 vs 预测 |
| 指标箱线图 | `metrics_boxplot.png` | Dice/IoU/HD95 的分布，可以看到 outlier |
| 多视角展示 | `multi_view_prediction.png` | 最佳案例的轴位/冠状/矢状面对比 |
| 偏置场校正 | `outputs/bias_field/bias_field_comparison.png` | N4 校正前后对比 |
| 去噪 | `outputs/denoised/denoising_comparison.png` | 曲率流去噪前后对比 |
| 配准 | `outputs/registration/registration_demo.png` | T1→T1ce 刚体配准 棋盘格可视化 |

---

## 12. 模拟面试 Q&A Mock Interview

### 基础问题 Basic Questions

---

**Q1: 请简单介绍一下你的项目。**

**A:** 我做的是 BraTS 2021 脑肿瘤分割任务。用 3D U-Net 模型，输入 4 种 MRI 模态（FLAIR/T1/T1ce/T2），输出 3 个肿瘤子区域（WT/TC/ET）的分割 mask。整个项目基于 MONAI 框架，包含完整的训练、推理、评估流水线。在 147 例验证集上取得了 overall Dice 0.913 的结果。

---

**Q2: 为什么 BraTS 数据集的标签是 0/1/2/4，没有 3？**

**A:** 这是 BraTS 数据集的历史原因。早期版本 (BraTS 2012-2015) 中 label 3 表示 non-enhancing tumor，后来在 BraTS 2017+ 版本中 label 3 被合并到 label 1 (NCR/NET)，但标签编码没有重新排列，所以直接跳过了 3，保留 4 表示 enhancing tumor。

---

**Q3: 你的模型是多分类 (multi-class) 还是多标签 (multi-label)？为什么？**

**A:** 是 **multi-label**。因为 BraTS 的三个评估区域 (WT, TC, ET) 是嵌套的、有重叠的——ET 是 TC 的子集，TC 是 WT 的子集。一个标签为 4 的体素，在 TC、WT、ET 三个 mask 中都应该是 1。所以不能用互斥的 softmax (multi-class)，必须用独立的 **sigmoid** (multi-label)，每个通道独立判断"是否属于该区域"。

---

**Q4: 解释一下 Dice Loss 的公式和为什么用它。**

**A:** Dice Loss = 1 - 2|P∩G|/(|P|+|G|)。它直接优化 Dice coefficient，而 Dice 就是 BraTS 的主要评估指标，所以优化目标和评估目标完全一致。另外，Dice Loss 天然对类别不平衡鲁棒——即使肿瘤只占整个 volume 的 1%，Dice Loss 关注的是"预测和真值的重叠比例"，不会被大量背景体素淹没。如果用 Cross Entropy，背景 class 的 loss 会主导，模型可能倾向于全预测背景。

---

### 架构与设计问题 Architecture & Design Questions

---

**Q5: 为什么用 3D U-Net 而不是 2D U-Net？**

**A:** 脑肿瘤是三维结构，相邻层 (slice) 之间的连续性信息非常重要。2D U-Net 只能看到单个 slice，无法利用层间的空间关系（比如一个肿瘤跨越 30 个 slice）。3D U-Net 的 3D 卷积核可以同时在三个方向上提取特征，捕获完整的 3D 上下文。

代价是 3D 卷积的计算量和显存需求大得多（3×3×3 kernel 有 27 个参数，vs 3×3 kernel 的 9 个），所以需要裁剪到 128³ 的 patch 来训练。

---

**Q6: 为什么用 Instance Normalization 而不是 Batch Normalization？**

**A:** 3D 医学影像的 batch size 通常很小（本项目 batch_size=2），因为每个 volume 占用大量 GPU 显存。Batch Normalization 在小 batch 时均值和方差估计不准，会导致训练不稳定。Instance Normalization 在每个样本、每个通道内独立计算统计量，不受 batch size 影响。经验上在小 batch 的医学影像任务中 Instance Norm 几乎总是优于 Batch Norm。

---

**Q7: Skip connection 的作用是什么？如果去掉会怎样？**

**A:** Skip connection 将编码器的浅层特征直接传到解码器的对应层。浅层特征保留了更多的空间细节（边缘、纹理），深层特征包含更多语义信息（"这是肿瘤"）。Skip connection 让解码器能同时利用两者，实现精确的边界定位。

如果去掉 skip connection，解码器只能从 bottleneck 的低分辨率特征恢复空间信息，分割边界会变得非常模糊，小结构可能完全丢失。这也是 U-Net 相比普通 Encoder-Decoder (如 FCN) 的核心优势。

---

**Q8: 你的模型有多少参数？是怎么算的？**

**A:** 19,223,978 个参数（约 19.2M）。主要来自 3D 卷积层：每个 3×3×3 Conv3d 的参数量 = in_channels × out_channels × 27 + out_channels（bias）。参数量最大的是 bottleneck 附近的层——编码器最后一层 (256→512) 约有 3.5M 参数，解码器对应层 (768→256，含 skip 拼接后 512+256=768) 约有 5.3M 参数。

---

### 训练细节 Training Details

---

**Q9: 解释你的数据增强策略。为什么选择这些增强？**

**A:** 我用了几类增强：

1. **几何增强**：随机翻转（3轴）+ 随机仿射（旋转±15°，缩放 0.9-1.1x）。因为肿瘤可以出现在任何位置和大小，翻转消除左右偏置，旋转和缩放模拟不同扫描角度和肿瘤大小变化。

2. **强度增强**：随机缩放强度（×0.9-1.1）+ 随机偏移（±0.1）。模拟不同 MRI 扫描仪产生的亮度差异，增强模型对强度变化的鲁棒性。

3. **随机裁剪**：每次从完整 volume 中随机裁 128³，相当于隐式的位置增强——同一个 volume 在不同 epoch 看到不同区域。

注意我没有用颜色抖动、Mixup、CutMix 等自然图像常用的增强——MRI 的"颜色"有物理含义，不能随意改变。

---

**Q10: 混合精度训练 (AMP) 是什么？为什么要用？**

**A:** AMP 让模型在前向传播时用 FP16（半精度浮点数），反向传播的梯度和权重更新用 FP32。

好处：
- **显存减半**：FP16 只需 2 bytes vs FP32 的 4 bytes
- **速度提升**：NVIDIA GPU 的 Tensor Core 对 FP16 有硬件加速，计算速度约快 2x
- **精度几乎不损失**：GradScaler 自动缩放 loss，防止 FP16 的梯度下溢 (underflow)

在本项目中，AMP 使得 batch_size=2 的 128³ volume 能在单 GPU 上训练。不用 AMP 可能需要减小 batch 或 patch size。

---

**Q11: 什么是 CosineAnnealingLR？为什么 T_max=300？**

**A:** CosineAnnealingLR 让学习率按余弦曲线从初始值 (1e-4) 平滑降到最低值 (1e-7)。T_max=300 表示一个完整的余弦半周期是 300 epochs。

这比简单的 StepLR（每 N epochs 降一次）更好，因为：
- 开始阶段快速探索（LR 较大）
- 中间阶段逐渐精细化
- 后期 LR 非常小，做微调

如果实际训练在 epoch 289 就 early stop 了，LR 已经降到了 4.3e-7，非常接近最低值，说明模型已经充分收敛。

---

**Q12: Early stopping 的 patience=50 是怎么选的？会不会太大？**

**A:** 50 epochs 的 patience 对于 300 epochs 的训练来说约占 1/6，是比较保守（宽松）的选择。选较大的 patience 是因为：

1. 医学影像训练中 validation Dice 的波动比较大（尤其在高 Dice 区间，0.001 的变化都有意义）
2. 太小的 patience（如 10）可能在模型还没收敛完就提前停止
3. CosineAnnealing 的后期 LR 很小，改善本身就很缓慢

实际效果：最佳在 epoch 277，early stop 在 289，说明 patience=50 没有浪费太多训练时间（只多跑了 12 个 epoch），设置合理。

---

### 推理与评估 Inference & Evaluation

---

**Q13: Sliding window inference 的 overlap=0.5 是什么意思？为什么要 overlap？**

**A:** overlap=0.5 意味着相邻窗口有 50% 的重叠。对于 128 的窗口，步长 = 128 × (1-0.5) = 64。

为什么要重叠？因为卷积网络在 patch 边缘的预测质量比中心差（边缘缺少完整的感受野/上下文）。重叠保证每个体素都被多个窗口覆盖，重叠区域取加权平均，减少边缘伪影。

Gaussian mode 进一步改善这个问题——给窗口中心更高权重、边缘更低权重，使融合更自然。

---

**Q14: 为什么 HD95 用 95% 而不是 100%？**

**A:** 100% Hausdorff distance (HD100) 是两个表面之间的**最大**距离，对单个离群点极度敏感。比如如果预测在远离肿瘤的地方有一个 1-voxel 的假阳性，HD100 可能飙到 100mm+，完全不能反映整体分割质量。

HD95 取 95% 分位数，过滤掉最极端的 5% 距离，对少量离群点更鲁棒，同时仍能反映边界的整体偏差程度。这是医学影像分割评估的标准做法。

---

**Q15: Dice=0.913 在 BraTS 比赛中算什么水平？**

**A:** BraTS 2021 比赛冠军方案 (nnU-Net 系列) 的 Dice 通常在 0.92-0.93（验证集）。我们的 0.913 是一个相当扎实的 baseline，比赛排名大概在中等偏上。

差距主要来自：
- 没用 ensemble（集成多个模型可提高 1-2 个点）
- 没用 test-time augmentation (TTA)（推理时做多次增强取平均）
- 没用更复杂的后处理（如 connected component analysis）
- 模型相对简单（冠军方案通常用更大的网络或 transformer 架构）

---

### 高级问题 Advanced Questions

---

**Q16: 如果让你改进这个项目，你会怎么做？**

**A:** 按优先级排列：

1. **Test-Time Augmentation (TTA)**：推理时对输入做翻转/旋转，得到多个预测后取平均。几乎不需要修改代码，通常提升 0.5-1.5 个 Dice 点。

2. **Model Ensemble**：训练多个模型（不同 seed/不同 fold），预测时取平均。BraTS 比赛的标配。

3. **Post-processing**：对预测做 connected component analysis，去掉小的孤立假阳性区域。尤其能改善 HD95。

4. **更大的模型**：如 nnU-Net（自动配置最优参数）、Swin UNETR（Vision Transformer）、或 SegResNet。

5. **Deep Supervision**：在解码器的每一层都加 loss，不只在最后一层，帮助梯度传播。

6. **Dice + CE 组合 Loss**：纯 Dice Loss 在预测全错时梯度接近 0，加入 CE loss 可以稳定早期训练。

---

**Q17: 你的项目中 sigmoid 和 softmax 的选择是怎么考虑的？**

**A:** 这是一个核心设计决策：

- **Softmax** 假设各类**互斥**（每个体素只属于一个类），适用于标签不重叠的场景
- **Sigmoid** 假设各通道**独立**（每个体素可以同时属于多个类），适用于标签有重叠的场景

在 BraTS 中：
- 如果直接预测 4 个标签 (0/1/2/4) → 用 softmax + CE，这是 multi-class
- 如果预测 3 个重叠区域 (WT/TC/ET) → 用 sigmoid + Dice，这是 multi-label ← **本项目的选择**

本项目选择 sigmoid 的原因：评估就是按 WT/TC/ET 三个重叠区域做的，模型直接输出这三个区域的 mask，更直接。

但也有成功的方案用 softmax 预测 4 个标签再转换成 3 个区域——两种都可行，各有优劣。

---

**Q18: 什么是 CacheDataset？和普通 Dataset 有什么区别？**

**A:**

- **普通 Dataset**：每次 `__getitem__` 都从磁盘读文件 → transform → 返回。对于 NIfTI 文件（大约 50MB/个），磁盘 I/O 成为瓶颈。
- **CacheDataset**：第一个 epoch 正常加载，但把 transform 后的结果缓存到 RAM。后续 epoch 直接从 RAM 读取，跳过磁盘 I/O 和 CPU-heavy 的 transform（如 Spacing resample）。

`cache_rate=0.3` 表示缓存 30% 的数据，剩下 70% 每次重新加载。这是 RAM 和速度的折中——全缓存 (1.0) 需要约 584×50MB ≈ 29GB RAM。

MONAI 还有 `PersistentDataset`（缓存到磁盘）和 `SmartCacheDataset`（自动替换缓存内容），适合不同的 RAM/磁盘/数据量场景。

---

**Q19: 训练过程中你怎么知道模型没有过拟合？**

**A:** 从几个方面判断：

1. **Train vs Val loss 曲线**：Train loss (0.061) 和 Val loss (0.080) 差距不大，没有明显的 gap 扩大趋势。如果严重过拟合，train loss 会持续下降但 val loss 开始上升。

2. **Val Dice 没有下降**：从 epoch 100 到 289，val Dice 一直在缓慢提升（0.904→0.922），没有先升后降的典型过拟合曲线。

3. **数据增强**：我用了大量增强（翻转、旋转、缩放、强度抖动），这本身就是正则化手段。

4. **Weight decay=1e-5**：L2 正则化也在防止过拟合。

5. **Instance Norm** 也有轻微的正则化效果。

不过 val Dice 后期提升非常缓慢（0.904→0.922 花了 ~190 epochs），可能存在轻微的 over-fitting 到训练集的分布。更严格的验证可以用 5-fold cross-validation。

---

**Q20: 为什么推理时要把预测从 RAS 转回原始方向？**

**A:** 因为最终的预测文件需要和原始图像**在同一个坐标系**下，这样医生或后续分析工具才能正确叠加显示。

训练时把所有数据统一到 RAS 方向，是为了让模型看到统一的数据分布。但保存预测时，必须恢复到原始的方向和分辨率，这样：
- 预测 NIfTI 可以直接和原始图像对齐
- 评估脚本可以正确对比预测和真值
- 临床软件可以正确显示

这个"标准化→处理→逆变换"是医学影像处理的标准范式。

---

**Q21: 如果某个病例完全没有增强肿瘤 (ET=0)，你的评估怎么处理？**

**A:** 看 `eval.py` 中的处理：

```python
def dice_score(pred, gt):
    if pred.sum() + gt.sum() == 0:
        return 1.0  # 两边都为空 → 完美匹配
```

如果真值和预测的 ET 都为空，Dice = 1.0（正确识别了"没有增强肿瘤"）。如果真值为空但预测不为空（假阳性），Dice = 0.0（完全错误）。如果真值不为空但预测为空（全部漏掉），Dice = 0.0。

HD95 的处理类似：两边都空返回 0.0（完美），一边空返回 inf（无穷大距离）。在计算平均 HD95 时，inf 值会被过滤掉。

---

**Q22: 你这个项目能直接用在临床上吗？有什么差距？**

**A:** 不能直接用于临床，还有几个关键差距：

1. **监管审批**：医疗 AI 需要 FDA 510(k) 或 CE 标志认证
2. **外部验证**：只在 BraTS 数据集上测试，没有在其他医院的数据上验证泛化性
3. **Uncertainty estimation**：临床需要知道模型对预测有多"确信"，而不只是一个二值 mask
4. **鲁棒性**：对不同扫描协议、不同厂商 MRI、不同肿瘤类型的鲁棒性未知
5. **可解释性**：医生需要理解模型为什么做出某个预测
6. **失败案例处理**：某些难例（如 Dice < 0.5）在临床中是不可接受的

BraTS 比赛更多是学术研究，而非临床部署。

---

*文档结束。如有疑问，可参考源代码中的注释和 docstring。*
