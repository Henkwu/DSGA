# DSGA 中文复现指南

[English README](README.md)

本仓库实现论文 **DSGA: Domain-Specific Generative Augmentation for Cross-Domain
Few-Shot Chest X-ray Disease Recognition** 的完整主流程，包括：

- ResNet-10 四层特征分支；
- miniImageNet 原图与胸片风格变换图的协同训练；
- 基于 episode 的多分支元训练；
- FLUX.1/SDXL 类别条件辅助图生成；
- 生成图自动技术筛查与人工审核清单；
- 只在 support 集引入生成图的 ChestX 元测试；
- 600 episodes 的平均准确率与 95% 置信区间；
- 原型相似度 Grad-CAM、论文 PDF 矢量图和 PNG 预览图。

生成图不是医生标注的真实胸片，不能用于临床诊断；评估 query 始终使用真实
ChestX 图像，生成图绝不会进入 query。

## 1. 环境

推荐 Python 3.10、PyTorch 2.3.1、CUDA 12.1：

```bash
conda env create -f environment.yml
conda activate dsga
python -m pip install -e .
```

只有运行 FLUX/SDXL 的机器需要额外安装：

```bash
python -m pip install -r requirements-generation.txt
```

## 2. 数据

miniImageNet 使用标准 64 个训练类，按类别文件夹组织：

```text
data/miniimagenet/train/类别名/*.jpg
```

下载 NIH ChestX-ray14 后生成目标域清单：

```bash
python prepare_chestx.py \
  --metadata /datasets/ChestX/Data_Entry_2017.csv \
  --images /datasets/ChestX/images \
  --output data/manifests/chestx.csv
```

目标类别为 Atelectasis、Cardiomegaly、Effusion、Infiltration、Mass、Nodule 和
Pneumothorax。ChestX 只由 `evaluate.py` 读取，不参与训练或调参。

## 3. 生成辅助 support 图

先在 Hugging Face 接受 FLUX.1-dev 的非商业许可并登录：

```bash
huggingface-cli login
python generate_support.py \
  --backend flux \
  --model black-forest-labs/FLUX.1-dev \
  --images-per-class 8 \
  --output data/generated \
  --cpu-offload
```

随后生成审核表：

```bash
python quality_control.py --input data/generated \
  --output data/generated/qc_manifest.csv
```

程序会检查分辨率、曝光、对比度、信息熵和灰度风格。仍需人工检查胸廓结构、
明显畸变、非医学物体和病灶合理性；仅对通过审核的行设置 `accepted=true`。

## 4. 训练与测试

论文超参数已写入 `configs/dsga.yaml`：协同训练 400 epochs；元训练 1200
episodes；四分支权重为 `(0.1, 0.27, 0.189, 0.441)`；各 shot 评估 600 episodes。

```bash
python train_collaborative.py --config configs/dsga.yaml
python train_meta.py --config configs/dsga.yaml
python evaluate.py --config configs/dsga.yaml
```

结果位于：

```text
outputs/dsga_resnet10/collaborative.pt
outputs/dsga_resnet10/meta.pt
outputs/dsga_resnet10/evaluation.json
```

CPU 全链路功能测试不需要下载真实数据：

```bash
python scripts/create_smoke_data.py
python train_collaborative.py --config configs/smoke.yaml
python train_meta.py --config configs/smoke.yaml
python evaluate.py --config configs/smoke.yaml
```

smoke 数值没有科研意义，只用于检查程序能否闭环运行。

## 5. 图片与可解释性

`assets/paper_figures/` 同时提供论文排版用 PDF 和 GitHub 展示用 PNG。
生成新的 Grad-CAM：

```bash
python visualize_attention.py \
  --checkpoint outputs/dsga_resnet10/meta.pt \
  --query /datasets/ChestX/images/query.png \
  --support-dir data/generated/Atelectasis \
  --output outputs/atelectasis_gradcam.png
```

完整的数据格式、消融命令、参数覆盖、故障排查和论文报告结果见
[英文 README](README.md)。公开仓库前请由所有权利人确认代码许可、数据及图片
再分发权限，并补充论文最终 DOI。

