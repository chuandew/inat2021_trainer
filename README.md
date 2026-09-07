# iNaturalist 2021 DingoFS Training

基于 PyTorch 的单 GPU 图片分类训练工具，使用 iNaturalist 2021 数据集和 ResNet-50，支持从 DingoFS 或本地文件系统读取图片。适合运行真实训练负载，观察数据加载、GPU 计算和 checkpoint 写入的表现。

- **训练与评估**：默认使用预训练 ResNet-50，支持 BF16 混合精度训练和 Top-1、Top-5 评估。
- **数据完整性**：通过 manifest 记录数据身份，全量内容复核可独立运行。
- **中断恢复**：保存模型、优化器及训练进度，从 checkpoint 继续运行。
- **运行观测**：输出 JSONL 日志，记录吞吐、数据加载等待、计算耗时和 GPU 内存。

## 环境要求

- Linux、Bash、Python 3.12，以及已安装的 [uv](https://docs.astral.sh/uv/getting-started/installation/)。
- 训练需要 CUDA 可用、支持 BF16 的 NVIDIA GPU，不支持 CPU 训练。
- 下载数据需要 wget、tar、flock 和 GNU coreutils；磁盘需同时容纳压缩包、解压数据和 checkpoint。

DingoFS 不是必需依赖。使用 DingoFS 时，应先完成挂载，再将数据路径指向挂载目录；本项目不负责部署或挂载文件系统。

## 快速开始

以下以 **mini 数据集**为例，先训练 200 steps。请在仓库根目录、同一个 Bash 会话中执行；已有数据的读者可以跳过下载和解压。

### 1. 安装

```bash
uv sync --locked --no-dev
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
python -c 'import torch; print("CUDA:", torch.cuda.is_available()); print("BF16:", torch.cuda.is_available() and torch.cuda.is_bf16_supported())'
```

训练前应看到 `CUDA: True` 和 `BF16: True`。manifest 创建及复核不需要 GPU。

### 2. 准备数据

选择存储目录。若已有图片，将 `DATA_ROOT` 改为实际图片根目录，不需要移动数据。

```bash
export BASE="$HOME/datasets/inat2021"
MODE=mini
TRAIN_SPLIT=train_mini
TRAIN_SAMPLES=500000
DATA_ROOT="$BASE/data"
MANIFEST="$BASE/manifest-${MODE}.json"
```

没有数据时，运行下载命令；支持断点续传，并检查归档 MD5：

```bash
bash scripts/download_inat2021.sh "$MODE"
```

下载成功后，将归档解压到数据目录：

```bash
test ! -e "$DATA_ROOT/$TRAIN_SPLIT" &&
  mkdir -p "$DATA_ROOT" &&
  tar -xzf "$BASE/archive/${TRAIN_SPLIT}.tar.gz" -C "$DATA_ROOT"

test ! -e "$DATA_ROOT/val" &&
  mkdir -p "$DATA_ROOT" &&
  tar -xzf "$BASE/archive/val.tar.gz" -C "$DATA_ROOT"
```

目录存在时，对应命令会停止，不覆盖已有图片。若上次解压中断，请先处理不完整的数据，不要继续训练。最终目录结构如下，两个 split 的类别目录必须一致：

```text
DATA_ROOT/
|-- train_mini/
|   |-- class-a/sample.jpg
|   `-- class-b/sample.jpg
`-- val/
    |-- class-a/sample.jpg
    `-- class-b/sample.jpg
```

### 3. 创建数据清单

设置后续命令共用的数据参数：

```bash
DATA_ARGS=(
  --data-root "$DATA_ROOT"
  --train-split "$TRAIN_SPLIT"
  --val-split val
  --expected-train-samples "$TRAIN_SAMPLES"
  --expected-validation-samples 100000
  --expected-classes 10000
)
```

首次准备好数据后创建 manifest：

```bash
inat2021-manifest "${DATA_ARGS[@]}" --output "$MANIFEST"
```

该命令完整读取图片并记录数据身份，成功时输出 `"status": "completed"`。**`--output` 会覆盖同名清单；已有 manifest 时，直接将 `MANIFEST` 指向它并跳过创建。**

### 4. 启动训练

使用新的输出目录运行训练：

```bash
OUTPUT_DIR="$BASE/runs/${MODE}-$(date -u +%Y%m%dT%H%M%SZ)"
TRAIN_ARGS=(
  "${DATA_ARGS[@]}"
  --dataset-manifest "$MANIFEST"
  --output-dir "$OUTPUT_DIR"
  --model resnet50
  --batch-size 256
  --workers 8
  --max-steps 200
  --warmup-steps 20
  --checkpoint-every 100
  --validation-batches 50
)
inat2021-train "${TRAIN_ARGS[@]}"
```

默认使用 `ResNet50_Weights.IMAGENET1K_V2`，首次运行可能下载预训练权重。离线运行须提前准备 Torch Hub 缓存；也可添加 `--weights none` 从随机权重开始训练。

命令完成 200 个训练 batch 后，会评估 50 个验证 batch。成功时退出码为 0，输出 `"status": "completed"`、验证集 loss、Top-1 和 Top-5；完整记录见输出目录中的 `train.jsonl`。

## 复核已有数据

数据迁移后、怀疑文件被修改时，可单独执行：

```bash
inat2021-manifest "${DATA_ARGS[@]}" --verify-existing "$MANIFEST"
```

成功时输出 `"status": "verified"`；不匹配时非零退出。该命令读取全部图片，但不会修改 manifest，也不会启动训练。首次创建清单后，数据未变化时不需要立即重复复核。

训练启动只检查 manifest、样本数量和类别映射，不重新计算全部图片的 SHA-256。因此应保持数据集不变；启动检查无法发现同数量、同尺寸的内容修改。内容复核检查文件字节，不保证所有 JPEG 都可正常解码，也不提供数据快照。

## 恢复训练

在保留原 `TRAIN_ARGS` 的会话中执行：

```bash
inat2021-train "${TRAIN_ARGS[@]}" --resume latest
```

新开终端时，先激活环境，再参考原输出目录的 `config.json` 重建参数。**使用原输出目录，保持原训练参数、数据路径和 manifest 不变**，不要重新生成运行目录或清单。

`--max-steps` 是整个 run 的总目标，不能通过增大它给旧 run 追加训练。恢复成功后，日志中会出现 `resume` 事件。

## 全量数据与常用参数

使用 full 数据集时，在快速开始第 2 步将 `MODE` 改为 `full`、`TRAIN_SPLIT` 改为 `train`、`TRAIN_SAMPLES` 改为 `2686843`，然后重新执行后续参数设置。它会使用独立的 manifest 和输出目录。

| 数据集 | 训练图片 | 验证图片 | 类别 | batch 256 时一轮 steps |
|---|---:|---:|---:|---:|
| mini | 500,000 | 100,000 | 10,000 | 1,953 |
| full | 2,686,843 | 100,000 | 10,000 | 10,495 |

要训练完整一轮，将新 run 的 `--max-steps` 设置为表中对应值；使用 `--validation-batches 0` 评估完整验证集。训练会丢弃最后不足一个 batch 的样本。

| 参数 | 用途 |
|---|---|
| `--batch-size` | 每个训练 batch 的图片数，按显存调整 |
| `--workers` | CPU 数据加载进程数，按 CPU、内存和加载等待调整 |
| `--learning-rate` | 学习率，默认 `0.05` |
| `--warmup-steps` | 学习率预热步数，必须小于 `--max-steps` |
| `--checkpoint-every` | 每隔多少 steps 保存一次 checkpoint；最后一步也会保存 |
| `--validation-batches` | 训练结束后评估的 batch 数，`0` 表示全部 |

## 输出文件

| 文件 | 内容 |
|---|---|
| `config.json` | 训练配置及数据身份 |
| `train.jsonl` | 训练进度、吞吐、数据加载等待、GPU 内存和评估指标 |
| `checkpoints/step-*.pt` | 模型及恢复状态 |
| `checkpoints/latest.json` | 最新 checkpoint 的位置和校验信息 |

数据和输出可以位于不同文件系统：`--data-root` 决定图片读取位置，`--output-dir` 决定日志和 checkpoint 写入位置。

查看完整命令参数：

```bash
inat2021-train --help
inat2021-manifest --help
```
