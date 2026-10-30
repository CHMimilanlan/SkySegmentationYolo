# SAM3 → YOLOv8s-seg 星空 / 地面分割

用现有图片，通过 SAM3 的文本提示 **`sky`** 生成天空伪标签，经过检查后转换为 Ultralytics YOLO 分割数据集，训练 **`yolov8s-seg.pt`**，最终输出原图尺寸的天空二值 mask。

只训练一个类别：**`0: sky`**。地面、建筑、树木等属于非天空背景，无须另标一个 ground 类。推理得到的所有 sky 实例取并集。若业务把画面严格分成天空 / 非天空，可取反得到非天空 mask；漏检时不能直接把整张图判定为可靠地面。

本项目提供代码和操作流程；没有附带你的图片、SAM3 权重或训练完成的 YOLO 权重。SAM3 输出是伪标签，最终效果取决于夜景样本覆盖和人工纠错。

## 1. 安装环境

建议 **Linux + NVIDIA GPU**；Windows 用户建议在 WSL2 Ubuntu 中完成 SAM3 标注。纯数据转换和检查不需要 GPU。SAM3 官方推荐 Python ≥ 3.12、PyTorch ≥ 2.7、CUDA 环境 ≥ 12.6；这里使用 Python 3.12 + CUDA 12.8 的 PyTorch wheel。显卡驱动须支持该 wheel，不能复用旧的 torch 1.x 环境。

以下命令在仓库根目录执行，示例使用 Bash：

```bash
git clone https://github.com/CHMimilanlan/SkySegmentationYolo.git
cd SkySegmentationYolo

conda create -n skyseg python=3.12 -y
conda activate skyseg
python -m pip install --upgrade pip
# 官方 SAM3 当前仍引用 pkg_resources，保留兼容版本。
python -m pip install 'setuptools>=68,<81' wheel

python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install -e '.[yolo,test]'

# 固定到编写本项目时核对过接口的官方 SAM3 源码版本。
git clone https://github.com/facebookresearch/sam3.git third_party/sam3
git -C third_party/sam3 checkout f1647e8d0c1d280991f4d1b7d974c2704537fe66
python -m pip install -e './third_party/sam3[notebooks]' --no-build-isolation
python -m pip check

python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); from sam3.model_builder import build_sam3_image_model; print("SAM3 imports OK")'
python -m skyseg --help
```

使用官方 `[notebooks]` 附加依赖补齐 SAM3 相关导入依赖，脚本运行本身不需要打开 Notebook。项目把 NumPy 限制为 `<2`、OpenCV 限制为 `<4.12`，以兼容官方 SAM3 的 NumPy 约束；不要随后单独升级 OpenCV。若服务器导入 `cv2` 提示缺少 `libGL.so.1`，Ubuntu 可安装系统库 `libgl1` 和 `libglib2.0-0`。

如果只做数据转换与 CPU 测试，不安装大模型：

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

### 获取 SAM3 权重

先在 [facebook/sam3](https://huggingface.co/facebook/sam3) 申请并获得访问权限，再登录；不把 token 写入代码或提交到 Git。

```bash
hf auth login
hf download facebook/sam3 sam3.pt --local-dir checkpoints/sam3
```

后续通过 `--checkpoint checkpoints/sam3/sam3.pt` 使用本地权重。也可以省略 `--checkpoint`，由官方 builder 自动从 Hugging Face 缓存或下载权重。本项目使用 **SAM3 的图片文本分割模型**，不是 SAM2，也不使用 SAM3.1 的视频 multiplex checkpoint。

## 2. 准备现有图片

把 JPG / PNG / BMP / TIFF / WEBP 放到一个目录，可以保留子目录。

| 示例路径 | 内容 |
| --- | --- |
| `data/raw/session_001/frame_0001.jpg` | 第一次拍摄的图片 |
| `data/raw/session_001/frame_0002.jpg` | 同一次拍摄的另一帧 |
| `data/raw/session_002/image.jpg` | 第二个独立拍摄场景 |
| `data/raw/session_003/image.jpg` | 第三个独立拍摄场景 |

同一次拍摄 / 同一段视频 / 同一固定场景的相似图片放在**同一个直接父目录**，随后使用 `--group-by parent`。不同目录中的同名文件不会覆盖，输出名称包含相对路径的哈希。

使用常规 **8-bit 可显示图像**。RAW / FITS / 科学 16-bit 图像应先使用统一、明确的拉伸策略导出为 8-bit RGB，不要让图片库随意截断。带 EXIF 旋转的图片会先转正，再以无损 PNG 保存为后续训练图片，保证 mask 与像素严格对齐。输出目录必须放在原始图片目录外。

## 3. 用 SAM3 生成 sky mask

先试前 20 张，确认夜景分割符合你的需求：

```bash
python -m skyseg annotate \
  --images data/raw \
  --output outputs/sam3 \
  --checkpoint checkpoints/sam3/sam3.pt \
  --prompt sky \
  --device cuda:0 \
  --confidence 0.5 \
  --limit 20
```

再处理全量图片；参数保持一致，去掉 `--limit`，加 `--resume`：

```bash
python -m skyseg annotate \
  --images data/raw \
  --output outputs/sam3 \
  --checkpoint checkpoints/sam3/sam3.pt \
  --prompt sky \
  --device cuda:0 \
  --confidence 0.5 \
  --resume
```

程序加载一次模型、逐图推理，不把整套数据集放进显存。默认 FP32；该模式使用原模型权重及普通 ViT MLP 计算，绕过上游强制转换为 BF16 的融合算子，避免 `mat1 and mat2 must have the same dtype, but got BFloat16 and Float`。只调整当前模型实例，不改动已安装的 SAM3 源文件。在支持 BF16 的 GPU 上可以从首次运行起加 `--precision bf16` 节省资源；该模式用同一个 autocast 上下文包住图像编码和文本分割，仍需先用样本比较效果。设备不支持 BF16 时回退到上述 FP32 路径。切换 prompt、阈值、精度、设备或权重位置时使用新的输出目录，防止混合不同设置。`cuda:1` 等设备编号会显式传给模型和 processor。

标注过程默认显示 **tqdm 进度条**，包含已处理数量、百分比、耗时、预计剩余时间、速度，以及 `ok` / `empty` / `error` / `skipped` 数量。`--resume` 跳过的已完成图片也会计入进度；错误信息单独输出，结束后保留最终统计，不再为每张图片打印一行进度。

如果旧版本已遇到这个 dtype 错误，先运行 `git pull --ff-only` 更新本仓库，然后**保留原命令的参数，添加 `--resume`** 重试。默认 `--precision fp32` 已修复，无须切换 BF16 或删除已有输出。成功项会跳过，`error` 项会重新处理。若你原来显式指定了 `--precision bf16`，续跑也保留该参数；改成 FP32 时需换新输出目录。失败记录现在还会保存完整 `traceback`，便于定位其他错误。

输出内容：

| 路径（相对 `outputs/sam3`） | 含义 |
| --- | --- |
| `images/<id>.png` | 经过 EXIF 转正、与 mask 对齐的 RGB 图片 |
| `masks/<id>.png` | 原图尺寸、单通道 PNG，天空 255，背景 0 |
| `overlays/<id>.jpg` | 蓝色半透明天空预览，用于人工检查 |
| `manifest.jsonl` | 图片、mask、来源、分数、天空面积和状态的对应关系 |
| `records/<id>.json` | 每张图片的独立记录，支持中断续跑 |
| `review.csv` | 可编辑的人工复核表 |
| `run.json` / `summary.json` | 标注配置和数量统计 |

同一张图中通过分数阈值的多个 sky mask 会取并集。**SAM3 没识别到天空不会自动当成真负样本**，而是标为 `empty`，默认排除。解码 / 推理报错标为 `error`，也不会当成背景；存在 error 时命令返回非零状态，但保留其他图片的结果。`--resume` 会重试失败项；原图文件改变时重新标注并清空该图原有审核结论。

## 4. 检查、修正伪标签

打开 `overlays/` 对照图像，重点看地平线、黑色树枝 / 建筑、灯光、反射、水面、云、极暗区域。默认天空面积不足 0.5% 或超过 99.5% 的图片会被标记需要检查；正常的全天空照片可以人工确认。

编辑 `outputs/sam3/review.csv`，保留原来的 `id`：

| decision | 含义 |
| --- | --- |
| 留空 | 自动采用正常非空结果；异常和 empty 仍排除 |
| `accept` | 已确认当前 mask，或接受 `mask_path` 指定的修正版 |
| `reject` | 排除该图 |
| `negative` | 人工确认整张图没有天空，生成全零 mask 和空 YOLO 标签 |

修改 mask 时建议把修正版单独保存到 `outputs/sam3/corrected_masks/`，**单通道、尺寸和 `images/` 中图片一致、数值为 0/255 或 0/1**。禁止模糊边缘、JPEG mask、缩放尺寸或反转语义。填写 `decision=accept`，`mask_path=corrected_masks/<id>.png`；相对路径以 **CSV 所在目录**为基准，也支持绝对路径。不要直接改原始输入图片。`error` 项需要先修复并重新标注。

完整制作过程会用到两种预览：本节检查 **SAM3 / 修正后的语义 mask**；构建后还要检查 `label_overlays/` 中的 **YOLO 多边形实际覆盖区域**。严格评估时，验证集和测试集应使用人工校正的 mask；只对照 SAM3 伪标签得到的指标主要反映与 SAM3 的一致性。

## 5. 制作 YOLO 分割数据集

建议采用已复核样本，并按拍摄场景分组：

```bash
python -m skyseg build \
  --annotations outputs/sam3 \
  --reviews outputs/sam3/review.csv \
  --approved-only \
  --output datasets/sky_v1 \
  --group-by parent \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

这条命令只纳入 `accept` / `negative`。若只是先跑通流程，可以去掉 `--approved-only`，自动纳入正常非空伪标签；有审核表时仍传 `--reviews` 才会应用人工结论。若图片完全独立、都放在一个目录，则用 `--group-by image`；**同一段视频帧不要这样划分**。

默认 80% / 10% / 10% 按**组数量**分 train / val / test；每组图片数量不同时，图片比例可能偏离该数值。至少需要 3 个独立组；只有 2 组可加 `--test-ratio 0`，暂时只做 train / val。有负样本时，每个训练 / 验证集仍须包含正样本。非常少的场景无法得到可信的泛化评估。

结果使用标准 YOLO 布局：

| 路径（相对 `datasets/sky_v1`） | 用途 |
| --- | --- |
| `images/{train,val,test}/<id>.png` | 训练、验证、测试图片 |
| `labels/{train,val,test}/<id>.txt` | `0 x1 y1 x2 y2 ...`，归一化坐标 |
| `masks/{train,val,test}/<id>.png` | 对应的完整语义参考 mask |
| `label_overlays/{train,val,test}/<id>.jpg` | YOLO 多边形转回 mask 的预览 |
| `data.yaml` | 可直接传入 Ultralytics 的数据配置 |
| `dataset_manifest.jsonl` | 每图 split、来源、标签及审核信息 |
| `conversion_report.jsonl` | 转换质量、纳入 / 排除原因 |
| `selection_summary.json` | 划分前的可用图片、组数、审核结论和排除原因统计；失败时也保存 |
| `build_config.json` / `summary.json` | 构建配置、种子和统计 |

**mask 转 polygon 的限制**：YOLOv8-seg 是实例分割，普通一行 polygon 无法表达内部孔洞。程序对不连通的天空分别生成标签，不仅保留最大区域；默认忽略轮廓面积不足 64 像素的碎片，简化误差 1 像素。然后把保存的多边形重新栅格化，与原始 mask 计算 IoU；低于 **0.98** 的图片默认排除，记录 `polygon_roundtrip_loss`，避免静默把树冠中的地物孔洞填成天空。

原始二值 mask 始终保留。若很多图片被排除，先检查 `holes` / `polygon_iou` / `dropped_components`，可用 `--epsilon 0 --min-area 0` 减少简化损失；这仍不能表达孔洞。降低 `--min-polygon-iou` 是有意识接受标签近似，不代表解决了孔洞问题。若必须精确描述复杂树枝 / 孔洞边界，应评估直接使用语义分割模型。

重复像素图片在划分前去重；完全相同图像对应冲突 mask 时直接报错。按父目录分组不能自动识别跨目录的近重复场景，原始目录组织仍需检查。输出目录必须为空；重新构建使用 `datasets/sky_v2` 等新目录。若样本过少导致构建失败，诊断会保存在输出目录中，修正后同样换一个新目录。

若构建提示没有可用图片，先查看 `selection_summary.json` 的 `eligible_images` 和 `exclusion_reasons`。最常见的 `not_reviewed` 表示使用了 `--approved-only`，但审核表中的 `decision` 仍为空；自动生成审核表不会自动批准样本。检查 mask 后填写 `accept` / `negative`，或者在明确使用未审核伪标签时去掉 `--approved-only`。不要批量把未检查的图片标成已审核。`found 0` 是筛选后没有可用图片，修改划分比例不能解决；有图片但只有一个原始父目录，则是分组不足，独立散图可用 `--group-by image`，视频帧应继续按场景分组。

```bash
python -m skyseg validate --data datasets/sky_v1/data.yaml
```

会检查标签格式 / 坐标 / 面积、图像-mask 尺寸、内容哈希、负样本空标签、重复图像、场景组泄漏和转换一致性。生成的 YAML 使用绝对数据路径；移动数据集到其他机器后修改 YAML 中的 `path`，再重新 validate。dataset manifest 内部均为相对路径。

校验会立即输出配置路径、目录扫描和各 split 的图片数量，然后显示 tqdm 进度、当前文件及检查阶段。每张图片、mask 和标签只读取一次，同时完成全部完整性检查；不缓存整套解码图像。图片 / mask 解码失败时，报错会带上对应文件路径。大图、PNG 解压或慢速磁盘仍可能耗时，进度条会停留在当前文件，不能仅凭解码函数处的堆栈判断文件损坏。

## 6. 训练 YOLOv8s-seg

`train` 首先执行上述数据校验，再加载 YOLO 和启动训练。启动时会立即打印 `[train]` / `[validate]` 阶段信息；校验完成前不会出现 YOLO 的 epoch / loss 日志。需要同时保存终端输出时，可使用 `python -u -m skyseg train ... 2>&1 | tee train.log`。

先检查参数和数据；不加载模型、不占 GPU：

```bash
python -m skyseg train \
  --data datasets/sky_v1/data.yaml \
  --config configs/train_yolov8s_seg.yaml \
  --dry-run
```

正式训练：

```bash
python -m skyseg train \
  --data datasets/sky_v1/data.yaml \
  --config configs/train_yolov8s_seg.yaml \
  --device 0 \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --name sky_yolov8s
```

默认加载 `yolov8s-seg.pt` 的 COCO 预训练权重并微调为单类 sky，首次使用需要下载。离线时自行准备该权重并传 `--model /path/to/yolov8s-seg.pt`。

配置关闭 Mosaic、垂直翻转、旋转，采用温和的亮度 / 缩放 / 平移增强；设置 `mask_ratio: 1` 减少训练 mask 下采样，`overlap_mask: false` 保留独立天空区域。显存不足先把 `--batch 8` 改为 `4` 或 `2`；边界过粗可在资源允许时尝试 `--imgsz 960`，不要预设不同分辨率的速度或精度。

首次默认权重路径：

```text
runs/segment/sky_yolov8s/weights/best.pt
runs/segment/sky_yolov8s/weights/last.pt
```

相同名称已存在时 Ultralytics 会创建带序号的目录，以训练日志实际路径为准。训练结果含损失 / 指标 / 示例图，额外保存 `pipeline_environment.json`。需要中断续训：

```bash
python -m skyseg train \
  --data datasets/sky_v1/data.yaml \
  --model runs/segment/sky_yolov8s/weights/last.pt \
  --device 0 --resume
```

`--resume` 恢复 checkpoint 内的训练设置和 epoch，不采用新 YAML 的 epochs / batch / imgsz 等参数；用相同数据集。想换数据 / 训练设置，用 `--model best.pt` 开启一次新的微调，不加 `--resume`。保留数据目录、构建配置、SAM3 源码版本以及 `python -m pip freeze` 的输出；固定 seed 不保证跨 GPU / CUDA 版本逐位一致。

## 7. 评估：实例 mAP + 天空像素 IoU

Ultralytics 常规实例分割指标：

```bash
yolo segment val \
  model=runs/segment/sky_yolov8s/weights/best.pt \
  data=datasets/sky_v1/data.yaml split=test imgsz=640 device=0
```

天空本质上更关注整体区域与边界，因此同时计算所有 sky 实例合并后的像素指标：

```bash
python -m skyseg evaluate \
  --model runs/segment/sky_yolov8s/weights/best.pt \
  --data datasets/sky_v1/data.yaml \
  --split test \
  --output outputs/evaluation_v1 \
  --device 0 --imgsz 640 --confidence 0.25
```

输出 `semantic_metrics.json`、逐图 `per_image.jsonl`、预测 mask 和预览。指标包括全局像素混淆计数、sky IoU、非天空 IoU、mIoU、sky Dice、像素准确率，以及逐图 sky IoU 的均值。预测和参考都没有某类别时，该类别 IoU 定义为 1。**没有人工真值时这些是对伪标签的一致性评估，不能当成真实分割精度。** 没有 test 集则使用 `--split val`；调参看 val，最终结果看独立 test。

训练 / 验证使用的是 polygon 近似与训练分辨率；这里对照保存的原始语义 mask，因此与 YOLO mAP 或转换 IoU 不会相同。高像素准确率还可能受天空 / 地面面积比例影响，不能只看这一项。

## 8. 对新图片输出二值 mask

```bash
python -m skyseg predict \
  --model runs/segment/sky_yolov8s/weights/best.pt \
  --images data/new_images \
  --output outputs/predict_v1 \
  --device 0 --imgsz 640 --confidence 0.25 \
  --save-ground
```

`masks/` 是原图尺寸 sky mask，`ground_masks/` 是可选的非天空补集，`overlays/` 用于检查。`predictions.jsonl` 保存原图与 mask 的对应关系和状态。没有检测到 sky 时状态为 `no_sky_detection`，mask 为全零，需要结合实际图像判断是否漏检。

Python 读取：

```python
from PIL import Image
import numpy as np

sky = np.asarray(Image.open("path/to/mask.png")) > 0  # [H, W], bool
non_sky = ~sky
```

## 9. 代码与验证范围

| 文件 | 功能 |
| --- | --- |
| `skyseg/annotate.py` | SAM3 图片文本分割、mask 合并、断点记录 |
| `skyseg/sam3_precision.py` | FP32 模式绕过 SAM3 强制 BF16 的融合 MLP |
| `skyseg/dataset.py` | 人工审核、修正 mask、去重、按组划分 |
| `skyseg/polygons.py` | mask → YOLO polygon、回转精度检查 |
| `skyseg/validate.py` | 数据完整性与泄漏检查 |
| `skyseg/train.py` | Ultralytics 训练 / 续训 / dry-run |
| `skyseg/inference.py` | 二值预测与像素指标 |
| `configs/train_yolov8s_seg.yaml` | YOLOv8s-seg 初始训练配置 |
| `tests/test_pipeline.py` | 不依赖模型的 CPU 回归测试 |
| `tests/test_validation.py` | 校验进度、单次读取、损坏文件路径及完整性检查回归测试 |
| `tests/test_sam3_precision.py` | 使用真实 PyTorch CPU 张量复现及验证 dtype 修复 |

测试集包含 25 项数据流程 / 校验测试，以及 9 项真实 PyTorch 张量精度回归测试。本次校验日志更新在 Python 3.12、NumPy 1.26、OpenCV 4.11 环境中通过全部 25 项数据流程 / 校验测试和 `pip check`；未安装 torch，精度测试模块跳过。9 项精度测试此前已在 PyTorch 2.10 CPU 环境中通过：用同样的强制 BF16 融合算子加 FP32 线性层复现原错误，验证 FP32 修复后结果与普通 MLP 一致、参数保持不变、不会修改其他模型实例、外层 autocast 被正确隔离、BF16 回退以及图像 / 文本阶段共享上下文。**测试使用小型模型或模拟 SAM3 / YOLO 接口；尚未在你的 GPU、真实图片或完整权重上执行 SAM3 推理与 YOLO 训练。**

参考官方接口（2026-10-29 核对）：

- [SAM3 README / 安装 / 图片文本分割](https://github.com/facebookresearch/sam3)
- [SAM3 图片 processor](https://github.com/facebookresearch/sam3/blob/f1647e8d0c1d280991f4d1b7d974c2704537fe66/sam3/model/sam3_image_processor.py)
- [PyTorch 安装版本对应关系](https://pytorch.org/get-started/previous-versions/)
- [Ultralytics YOLO 分割数据格式](https://docs.ultralytics.com/datasets/segment/)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- [Ultralytics 分割训练与推理](https://docs.ultralytics.com/tasks/segment/)

SAM3 和 Ultralytics 的源码、权重使用分别遵循其上游许可证；本仓库不重新分发这些资产。
