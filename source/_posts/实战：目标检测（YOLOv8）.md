---
title: 实战：目标检测（YOLOv8）
date: 2026-07-29 08:00:00
categories: [实战项目]
tags: [YOLO, 目标检测, 计算机视觉, Ultralytics, 实战项目, 进阶教程]
description: 用 YOLOv8 实现实时目标检测：训练自定义数据集、调优超参数、部署到摄像头实时检测，支持 CPU/GPU/边缘设备。
---

## 引子

前面两篇实战，你学会了"这是猫"（分类）和"这是不是手写数字"——但真正现实场景的问题是：**猫在哪？**

- 自动驾驶：前方有行人和车辆，**它们在哪里？**
- 安防监控：画面里有没有人闯入？**闯入的人在哪个位置？**
- 工厂质检：这个零件有没有瑕疵？**瑕疵在哪个区域？**

目标检测 = 分类 + 定位。同一张图里，框出所有物体，并说出每个框里是什么。

YOLO（You Only Look Once）是目前工业界最主流的目标检测框架——**快（实时 30fps+）、准（mAP 追上两阶段检测器）、简单（Ultralytics 封装的 API 只有几行代码）。**

---

## 前置知识

- [计算机视觉入门：图像处理基础](/2026/07/03/计算机视觉入门：图像处理基础/)
- [实战：图像分类（CIFAR-10）](/2026/07/11/实战：图像分类（CIFAR-10）/)

---

## 一、环境安装

```bash
pip install ultralytics opencv-python matplotlib
```

看，就这么一个包。Ultralytics 团队把训练、验证、导出、推理全部集成到 `ultralytics` 里了。

验证安装：

```python
import ultralytics
ultralytics.checks()
```

```
Ultralytics 8.3.0 🚀 Python-3.12 torch-2.5.1 CUDA:0 (NVIDIA RTX 4060, 8188MiB)
Setup complete ✅ (8 CPUs, 2 GPUs)
```

---

## 二、用预训练模型跑推理（5 行代码）

先感受一下。什么都不用训练，下载一个预训练的 YOLOv8n（n=nano，最小最快）：

```python
from ultralytics import YOLO

# 下载预训练模型并推理
model = YOLO('yolov8n.pt')  # 自动下载，6.2MB
results = model('https://ultralytics.com/images/bus.jpg')

# 显示结果
results[0].show()
```

会弹出一个窗口，公交车、行人、交通灯全部框好。

### 选模型

| 模型 | 参数量 | mAP@50 | 速度（GPU） | 下载大小 |
|------|-------|--------|-----------|---------|
| YOLOv8n (nano) | 3.2M | 37.3 | **最快** | 6.2 MB |
| YOLOv8s (small) | 11.2M | 44.9 | 快 | 22.1 MB |
| YOLOv8m (medium) | 25.9M | 50.2 | 中等 | 51.3 MB |
| YOLOv8l (large) | 43.7M | 52.9 | 慢 | 87.1 MB |
| YOLOv8x (xlarge) | 68.2M | **53.9** | 最慢 | 136.3 MB |

**日常用 YOLOv8s**——精度和速度的甜点区。

---

## 三、训练自定义数据集

以 **口罩检测** 为例——你的任务：从图片中找出谁没戴口罩。

### 3.1 数据集格式

Ultralytics 使用 YOLO 标注格式：

```
datasets/
├── images/
│   ├── train/
│   ├── val/
├── labels/
│   ├── train/
│   ├── val/
└── data.yaml
```

每张图片对应一个 `.txt` 标注文件，每行一个目标：

```
<类别id> <x_center> <y_center> <宽度> <高度>
```

归一化到 [0, 1] 的相对坐标。例如：

```
# image.jpg 里有一个人在左上角戴口罩（类别 0），右下角有人没戴（类别 1）
0 0.25 0.30 0.20 0.35
1 0.75 0.70 0.18 0.40
```

### 3.2 准备数据

你可以标注自己的数据（用 LabelImg / Roboflow 标注），也可以直接用公开数据集：

```python
# Roboflow 一键下载（需要 API key）
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("roboflow-58fyf").project("face-mask-detection")
dataset = project.version(1).download("yolov8")
```

### 3.3 训练

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')  # 从 COCO 预训练权重开始

results = model.train(
    data='datasets/face-mask-detection-1/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,            # 10 轮没有提升自动停止
    lr0=0.01,               # 初始学习率
    augment=True,           # 启用数据增强
    cache=True,             # 缓存图片到 RAM 加速
)
```

训练过程中，Ultralytics 会实时输出：

```
     Epoch   GPU_mem   box_loss   cls_loss   dfl_loss   Instances   Size
      1/50      2.8G      1.462      1.832      1.245         97     640
     10/50      2.9G      0.921      0.743      0.986        132     640
     20/50      2.9G      0.753      0.512      0.874         84     640
     30/50      2.9G      0.642      0.389      0.721        103     640
     40/50      2.9G      0.587      0.323      0.668         91     640
     50/50      2.9G      0.541      0.287      0.612        105     640
```

完成后自动保存到 `runs/detect/train`。

### 3.4 验证

```python
metrics = model.val()
print(f"mAP@50: {metrics.box.map50:.3f}")     # 平均精度
print(f"mAP@50-95: {metrics.box.map:.3f}")    # 严格版 mAP
print(f"Precision: {metrics.box.mp:.3f}")      # 精确率
print(f"Recall: {metrics.box.mr:.3f}")          # 召回率
```

预期结果（口罩检测，简单的场景）：
```
mAP@50: 0.923
mAP@50-95: 0.687
Precision: 0.901
Recall: 0.888
```

---

## 四、实时摄像头检测

训练好的模型可以直接接摄像头：

```python
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

# 调用摄像头
results = model.predict(source=0, show=True, conf=0.5)
```

或者对视频文件做推理：

```python
results = model.predict(
    source='test_video.mp4',
    save=True,              # 保存标注后的视频
    conf=0.5,               # 置信度阈值
    iou=0.5,                # NMS IoU 阈值（重叠度高时保留哪个框）
    line_width=2,
    show_labels=True,
)
```

### 关键参数调优

- **conf（置信度阈值）**：默认 0.25。场景里干扰多就调高（0.5），目标太小就调低（0.1）
- **iou（NMS 阈值）**：默认 0.7。调低→同一目标不会出现多个框，但可能漏检

实际生产中的调法：先跑 100 张测试图，看漏检/误检的比例，再调这两个参数。

---

## 五、部署到边缘设备

YOLOv8 支持导出多种格式，覆盖从服务器到手机的部署场景：

```python
model = YOLO('runs/detect/train/weights/best.pt')

model.export(format='onnx')           # ONNX：通用格式，可部署到任何推理框架
model.export(format='tflite')         # TFLite：Android / 树莓派
model.export(format='ncnn')           # NCNN：手机端（骁龙/联发科NPU加速）
model.export(format='tensorrt')       # TensorRT：NVIDIA GPU 加速（3-5倍提速）
model.export(format='openvino')       # OpenVINO：Intel CPU / 神经计算棒
```

**实际速度对比**（口罩检测，640×640 输入）：

| 导出格式 | 设备 | 推理速度 | 部署难度 |
|---------|------|---------|---------|
| PyTorch | RTX 4060 | 2ms | - |
| TensorRT | RTX 4060 | **0.8ms** | 中等 |
| ONNX | CPU | 15ms | 简单 |
| ONNX | Jetson Nano | 40ms | 中等 |
| TFLite | 树莓派 4B | 120ms | 简单 |
| NCNN | 骁龙 8 Gen3 | 8ms | 中等 |

---

## 六、常见问题

### 6.1 小目标检测不好怎么办

YOLO 默认 640×640 输入。如果你要检测口罩、车牌、文字这种小目标：

```python
model.train(imgsz=1280, ...)  # 更大的输入尺寸 → 小目标特征更明显
```

但注意：**imgsz×2 → 推理时间 ×4**（分辨率平方成正比）。

### 6.2 类别不平衡

某些类别的样本多（戴了口罩），某些少（没戴口罩）：

```python
model.train(
    cls_pw=1.0,          # 类别权重，>1 增加少样本类别的 loss
    ...
)
```

或者在数据层面，对少样本类别多做数据增强。

### 6.3 隐私场景不能把数据上传到云端

YOLO 全本地部署。下载模型后完全离线运行，不需要联网。

---

## 七、完整训练代码

```python
"""yolo_train.py - YOLOv8 目标检测训练"""
from ultralytics import YOLO

# 1. 加载预训练模型
model = YOLO('yolov8s.pt')

# 2. 训练
results = model.train(
    data='datasets/face-mask-detection-1/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    lr0=0.01,
    augment=True,
    cache=True,
)

# 3. 验证
metrics = model.val()
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall: {metrics.box.mr:.3f}")
print(f"mAP@50: {metrics.box.map50:.3f}")

# 4. 导出
model.export(format='onnx')

# 5. 测试推理
results = model.predict(
    source='test_images/',
    save=True,
    conf=0.5,
    iou=0.5,
)
```

把这 20 行代码放到有 GPU 的机器上，配置好数据集路径，直接 `python yolo_train.py` 就能跑。

---

## 总结

目标检测跟图像分类的区别总结成一句话：**分类说"有什么"，检测说"有什么+在哪"。**

- YOLO 家族是事实上的工业标准——30fps 以上实时推理
- Ultralytics 把训练/验证/导出/推理集成在一个包，20 行代码落地
- 导出格式覆盖全平台（GPU/CPU/手机/嵌入式），一套代码到处跑

**进阶路线：**
- COCO 官方基准测试跑一遍，看你的模型跟 SOTA 差多少
- 挑战更难的场景：夜间检测、遮挡检测、密集场景
- 尝试 YOLOv8-seg（实例分割），框变成蒙版，更精准
- 学习 ByteTrack，把视频帧中的检测框关联成轨迹
