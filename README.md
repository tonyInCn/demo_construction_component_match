# 建筑构件识别系统

基于深度学习（GroundingDINO + CLIP）和传统计算机视觉的装配式建筑构件自动检测与识别系统。

## 目录

- [项目概述](#项目概述)
- [技术架构](#技术架构)
- [目录结构](#目录结构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [功能特性](#功能特性)
- [检测流程](#检测流程)
- [API 文档](#api-文档)
- [模型说明](#模型说明)
- [配置说明](#配置说明)

---

## 项目概述

本系统用于在施工现场视频中自动检测和识别装配式建筑构件（预制柱、预制梁、预制板、预制墙板等），支持实时视频分析和批量视频处理。

**核心特点：**
- 多方法融合检测：深度学习 + 传统 CV，提高检测准确率
- 实时检测：支持视频流实时分析
- 构件匹配：基于 CLIP 语义匹配对检测到的构件进行精确分类
- 可视化界面：实时显示检测进度、检测结果、构件信息

---

## 技术架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         前端 (Vue 3)                            │
│  ┌─────────────────┐  ┌─────────────────────────────────────┐  │
│  │  视频检测面板    │  │  检测详情面板                       │  │
│  │  · 视频播放      │  │  · 模型加载进度                    │  │
│  │  · 检测标注显示  │  │  · 运行状态                         │  │
│  │                  │  │  · 检测步骤实时显示                │  │
│  └─────────────────┘  │  · 检测结果表格                     │  │
│                        └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP API
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      后端 (FastAPI)                             │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                      Pipeline Router                        │ │
│  │  /api/pipeline/model_status    - 模型状态                   │ │
│  │  /api/pipeline/upload_video   - 视频上传                   │ │
│  │  /api/pipeline/start           - 启动检测                   │ │
│  │  /api/pipeline/stop/{id}       - 停止检测                   │ │
│  │  /api/pipeline/status/{id}     - 运行状态                  │ │
│  │  /api/pipeline/frame_results/{id} - 帧结果                 │ │
│  │  /api/pipeline/stream_video    - 视频流                    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Processor (检测流程)                      │ │
│  │  1. 吊机检测 (CraneDetector)                               │ │
│  │  2. 确定检测区域                                           │ │
│  │  3. 多方法构件检测 (MultiMethodDetector)                    │ │
│  │  4. 目标跟踪                                               │ │
│  │  5. 标注帧生成                                             │ │
│  │  6. 构件匹配与判定                                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              ModelDetector (模型检测)                       │ │
│  │  ┌───────────────────┐  ┌───────────────────┐             │ │
│  │  │  GroundingDINO    │  │      CLIP         │             │ │
│  │  │  开放词汇目标检测  │  │   语义相似度匹配   │             │ │
│  │  └───────────────────┘  └───────────────────┘             │ │
│  │  ┌───────────────────────────────────────────┐             │ │
│  │  │        传统CV检测 (边缘+轮廓+特征匹配)      │             │ │
│  │  └───────────────────────────────────────────┘             │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
demo_construction_component_match/
├── backend/                              # 后端代码
│   ├── pipeline/                         # 检测流水线
│   │   ├── model_detector.py             # 模型检测器（GroundingDINO + CLIP + 传统CV）
│   │   └── processor.py                  # 检测流程处理器
│   ├── routers/                          # API 路由
│   │   └── pipeline_router.py            # 检测流水线路由
│   ├── config.py                         # 配置文件
│   ├── logger.py                         # 日志模块
│   └── main.py                           # FastAPI 主入口
├── frontend/                             # 前端代码
│   ├── css/styles.css                    # 样式表
│   ├── js/
│   │   ├── api.js                        # API 客户端
│   │   └── dashboard.js                  # 控制面板逻辑
│   └── index.html                        # 主页面
├── video_input/                          # 上传视频存储
├── output/                               # 检测输出
│   ├── annotated/                        # 标注帧
│   └── rois/                             # 检测到的 ROI
├── start.py                              # 启动脚本（清理端口+启动服务）
├── stop.py                               # 停止脚本
└── README.md                             # 本文档
```

---

## 环境要求

| 组件 | 要求 |
|------|------|
| Python | 3.10+ |
| CUDA | 11.8+ (GPU 推理) |
| 显存 | ≥ 4GB |
| 操作系统 | Windows / Linux |

### Python 依赖

```
fastapi>=0.100.0
uvicorn>=0.23.0
torch>=2.0.0
transformers>=4.30.0
opencv-python>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
huggingface-hub>=0.16.0
```

### 安装步骤

```bash
# 1. 克隆项目
git clone <repository-url>
cd demo_construction_component_match

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 首次运行会自动下载模型（约 2GB）
#    GroundingDINO: IDEA-Research/grounding-dino-tiny
#    CLIP: openai/clip-vit-base-patch32
```

---

## 快速开始

### 启动服务

```bash
# 方式1: 使用启动脚本（推荐）
python start.py

# 自定义端口
python start.py 8080

# 方式2: 直接启动
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 停止服务

```bash
python stop.py
# 或指定端口
python stop.py 8080
```

### 访问系统

- **Web 界面**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/api/health

### 使用流程

1. 打开浏览器访问系统
2. 等待模型加载完成（页面显示"● 就绪"）
3. 点击"选择"按钮上传视频文件
4. 点击"▶ 开始运行"启动检测
5. 实时查看检测进度和结果

---

## 功能特性

### 1. 视频输入

- 支持格式：MP4, AVI, MOV, MKV, WMV, FLV, WEBM, M4V, MPG, MPEG
- 最大文件大小：500MB
- 本地文件上传

### 2. 吊机检测

```python
# 检测策略：
1. 颜色检测：HSV 空间查找工程设备色系
   - 黄色: H(18-38), S(80-255), V(100-255)
   - 橙色: H(5-18), S(100-255), V(100-255)
   - 红色: H(0-5) 或 H(160-180)
2. 形态学处理：去噪、填充
3. 轮廓分析：查找塔身（竖向细长）和吊臂（横向长条）
4. 区域扩展：围绕塔吊生成工作区域
```

### 3. 多方法构件检测

#### 方法 A: GroundingDINO（深度学习）

- **模型**: IDEA-Research/grounding-dino-tiny
- **类型**: 开放词汇目标检测
- **优势**: 可检测训练集中不存在的类别
- **检测对象**:
  ```
  tower crane, precast column, precast beam, precast slab,
  precast wall, precast staircase, building component,
  construction machinery, concrete structure
  ```

#### 方法 B: 传统 CV 检测

```python
# 检测策略：
1. Canny 边缘检测
2. 轮廓提取与筛选
3. 形状分析（宽高比、面积、颜色）
4. 特征匹配（与构件库进行相似度比对）
```

#### 结果融合

```python
# NMS (非极大值抑制) 融合策略：
1. 收集所有方法的检测结果
2. 按置信度排序
3. 应用 IoU 阈值 (0.5) 去除重叠检测
4. 保留高置信度检测
```

### 4. 构件匹配

```python
# CLIP 语义匹配：
1. 提取检测区域 (ROI)
2. 编码图像特征向量
3. 与构件库进行余弦相似度比对
4. 返回最佳匹配结果

# 构件库 (8 种构件)：
┌─────────────────────────────────────────────────────────────┐
│ ID       │ 名称          │ 形状          │ 颜色      │ 置信阈值 │
├─────────────────────────────────────────────────────────────┤
│ comp_001 │ 预制柱-标准柱  │ rect_vertical │ gray      │ 0.45    │
│ comp_005 │ 预制梁-主梁    │ rect_horizontal│ gray     │ 0.45    │
│ comp_012 │ 预制板-楼板    │ flat_large    │ light_gray│ 0.45    │
│ comp_023 │ 预制墙板-外墙  │ rect_vertical │ textured  │ 0.45    │
│ comp_034 │ 预制楼梯      │ stepped       │ gray      │ 0.45    │
│ comp_041 │ 预制节点-连接  │ small_circle  │ metallic  │ 0.45    │
│ comp_056 │ 预制柱-顶层    │ rect_vertical │ gray      │ 0.45    │
│ comp_067 │ 预制墙-内墙    │ rect_vertical │ light     │ 0.45    │
└─────────────────────────────────────────────────────────────┘
```

---

## 检测流程

```
视频输入
    │
    ▼
┌─────────────────────────────────────────────┐
│ Step 1: 吊机检测                             │
│ · HSV 颜色空间检测                           │
│ · 轮廓分析                                   │
│ · 生成工作区                                 │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ Step 2: 确定检测区域                         │
│ · 使用检测到的工作区                          │
│ · 生成区域掩码                               │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ Step 3: 多方法构件检测                       │
│ ┌─────────────────────────────────────────┐ │
│ │ 方法 A: GroundingDINO 深度学习检测      │ │
│ │ · 分辨率自适应 (max_side=640)           │ │
│ │ · GPU 推理                              │ │
│ └─────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────┐ │
│ │ 方法 B: 传统 CV 检测                     │ │
│ │ · Canny 边缘检测                        │ │
│ │ · 轮廓分析                              │ │
│ │ · 特征匹配                              │ │
│ └─────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────┐ │
│ │ 融合: NMS 去重 (IoU=0.5)               │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ Step 4: 目标跟踪                            │
│ · 为每个检测分配 UUID                        │
│ · 跨帧关联                                  │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ Step 5: 标注帧生成                          │
│ · 绘制检测框、标签                          │
│ · 显示置信度                                │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│ Step 6: 构件匹配与判定                      │
│ · CLIP 语义匹配                             │
│ · 更新唯一构件池                            │
└─────────────────────────────────────────────┘
```

---

## API 文档

### 1. 健康检查

```
GET /api/health
```

**Response:**
```json
{
  "status": "ok",
  "service": "construction-component-match"
}
```

### 2. 模型状态

```
GET /api/pipeline/model_status
```

**Response:**
```json
{
  "initialized": true,
  "grounding_dino_loaded": true,
  "clip_loaded": true,
  "active_methods": ["grounding_dino", "clip"],
  "loading": false,
  "load_progress": 100,
  "load_step": "所有模型加载完成！",
  "load_steps": ["初始化环境", "环境就绪", "GroundingDINO 就绪", "CLIP 就绪", "全部就绪"],
  "load_done": true,
  "load_error": null
}
```

### 3. 上传视频

```
POST /api/pipeline/upload_video
Content-Type: multipart/form-data
```

**Request:**
| Parameter | Type | Description |
|-----------|------|-------------|
| file | File | 视频文件 |

**Response:**
```json
{
  "path": "video_input/uuid.mp4",
  "filename": "video.mp4",
  "size": 12345678,
  "info": {
    "width": 1920,
    "height": 1080,
    "fps": 30.3,
    "frames": 305,
    "duration": 10.07
  }
}
```

### 4. 启动检测

```
POST /api/pipeline/start
Content-Type: application/json
```

**Request:**
```json
{
  "video_path": "video_input/uuid.mp4",
  "device": "cuda"
}
```

**Response:**
```json
{
  "pipeline_id": "uuid-string",
  "status": "started"
}
```

### 5. 停止检测

```
POST /api/pipeline/stop/{pipeline_id}
```

**Response:**
```json
{
  "status": "stopped"
}
```

### 6. 运行状态

```
GET /api/pipeline/status/{pipeline_id}
```

**Response:**
```json
{
  "status": "running",
  "current_frame": 50,
  "total_frames": 305,
  "progress": 16.4,
  "fps": 28.5,
  "detections_count": 2,
  "elapsed_time": "595ms",
  "last_step": "多方法构件检测"
}
```

### 7. 帧检测结果

```
GET /api/pipeline/frame_results/{pipeline_id}
```

**Response:**
```json
{
  "frame": 50,
  "detections": [
    {
      "track_id": "uuid-1",
      "bbox": [x, y, w, h],
      "confidence": 0.85,
      "detector": "grounding_dino",
      "class_name": "预制柱-标准柱"
    }
  ]
}
```

### 8. 视频流

```
GET /api/pipeline/stream_video?video_path={path}
```

**Response:** MP4 视频流

---

## 模型说明

### GroundingDINO

| 属性 | 值 |
|------|-----|
| 模型ID | IDEA-Research/grounding-dino-tiny |
| 类型 | 开放词汇目标检测 |
| 输入分辨率 | 自适应 (max_side=640) |
| 推理设备 | CUDA (GPU) |
| 下载大小 | ~700MB |
| 特点 | 可检测训练集中不存在的类别 |

### CLIP

| 属性 | 值 |
|------|-----|
| 模型ID | openai/clip-vit-base-patch32 |
| 类型 | 视觉-语言语义嵌入 |
| 输入分辨率 | 224x224 |
| 推理设备 | CUDA (GPU) |
| 下载大小 | ~400MB |
| 特点 | 计算图像与文本的语义相似度 |

### 传统 CV 检测

| 属性 | 值 |
|------|-----|
| 类型 | 基于图像处理 |
| 算法 | Canny + 轮廓分析 + 特征匹配 |
| 输入分辨率 | 原始分辨率 |
| 推理设备 | CPU |
| 特点 | 无需深度学习模型，补充检测 |

---

## 配置说明

### 主要配置项 (backend/config.py)

```python
# 基础路径
BASE_DIR = Path(__file__).resolve().parent.parent

# CORS 允许的源
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

# 静态文件目录
STATIC_DIR = BASE_DIR / "frontend"

# 上传目录
UPLOAD_DIR = BASE_DIR / "uploads"
```

### 检测配置 (backend/pipeline/processor.py)

```python
# 最大输入分辨率 (GPU 优化)
max_side = 640

# NMS IoU 阈值
iou_threshold = 0.5

# CLIP 匹配置信度阈值
confidence_threshold = 0.45
```

### 视频上传配置 (backend/routers/pipeline_router.py)

```python
# 支持的视频格式
ALLOWED_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv",
    ".flv", ".webm", ".m4v", ".mpg", ".mpeg"
}

# 最大文件大小 (500MB)
MAX_FILE_SIZE = 500 * 1024 * 1024
```

---

## 故障排查

### 模型加载失败

```bash
# 检查 GPU 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 检查模型缓存
ls ~/.cache/huggingface/hub/

# 手动下载模型
python -c "from huggingface_hub import snapshot_download; snapshot_download('IDEA-Research/grounding-dino-tiny')"
```

### CUDA 错误

```bash
# 确认 CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 更新 PyTorch
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu118
```

### 端口被占用

```bash
# Windows
netstat -ano | findstr :8000
taskkill /F /PID <PID>

# 使用脚本
python stop.py
```

### 显存不足

修改 `model_detector.py` 中的 `max_side` 参数：

```python
# 降低输入分辨率
max_side = 480  # 默认 640
```

---

## 性能优化

### GPU 内存管理

```python
# 每次推理后清理 GPU 内存
del inputs, outputs, results, pil_image
torch.cuda.synchronize()
torch.cuda.empty_cache()
```

### 输入分辨率优化

```python
# 原始分辨率 -> 640px 以内
max_side = 640
if max(h, w) > max_side:
    scale = max_side / max(h, w)
    frame_small = cv2.resize(frame, (int(w*scale), int(h*scale)))
```

### 预期性能

| 指标 | 值 |
|------|-----|
| GroundingDINO 推理 | 100-200ms/帧 |
| 传统 CV 检测 | 10-20ms/帧 |
| 总检测耗时 | 200-600ms/帧 |
| GPU 显存占用 | ~3GB |

---

## 更新日志

### v1.0.0 (2026-08-27)

- 精简项目结构，只保留核心检测功能
- 实现模型加载进度实时反馈
- 优化检测流程：多方法顺序执行
- 添加 GPU 内存清理，防止内存泄漏
- 添加 start.py / stop.py 启动停止脚本
- 页面布局：视频检测与检测详情各占 50%

---

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请提交 Issue。