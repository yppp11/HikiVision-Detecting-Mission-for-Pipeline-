# Pipe Edge Detection & Measurement

基于工业相机的管道上下边缘检测、亚像素细化与形变测量系统。

---

## 项目简介

本项目针对灰度 ROI 图像中的管道（或类似条带结构），实现了从图像采集到边缘精确重建、形变量化的完整流水线：

1. **图像采集** — 支持海康威视工业相机（GigE / USB），可选灰度（Mono8）或彩色（BayerBG8/RG8）模式
2. **ROI 提取** — 阈值 + 形态学闭运算，自动裁剪出管道感兴趣区域
3. **边缘分析** — 梯度热力图、LoG、相位一致性三种特征可视化
4. **完整检测流水线** — 梯度粗定位 → 亚像素细化 → 离群点清洗 → 多项式基准线 + 样条精细拟合 → 厚度/缺陷测量

---

## 文件结构

```
collect/
├── collect_image/
│   ├── collect_mono.py     # 海康相机采集：Mono8 灰度模式
│   └── collect_rgb.py      # 海康相机采集：Bayer 彩色模式
├── include/                # 海康 MVS Python SDK 封装
│   ├── MvCameraControl_class.py
│   ├── MvErrorDefine_const.py
│   ├── CameraParams_header.py
│   ├── CameraParams_const.py
│   └── PixelType_header.py
├── create_roi.py           # 从全幅图自动裁剪管道 ROI
├── a_panduan.py            # 可视化：梯度粗定位 + 亚像素散点 + 热力图（4图）
├── b_panduan.py            # 可视化：Gradient / LoG / Phase Congruency 热力图对比
├── h_test.py               # 完整 Pipeline：检测 + 拟合 + 测量 + 交互 GUI
├── note/
│   ├── 版本修订说明.txt
│   └── opencv_learning.markdown
└── old_vision/             # 历史版本（阈值划分、梯度粗分等早期算法）
```

---

## 依赖安装

```bash
pip install opencv-python numpy matplotlib scipy
```

海康相机采集额外需要安装 [MVS 工业相机客户端](https://www.hikrobotics.com/)（含 Python SDK）。

---

## 快速上手

### 1. 采集图像

```bash
# 灰度模式（推荐，带宽低、处理快）
python collect_image/collect_mono.py

# 彩色模式（自动尝试 BayerBG8 / BayerRG8）
python collect_image/collect_rgb.py
```

主要参数（在脚本顶部直接修改）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CAMERA_INDEX` | 0 | 相机索引 |
| `CAPTURE_INTERVAL` | 0.5 s | 采集间隔 |
| `DURATION_SECONDS` | 30 s | 采集总时长 |
| `SAVE_ROOT` | `data_mono` / `data` | 图像保存根目录 |

图像保存路径格式：`{SAVE_ROOT}/single_cam_YYYYMMDD_HHMMSS/images/img_000000.jpg`

---

### 2. 提取 ROI

```bash
python create_roi.py
```

输入：`test.png`（全幅图）
输出：`roi_pipe.png`（裁剪后 ROI）、`roi_on_full.png`（标注框可视化）

关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `target_thresh` | 220 | 固定阈值（0~255，管子亮/背景暗时建议 150~220） |
| `margin_x` | 400 px | 水平扩边 |
| `margin_y` | 50 px | 垂直扩边 |

---

### 3. 边缘特征可视化

```bash
# 4 图：灰度热力图 / 边缘散点 / 垂直梯度 / 梯度幅值
python a_panduan.py

# 3 图热力图对比：一阶梯度 / LoG / 相位一致性
python b_panduan.py
```

两个脚本读取的图像路径均为 `roi_pipe.png`（可在文件顶部修改 `IMAGE_PATH`）。

---

### 4. 完整 Pipeline（检测 + 拟合 + 测量）

```bash
python h_test.py --input roi_pipe.png --save result.png

# 不启动交互窗口
python h_test.py --input roi_pipe.png --no-gui

# 调整日志等级
python h_test.py --log DEBUG
```

**Pipeline 步骤：**

```
[Step 1] Sobel(Y) 梯度粗定位
         ↓ 按列扫描，上边找正梯度峰，下边找负梯度峰（含 look-ahead 抗虚影）
[Step 2] 亚像素细化
         ↓ 局部灰度剖面线性插值，定位到半像素精度
[Step 3] 离群点清洗
         ↓ 滑动中值基准 + 阈值剔除突变点
[Step 4] 拟合
         ↓ 多项式 + sigma-clipping（基准线）
         ↓ UnivariateSpline（精细轮廓）
[Step 5] 静态结果图保存（OpenCV 渲染）
[Step 6] 厚度与缺陷统计（min / mean / p95 / max）
[Step 7] Matplotlib 交互 GUI（鼠标悬停查看局部厚度与缺陷量）
```

**主要可调参数（在 `h_test.py` 中 `dataclass` 配置区修改）：**

| 配置类 | 关键参数 | 说明 |
|--------|----------|------|
| `EdgeDetectConfig` | `search_ratio_top/bottom` | 上/下边缘搜索范围（相对图像高度） |
| `EdgeDetectConfig` | `rel_thresh_top/bottom` | 梯度阈值相对比例 |
| `SubpixelRefineConfig` | `window_radius` | 亚像素细化半窗口（px） |
| `SubpixelRefineConfig` | `intensity_ratio` | 插值目标灰度比例（0.5 = 半高） |
| `CleanConfig` | `median_window` | 滑动中值窗口大小 |
| `CleanConfig` | `diff_thresh` | 最大允许突变偏差（px） |
| `FitConfig` | `spline_s_factor` | 样条平滑系数（越小越贴点） |
| `PlotConfig` | `px_to_um` | 像素→微米换算比（默认 15.47） |

---

## 控制台输出示例

```
[10:23:01] INFO - 图像尺寸：480x2048 (H×W)
[10:23:01] INFO - 粗定位点数：top=2040 (99.6%), bottom=2035 (99.3%)
[10:23:02] INFO - 细化后点数：top=2040, bottom=2035
[10:23:02] INFO - 清洗后点数：top=2031 (剔除 9), bottom=2028 (剔除 7)
[10:23:02] INFO - 拟合残差标准差(spline)：top=0.3142 px, bottom=0.2876 px
[10:23:02] INFO - Thickness(px)：min=180.3, mean=183.7, p95=186.1, max=191.4
[10:23:02] INFO - Thickness(µm)：min=2789.3, mean=2841.4, p95=2879.9, max=2960.0
```

---

## 算法说明

### 梯度粗定位

对每一列：
- **上边缘**：从上往下扫描，找到第一个超过 `max(min_abs_thresh, max_grad * rel_thresh)` 的正梯度点
- **下边缘**：从下往上扫描，在找到负梯度触发点后，在其上方 `look_ahead_win` 范围内再取局部最小梯度，消除"虚影"引起的偏差

### 亚像素细化

以粗定位点为中心，取 `±window_radius` 的局部灰度剖面，在背景-前景过渡的 `intensity_ratio` 处做线性插值，将精度提升到亚像素级别。

### 形变测量

- **厚度**：每列 `|y_bottom_spline - y_top_spline|`
- **缺陷**：样条精细轮廓与多项式基准线之差的绝对值，反映局部凹陷/凸起量

---

## 开发计划

- [ ] 去掉对管路水平的强假设，改用全局路径规划做边缘追踪（提升鲁棒性）
- [ ] 参数自动化调整，减少手动调参负担
- [ ] 引入高级算法适应更加复杂的环境
---