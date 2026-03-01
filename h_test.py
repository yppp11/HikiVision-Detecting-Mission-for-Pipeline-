#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
管道（或类似条带）上下边缘重建 + 形变/厚度测量

功能概览
- 读取灰度 ROI 图
- Sobel(Y) 梯度粗定位上下边缘
- 垂直方向 1D 灰度剖面做亚像素细化（按阈值线性插值）
- 中值滤波 + 阈值剔除离群点
- 拟合：
  - 基准线：多项式 + 迭代 sigma-clipping（近似 RANSAC 思路）
  - 精细轮廓：UnivariateSpline（较小的平滑因子，尽量贴合凹陷/形变）
- 输出：
  - 静态结果图（OpenCV 绘制）
  - 可交互图（Matplotlib 鼠标悬停显示厚度/缺陷）

依赖
- opencv-python
- numpy
- matplotlib
- scipy

用法（默认参数与原脚本一致）
python f_test_refactored.py --input roi_pipe.png --save final_fit_result.png

说明
- 参数集中在 dataclass 里，便于你后续扩展/调参/序列化保存。
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


# ==============================================================================
# 配置区（可调参数与含义）
# ==============================================================================

@dataclass(frozen=True)
class IOConfig:
    """
    输入/输出与运行方式相关参数

    参数说明
    - input_path:
        输入灰度图路径（建议为 ROI 后的单通道图片）
    - static_vis_save_path:
        保存静态结果图（OpenCV 渲染：点 + 上下边缘曲线）
    - enable_gui:
        是否启动 Matplotlib 交互窗口（鼠标移动显示测量值）
    """
    input_path: str = "roi_pipe.png"
    static_vis_save_path: str = "final_fit_result.png"
    enable_gui: bool = True


@dataclass(frozen=True)
class EdgeDetectConfig:
    """
    Sobel 梯度粗定位参数

    参数说明
    - median_ksize:
        中值滤波核大小（必须为奇数；越大越平滑，但会模糊边缘）
    - search_ratio_top:
        在图像顶部查找上边缘的搜索范围比例（0~1）。
        例如 0.55 表示只在 [0, 0.55*H) 范围内找上边缘。
    - rel_thresh_top:
        上边缘阈值的相对比例：阈值 = max(min_abs_thresh, max(梯度)*rel_thresh_top)
        该值越大越“严格”，点会变少但更干净。
    - search_ratio_bottom:
        在图像底部查找下边缘的搜索范围比例（0~1）。
        例如 0.45 表示只在 (H*(1-0.45), H] 范围内找下边缘。
    - rel_thresh_bottom:
        下边缘阈值相对比例：阈值 = min(-min_abs_thresh, min(梯度)*rel_thresh_bottom)
        下边缘通常表现为负梯度。
    - look_ahead_win:
        找到底部边缘后，在其上方 look_ahead_win 范围内再找局部最小值，
        用于抵消“第一次穿阈值点”带来的偏差（更接近真实边缘极值）。
    - min_abs_thresh:
        绝对阈值下限（梯度幅值小于它时，不接受为边缘）
    """
    median_ksize: int = 5
    search_ratio_top: float = 0.55
    rel_thresh_top: float = 0.15
    search_ratio_bottom: float = 0.45
    rel_thresh_bottom: float = 0.55
    look_ahead_win: int = 5
    min_abs_thresh: float = 4.0


@dataclass(frozen=True)
class SubpixelRefineConfig:
    """
    亚像素细化参数（沿列方向的 1D 灰度剖面）

    参数说明
    - window_radius:
        以粗定位 y0 为中心，向上/向下截取的半窗口大小 r。
        实际剖面长度为 2*r+1。r 越大，抗噪更强，但更可能跨越非目标结构。
    - intensity_ratio:
        目标灰度所在的“过渡比例”（0~1）：
        target = bg + (fg-bg)*ratio
        一般 0.5 对应“半高”位置；可用于抵消曝光/对比度差异。
    """
    window_radius: int = 10
    intensity_ratio: float = 0.5


@dataclass(frozen=True)
class CleanConfig:
    """
    点清洗参数（去除离群点）

    参数说明
    - median_window:
        沿 x 方向对 y 做滑动中值滤波的窗口长度（必须为奇数）
    - diff_thresh:
        若 |y - y_median| >= diff_thresh，则认为是离群点并剔除（单位：像素）
    """
    median_window: int = 199
    diff_thresh: float = 2


@dataclass(frozen=True)
class FitConfig:
    """
    拟合参数

    参数说明
    - poly_degree:
        基准线多项式阶数（建议 1~3）
    - sigma_thresh:
        sigma-clipping 的门限倍数；越大越“宽松”（保留更多点）
    - max_iter:
        sigma-clipping 最大迭代次数
    - spline_s_factor:
        精细样条平滑因子 s 的系数：s = len(points) * spline_s_factor
        越小越贴点（更能捕捉凹陷，但对噪声更敏感）
    - min_points:
        拟合所需的最小点数（点太少时直接失败）
    """
    poly_degree: int = 3
    sigma_thresh: float = 3.0
    max_iter: int = 5
    spline_s_factor: float = 0.1
    min_points: int = 10


@dataclass(frozen=True)
class PlotConfig:
    """
    可视化参数

    参数说明
    - mpl_style:
        Matplotlib 风格（如 'dark_background'）
    - fig_size:
        Matplotlib 图像尺寸 (W, H)
    - line_thickness:
        OpenCV 静态图绘制边缘曲线粗细（像素）
    - alpha_fill:
        管道填充区域透明度
    - alpha_center_line:
        中心线透明度
    - grid_alpha:
        网格透明度
    - color_*:
        各元素颜色（matplotlib 接受的颜色格式）
    - text_title/xlabel/ylabel:
        标题与坐标轴文字
    """
    mpl_style: str = "dark_background"
    fig_size: Tuple[int, int] = (12, 7)

    line_thickness: int = 2
    alpha_fill: float = 0.3
    alpha_center_line: float = 0.7
    grid_alpha: float = 0.4

    # 单位换算：1 px = px_to_um 微米。仅影响 Matplotlib 交互 GUI 的坐标与显示（静态 OpenCV 图仍是像素坐标）。
    px_to_um: float = 15.47
    length_unit: str = "µm"

    color_pipe_fill: str = "#00CED1"
    color_edge_top: str = "#FF6347"
    color_edge_bottom: str = "#1E90FF"
    color_center_line: str = "#32CD32"
    color_interact_point: str = "yellow"
    color_text_box_bg: str = "#333333"

    center_line_style: str = "--"

    text_title: str = "Pipe Deformation Reconstruction & Measurement"
    text_xlabel: str = "X Position (µm)"
    text_ylabel: str = "Y Position (µm)"


@dataclass(frozen=True)
class PipelineConfig:
    """把所有可调参数集中管理，便于后续扩展（如 YAML/JSON 配置化）。"""
    io: IOConfig = IOConfig()
    edge: EdgeDetectConfig = EdgeDetectConfig()
    refine: SubpixelRefineConfig = SubpixelRefineConfig()
    clean: CleanConfig = CleanConfig()
    fit: FitConfig = FitConfig()
    plot: PlotConfig = PlotConfig()


# ==============================================================================
# 数据结构
# ==============================================================================

@dataclass
class EdgePoints:
    """上下边缘点集（x 与 y 一一对应）。"""
    x_top: np.ndarray
    y_top: np.ndarray
    x_bottom: np.ndarray
    y_bottom: np.ndarray


@dataclass
class FitResult:
    """单条边缘的拟合结果。"""
    spline: Optional[UnivariateSpline]
    baseline_poly: Optional[np.poly1d]
    inlier_mask: np.ndarray
    residual_std: float
    x_min: float
    x_max: float


@dataclass
class MeasurementSummary:
    """测量统计摘要（用于控制台输出）。"""
    thickness_min: float
    thickness_mean: float
    thickness_max: float
    thickness_p95: float
    top_defect_max: float
    top_defect_p95: float
    bottom_defect_max: float
    bottom_defect_p95: float


# ==============================================================================
# 工具函数
# ==============================================================================



def build_logger(level: str = "INFO") -> logging.Logger:
    """创建统一的控制台 logger。"""
    logger = logging.getLogger("pipe_reconstruct")
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logger.level)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _ensure_odd(name: str, value: int, min_value: int = 3) -> int:
    """确保 OpenCV/滑窗参数为奇数。"""
    if value < min_value:
        raise ValueError(f"{name} 不能小于 {min_value}，当前={value}")
    if value % 2 == 0:
        return value + 1
    return value


def load_grayscale_image(path: str) -> np.ndarray:
    """读取灰度图；失败则抛异常。"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图片：{path}")
    return img


# ==============================================================================
# 核心算法
# ==============================================================================

def find_edges_gradient_coarse(
    img: np.ndarray,
    cfg: EdgeDetectConfig,
) -> EdgePoints:
    """
    使用 Sobel(Y) 梯度做“按列扫描”的粗边缘定位。

    策略（与原逻辑一致）：
    - 上边缘：在顶部区域找到第一个超过阈值的正梯度点
    - 下边缘：在底部区域从下往上找到第一个低于阈值的负梯度点，
      并在其上方 look_ahead_win 范围内再取局部最小值

    返回
    - EdgePoints: x_top/y_top/x_bottom/y_bottom，长度可能 < 图像宽度（某些列可能找不到边缘）
    """
    h, w = img.shape[:2]
    median_ksize = _ensure_odd("edge.median_ksize", cfg.median_ksize, min_value=3)
    img_smooth = cv2.medianBlur(img, median_ksize)

    grad_y = cv2.Sobel(img_smooth, cv2.CV_64F, 0, 1, ksize=3)

    x_top: list[int] = []
    y_top: list[int] = []
    x_bottom: list[int] = []
    y_bottom: list[int] = []

    lim_top = int(h * float(cfg.search_ratio_top))
    start_bottom = int(h * (1.0 - float(cfg.search_ratio_bottom)))

    lim_top = np.clip(lim_top, 1, h)
    start_bottom = np.clip(start_bottom, 0, h - 1)

    for x in range(w):
        col = grad_y[:, x]

        # --- 上边缘 ---
        reg_top = col[:lim_top]
        max_top = float(np.max(reg_top)) if reg_top.size else 0.0
        th_top = max(float(cfg.min_abs_thresh), max_top * float(cfg.rel_thresh_top))

        found_top = False
        for y in range(2, lim_top):
            if col[y] > th_top:
                x_top.append(x)
                y_top.append(y)
                found_top = True
                break
        # 找不到就跳过该列（不强行填充）

        # --- 下边缘 ---
        reg_bottom = col[start_bottom:]
        min_bottom = float(np.min(reg_bottom)) if reg_bottom.size else 0.0
        # 下边缘是负梯度：阈值应为负数，且至少小于 -min_abs_thresh
        th_bottom = min(-float(cfg.min_abs_thresh), min_bottom * float(cfg.rel_thresh_bottom))

        found_bottom = False
        for y in range(h - 3, start_bottom, -1):
            if col[y] < th_bottom:
                peek_start = max(0, y - int(cfg.look_ahead_win))
                local = col[peek_start : y + 1]
                if local.size > 0:
                    real_y = int(peek_start + np.argmin(local))
                else:
                    real_y = int(y)
                x_bottom.append(x)
                y_bottom.append(real_y)
                found_bottom = True
                break

    return EdgePoints(
        x_top=np.asarray(x_top, dtype=np.int32),
        y_top=np.asarray(y_top, dtype=np.int32),
        x_bottom=np.asarray(x_bottom, dtype=np.int32),
        y_bottom=np.asarray(y_bottom, dtype=np.int32),
    )


def refine_edge_subpixel(
    img: np.ndarray,
    x_list: np.ndarray,
    y_list: np.ndarray,
    *,
    is_top_edge: bool,
    cfg: SubpixelRefineConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对粗定位的边缘点做亚像素细化：沿同一列提取局部灰度剖面，在目标灰度处做线性插值。

    参数
    - img:
        灰度图
    - x_list, y_list:
        粗边缘点坐标（同长度）
    - is_top_edge:
        True=上边缘，False=下边缘
        用于决定背景/前景灰度取样方向
    - cfg.window_radius:
        半窗口 r，剖面长度 2r+1
    - cfg.intensity_ratio:
        目标过渡比例（0~1）

    返回
    - (x_kept, y_refined): 仅保留有效范围内的点
    """
    if x_list.size == 0:
        return x_list.astype(np.int32), y_list.astype(np.float32)

    h, w = img.shape[:2]
    r = int(cfg.window_radius)
    if r <= 0:
        return x_list.astype(np.int32), y_list.astype(np.float32)

    ratio = float(cfg.intensity_ratio)
    ratio = float(np.clip(ratio, 0.0, 1.0))

    y_refined: list[float] = []
    x_kept: list[int] = []

    for x0, y0 in zip(x_list.astype(int), y_list.astype(int)):
        if x0 < 0 or x0 >= w:
            continue
        if y0 - r < 0 or y0 + r >= h:
            continue

        prof = img[y0 - r : y0 + r + 1, x0].astype(np.float32)

        # 背景/前景取样：与原逻辑一致（头2个 vs 尾2个）
        if is_top_edge:
            val_bg = float(np.mean(prof[:2]))
            val_fg = float(np.mean(prof[-2:]))
        else:
            val_bg = float(np.mean(prof[-2:]))
            val_fg = float(np.mean(prof[:2]))

        target = val_bg + (val_fg - val_bg) * ratio

        found = False
        for j in range(len(prof) - 1):
            v1, v2 = float(prof[j]), float(prof[j + 1])
            if (v1 <= target <= v2) or (v2 <= target <= v1):
                denom = (v2 - v1)
                offset = (target - v1) / denom if abs(denom) > 1e-6 else 0.5
                y_refined.append((y0 - r) + j + float(offset))
                x_kept.append(int(x0))
                found = True
                break

        if not found:
            # 找不到交点时回退到整数 y0，保证长度一致
            y_refined.append(float(y0))
            x_kept.append(int(x0))

    return np.asarray(x_kept, dtype=np.int32), np.asarray(y_refined, dtype=np.float32)


def clean_edge_points(
    x: np.ndarray,
    y: np.ndarray,
    cfg: CleanConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    通过“滑动中值”构造基准曲线，再按阈值剔除离群点。

    返回
    - x_clean, y_clean, mask
      mask 为原数组长度的布尔数组，True 表示保留
    """
    if x.size == 0:
        return x, y, np.zeros_like(x, dtype=bool)

    win = _ensure_odd("clean.median_window", int(cfg.median_window), min_value=3)
    if x.size < win:
        mask = np.ones_like(x, dtype=bool)
        return x, y, mask

    # 为了滑窗稳定：按 x 排序后再处理
    sort_idx = np.argsort(x)
    xs = x[sort_idx]
    ys = y[sort_idx].astype(np.float32)

    pad = win // 2
    padded = np.pad(ys, (pad, pad), mode="edge")
    y_med = np.array([np.median(padded[i : i + win]) for i in range(ys.size)], dtype=np.float32)

    mask_sorted = np.abs(ys - y_med) < float(cfg.diff_thresh)

    # 还原回原顺序的 mask
    mask = np.zeros_like(mask_sorted, dtype=bool)
    mask[sort_idx] = mask_sorted

    return x[mask], y[mask], mask


def fit_edge(
    x: np.ndarray,
    y: np.ndarray,
    cfg: FitConfig,
) -> FitResult:
    """
    拟合单条边缘：
    - baseline_poly: 多项式 + 迭代 sigma-clipping（用于“理想基准”）
    - spline: UnivariateSpline（用于“精细真实轮廓”）

    返回 FitResult；若点数不足，则 spline/baseline_poly 可能为 None。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.size < int(cfg.min_points):
        return FitResult(
            spline=None,
            baseline_poly=None,
            inlier_mask=np.zeros_like(x, dtype=bool),
            residual_std=float("nan"),
            x_min=float("nan"),
            x_max=float("nan"),
        )

    # --- 阶段 A：多项式基准线（sigma-clipping） ---
    mask = np.ones_like(x, dtype=bool)
    baseline_poly: Optional[np.poly1d] = None

    for _ in range(int(cfg.max_iter)):
        if mask.sum() < int(cfg.poly_degree) + 5:
            break
        try:
            coef = np.polyfit(x[mask], y[mask], int(cfg.poly_degree))
            baseline_poly = np.poly1d(coef)
        except Exception:
            baseline_poly = None
            break

        resid = y - baseline_poly(x)
        std = float(np.std(resid[mask]))
        if std < 1e-9:
            break

        new_mask = np.abs(resid) < float(cfg.sigma_thresh) * std
        if int(new_mask.sum()) == int(mask.sum()):
            break
        mask = new_mask

    # --- 阶段 B：精细样条 ---
    sort_idx = np.argsort(x)
    sx, sy = x[sort_idx], y[sort_idx]
    sx, unique_idx = np.unique(sx, return_index=True)
    sy = sy[unique_idx]

    s = float(len(sx)) * float(cfg.spline_s_factor)
    spline = UnivariateSpline(sx, sy, s=s)

    residual_std = float(np.std(sy - spline(sx)))
    return FitResult(
        spline=spline,
        baseline_poly=baseline_poly,
        inlier_mask=mask,
        residual_std=residual_std,
        x_min=float(sx.min()),
        x_max=float(sx.max()),
    )


def _eval_spline_clipped(spline: UnivariateSpline, x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """
    为避免 spline 在外推区间发散，对 x 做裁剪后再评估。
    """
    xc = np.clip(x, x_min, x_max)
    return spline(xc)


# ==============================================================================
# 可视化与测量
# ==============================================================================

def save_static_visual_result(
    img: np.ndarray,
    pts: EdgePoints,
    fit_top: FitResult,
    fit_bottom: FitResult,
    save_path: str,
    thickness: int,
) -> None:
    """
    保存静态结果图：
    - 绿色点：清洗后有效点
    - 红线：上边缘 spline
    - 蓝线：下边缘 spline
    """
    h, w = img.shape[:2]
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for x, y in zip(pts.x_top, pts.y_top):
        cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 0), -1)
    for x, y in zip(pts.x_bottom, pts.y_bottom):
        cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 0), -1)

    x_eval = np.arange(w, dtype=np.float32)

    if fit_top.spline is not None:
        y_fit = _eval_spline_clipped(fit_top.spline, x_eval, fit_top.x_min, fit_top.x_max)
        pts_poly = np.column_stack((x_eval, y_fit)).astype(np.int32)
        cv2.polylines(vis, [pts_poly], False, (0, 0, 255), int(thickness))

    if fit_bottom.spline is not None:
        y_fit = _eval_spline_clipped(fit_bottom.spline, x_eval, fit_bottom.x_min, fit_bottom.x_max)
        pts_poly = np.column_stack((x_eval, y_fit)).astype(np.int32)
        cv2.polylines(vis, [pts_poly], False, (255, 0, 0), int(thickness))

    cv2.imwrite(save_path, vis)


def compute_measurements(
    x_fit: np.ndarray,
    fit_top: FitResult,
    fit_bottom: FitResult,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算厚度与缺陷（相对基准线偏离）。

    返回
    - thickness: |y_bottom - y_top|
    - top_defect: |y_top - y_top_baseline|
    - bottom_defect: |y_bottom - y_bottom_baseline|
    """
    if fit_top.spline is None or fit_bottom.spline is None:
        raise ValueError("spline 拟合失败，无法测量。")
    if fit_top.baseline_poly is None or fit_bottom.baseline_poly is None:
        raise ValueError("baseline_poly 拟合失败，无法测量。")

    y_top = _eval_spline_clipped(fit_top.spline, x_fit, fit_top.x_min, fit_top.x_max)
    y_bottom = _eval_spline_clipped(fit_bottom.spline, x_fit, fit_bottom.x_min, fit_bottom.x_max)

    y_top_base = fit_top.baseline_poly(x_fit)
    y_bottom_base = fit_bottom.baseline_poly(x_fit)

    thickness = np.abs(y_bottom - y_top).astype(np.float64)
    top_defect = np.abs(y_top - y_top_base).astype(np.float64)
    bottom_defect = np.abs(y_bottom - y_bottom_base).astype(np.float64)

    return thickness, top_defect, bottom_defect


def summarize_measurements(thickness: np.ndarray, top_def: np.ndarray, bot_def: np.ndarray) -> MeasurementSummary:
    """把测量数组压缩成控制台友好的统计摘要。"""
    def p(a: np.ndarray, q: float) -> float:
        return float(np.percentile(a, q))

    return MeasurementSummary(
        thickness_min=float(np.min(thickness)),
        thickness_mean=float(np.mean(thickness)),
        thickness_max=float(np.max(thickness)),
        thickness_p95=p(thickness, 95),
        top_defect_max=float(np.max(top_def)),
        top_defect_p95=p(top_def, 95),
        bottom_defect_max=float(np.max(bot_def)),
        bottom_defect_p95=p(bot_def, 95),
    )


def plot_interactive(
    roi_shape: Tuple[int, int],
    x_fit: np.ndarray,
    fit_top: FitResult,
    fit_bottom: FitResult,
    plot_cfg: PlotConfig,
) -> None:
    """
    Matplotlib 交互可视化：
    - 填充管道区域
    - 鼠标悬停显示当前 x 的厚度、上下边缘缺陷
    """
    h_roi, w_roi = roi_shape[:2]

    if fit_top.spline is None or fit_bottom.spline is None:
        raise ValueError("spline 拟合失败，无法绘图。")
    if fit_top.baseline_poly is None or fit_bottom.baseline_poly is None:
        raise ValueError("baseline_poly 拟合失败，无法绘图。")

    y_top = _eval_spline_clipped(fit_top.spline, x_fit, fit_top.x_min, fit_top.x_max)
    y_bottom = _eval_spline_clipped(fit_bottom.spline, x_fit, fit_bottom.x_min, fit_bottom.x_max)
    y_center = (y_top + y_bottom) / 2.0

    # --- GUI 单位换算：把坐标从 px 映射到 µm（或 plot_cfg.length_unit） ---
    scale = float(getattr(plot_cfg, "px_to_um", 1.0))
    x_plot = x_fit * scale
    y_top_plot = y_top * scale
    y_bottom_plot = y_bottom * scale
    y_center_plot = y_center * scale
    h_plot = float(h_roi) * scale
    w_plot = float(w_roi) * scale

    y_top_base = fit_top.baseline_poly(x_fit)
    y_bottom_base = fit_bottom.baseline_poly(x_fit)

    plt.style.use(plot_cfg.mpl_style)
    fig, ax = plt.subplots(figsize=plot_cfg.fig_size)

    ax.set_xlim(0, w_plot)
    ax.set_ylim(h_plot, 0)
    ax.set_title(plot_cfg.text_title, color="white", fontsize=14)
    ax.set_xlabel(plot_cfg.text_xlabel)
    ax.set_ylabel(plot_cfg.text_ylabel)

    ax.fill_between(x_plot, y_top_plot, y_bottom_plot, color=plot_cfg.color_pipe_fill, alpha=plot_cfg.alpha_fill, label="Pipe Body")
    ax.plot(x_plot, y_top_plot, color=plot_cfg.color_edge_top, linewidth=1.5, label="Top Edge (Refined)")
    ax.plot(x_plot, y_bottom_plot, color=plot_cfg.color_edge_bottom, linewidth=1.5, label="Bottom Edge (Refined)")
    ax.plot(
        x_plot,
        y_center_plot,
        color=plot_cfg.color_center_line,
        linestyle=plot_cfg.center_line_style,
        linewidth=1.0,
        alpha=plot_cfg.alpha_center_line,
        label="Center Line",
    )

    ax.legend(loc="upper right", facecolor=plot_cfg.color_text_box_bg, edgecolor="none")
    ax.grid(True, linestyle=":", alpha=plot_cfg.grid_alpha)

    # 交互指示器
    vline = ax.axvline(x=0, color=plot_cfg.color_interact_point, linestyle="-", linewidth=1, alpha=0.0)
    p_top, = ax.plot([], [], "o", color=plot_cfg.color_interact_point, markersize=5)
    p_bot, = ax.plot([], [], "o", color=plot_cfg.color_interact_point, markersize=5)

    text_annot = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        color="white",
        bbox=dict(facecolor=plot_cfg.color_text_box_bg, alpha=0.85, edgecolor="none", boxstyle="round,pad=0.5"),
        va="top",
    )

    def on_mouse_move(event):
        if (not event.inaxes) or (event.xdata is None):
            return
        idx = int(np.clip(round(event.xdata / scale), 0, len(x_fit) - 1))
        curr_x_px = float(x_fit[idx])
        curr_x_um = float(x_plot[idx])

        yt_um = float(y_top_plot[idx])
        yb_um = float(y_bottom_plot[idx])

        defect_t_um = abs(yt_um - float(y_top_base[idx]) * scale)
        defect_b_um = abs(yb_um - float(y_bottom_base[idx]) * scale)
        thick_um = abs(yb_um - yt_um)

        vline.set_xdata([curr_x_um])
        vline.set_alpha(0.85)
        p_top.set_data([curr_x_um], [yt_um])
        p_bot.set_data([curr_x_um], [yb_um])

        text_annot.set_text(
            f"""X: {curr_x_um:.1f} {plot_cfg.length_unit} ({curr_x_px:.0f} px)
Thickness: {thick_um:.3f} {plot_cfg.length_unit}
Top Defect: {defect_t_um:.3f} {plot_cfg.length_unit}
Bottom Defect: {defect_b_um:.3f} {plot_cfg.length_unit}"""
        )
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
    plt.show()


# ==============================================================================
# Pipeline 入口
# ==============================================================================

def run_pipeline(cfg: PipelineConfig, logger: logging.Logger) -> None:
    """按步骤运行整条 pipeline，并输出尽量完整的控制台信息。"""
    logger.info("开始：管道边缘重建与测量")
    logger.info("输入图像：%s", cfg.io.input_path)

    # 1) 读图
    img = load_grayscale_image(cfg.io.input_path)
    h, w = img.shape[:2]
    logger.info("图像尺寸：%dx%d (H×W)", h, w)

    # 2) 粗定位
    logger.info("步骤1：Sobel 梯度粗定位边缘点 ...")
    pts_coarse = find_edges_gradient_coarse(img, cfg.edge)
    logger.info(
        "粗定位点数：top=%d (覆盖 %.1f%%), bottom=%d (覆盖 %.1f%%)",
        pts_coarse.x_top.size,
        100.0 * pts_coarse.x_top.size / max(1, w),
        pts_coarse.x_bottom.size,
        100.0 * pts_coarse.x_bottom.size / max(1, w),
    )

    # 3) 亚像素细化
    logger.info("步骤2：亚像素细化 ...")
    xt, yt = refine_edge_subpixel(img, pts_coarse.x_top, pts_coarse.y_top, is_top_edge=True, cfg=cfg.refine)
    xb, yb = refine_edge_subpixel(img, pts_coarse.x_bottom, pts_coarse.y_bottom, is_top_edge=False, cfg=cfg.refine)
    logger.info("细化后点数：top=%d, bottom=%d", xt.size, xb.size)

    # 4) 清洗
    logger.info("步骤3：离群点清洗 ...")
    xt2, yt2, mask_t = clean_edge_points(xt, yt, cfg.clean)
    xb2, yb2, mask_b = clean_edge_points(xb, yb, cfg.clean)
    logger.info(
        "清洗后点数：top=%d (剔除 %d), bottom=%d (剔除 %d)",
        xt2.size,
        int(xt.size - xt2.size),
        xb2.size,
        int(xb.size - xb2.size),
    )

    pts_clean = EdgePoints(x_top=xt2, y_top=yt2, x_bottom=xb2, y_bottom=yb2)

    # 5) 拟合
    logger.info("步骤4：拟合（基准线 + 精细样条） ...")
    fit_top = fit_edge(pts_clean.x_top, pts_clean.y_top, cfg.fit)
    fit_bottom = fit_edge(pts_clean.x_bottom, pts_clean.y_bottom, cfg.fit)

    if fit_top.spline is None or fit_bottom.spline is None:
        logger.error("拟合失败：有效点太少或数据异常。请检查阈值/ROI/曝光。")
        return
    if fit_top.baseline_poly is None or fit_bottom.baseline_poly is None:
        logger.error("拟合失败：baseline_poly 计算失败（可能点分布太差或阶数过高）。")
        return

    logger.info(
        "拟合残差标准差（spline）：top=%.4f px, bottom=%.4f px",
        fit_top.residual_std,
        fit_bottom.residual_std,
    )
    logger.info(
        "基准线内点比例：top=%.1f%%, bottom=%.1f%%",
        100.0 * float(np.mean(fit_top.inlier_mask)) if fit_top.inlier_mask.size else 0.0,
        100.0 * float(np.mean(fit_bottom.inlier_mask)) if fit_bottom.inlier_mask.size else 0.0,
    )

    # 6) 静态图保存
    logger.info("步骤5：保存静态结果图 ...")
    save_static_visual_result(
        img=img,
        pts=pts_clean,
        fit_top=fit_top,
        fit_bottom=fit_bottom,
        save_path=cfg.io.static_vis_save_path,
        thickness=cfg.plot.line_thickness,
    )
    logger.info("静态结果图已保存：%s", cfg.io.static_vis_save_path)

    # 7) 测量统计
    logger.info("步骤6：计算厚度与缺陷统计 ...")
    x_fit = np.arange(w, dtype=np.float64)
    thickness, top_def, bot_def = compute_measurements(x_fit, fit_top, fit_bottom)
    summary = summarize_measurements(thickness, top_def, bot_def)

    logger.info(
        "Thickness(px)：min=%.3f, mean=%.3f, p95=%.3f, max=%.3f",
        summary.thickness_min,
        summary.thickness_mean,
        summary.thickness_p95,
        summary.thickness_max,
    )

    # 同时输出微米单位（便于和实际尺寸对齐）
    scale = float(getattr(cfg.plot, "px_to_um", 1.0))
    logger.info(
        "Thickness(%s)：min=%.3f, mean=%.3f, p95=%.3f, max=%.3f",
        getattr(cfg.plot, "length_unit", "µm"),
        summary.thickness_min * scale,
        summary.thickness_mean * scale,
        summary.thickness_p95 * scale,
        summary.thickness_max * scale,
    )
    logger.info(
        "Top Defect(px)：p95=%.3f, max=%.3f | Bottom Defect(px)：p95=%.3f, max=%.3f",
        summary.top_defect_p95,
        summary.top_defect_max,
        summary.bottom_defect_p95,
        summary.bottom_defect_max,
    )

    logger.info(
        "Top Defect(%s)：p95=%.3f, max=%.3f | Bottom Defect(%s)：p95=%.3f, max=%.3f",
        getattr(cfg.plot, "length_unit", "µm"),
        summary.top_defect_p95 * scale,
        summary.top_defect_max * scale,
        getattr(cfg.plot, "length_unit", "µm"),
        summary.bottom_defect_p95 * scale,
        summary.bottom_defect_max * scale,
    )

    # 8) GUI
    if cfg.io.enable_gui:
        logger.info("步骤7：启动交互窗口（鼠标悬停查看局部测量） ...")
        plot_interactive((h, w), x_fit, fit_top, fit_bottom, cfg.plot)
    else:
        logger.info("已关闭交互窗口（enable_gui=False）")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipe deformation reconstruction & measurement")
    parser.add_argument("--input", type=str, default=IOConfig().input_path, help="输入灰度图路径")
    parser.add_argument("--save", type=str, default=IOConfig().static_vis_save_path, help="静态结果图保存路径")
    parser.add_argument("--no-gui", action="store_true", help="不启动 Matplotlib 交互窗口")
    parser.add_argument("--log", type=str, default="INFO", help="日志等级：DEBUG/INFO/WARNING/ERROR")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = build_logger(args.log)

    # 把 CLI 覆盖到 config
    cfg = PipelineConfig(
        io=IOConfig(
            input_path=args.input,
            static_vis_save_path=args.save,
            enable_gui=(not args.no_gui),
        ),
        edge=PipelineConfig().edge,
        refine=PipelineConfig().refine,
        clean=PipelineConfig().clean,
        fit=PipelineConfig().fit,
        plot=PipelineConfig().plot,
    )

    try:
        run_pipeline(cfg, logger)
    except Exception as e:
        logger.exception("运行失败：%s", str(e))


if __name__ == "__main__":
    main()
