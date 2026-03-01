#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
只输入 ROI 灰度图，复用 h_test.py 的点集生成流程：
1) Sobel 梯度粗定位（find_edges_gradient_coarse）
2) 1D 剖面亚像素细化（refine_edge_subpixel）
3) 滑动中位数 + 阈值剔除（clean_edge_points）

输出（matplotlib，透明底，适合直接拖进 PPT）：
- top/bottom ROI 叠加图（有效点 vs 离群点）
- top/bottom 清洗前后对比图（含滑动中位数 + 阈值带）
- top/bottom 局部窗口放大示意图（讲清 W/T）
- 一张流程图（PPT 风格）
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

import h_test as ht  # <-- 关键：直接复用你现成的 h_test.py
import matplotlib.pyplot as plt

def set_chinese_font():
    # 常见中文字体候选（按优先级）
    candidates = [
        "Microsoft YaHei",   # 微软雅黑
        "SimHei",            # 黑体
        "SimSun",            # 宋体
        "KaiTi",             # 楷体
        "Noto Sans CJK SC",  # 思源黑体 / Noto CJK
        "Source Han Sans SC"
    ]
    for f in candidates:
        try:
            plt.rcParams["font.sans-serif"] = [f]
            plt.rcParams["axes.unicode_minus"] = False
            # 触发一次字体查找：如果字体不存在，matplotlib会fallback，但不会抛异常
            # 所以我们再用一个简单办法：直接返回即可（一般 Windows 上 YaHei/SimHei 足够）
            return
        except Exception:
            pass

# PPT 友好的全局样式（必须在 set_chinese_font 之前，否则 style.use 会重置字体设置）
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")

set_chinese_font()

_FIG_DPI = 150          # 输出分辨率
_FIG_W   = 8.5          # 单图宽度（英寸）
_FIG_W2  = 10.0         # 双列图宽度（英寸）
_FIG_H   = 4.5          # 图高度（英寸）


def _ensure_odd(v: int, min_value: int = 3) -> int:
    v = int(v)
    if v < min_value:
        v = min_value
    if v % 2 == 0:
        v += 1
    return v


def _sliding_median_baseline(x: np.ndarray, y: np.ndarray, win: int):
    """为了画图：按 x 排序后做滑动中位数基准（与 h_test.clean_edge_points 同思路）"""
    win = _ensure_odd(win, 3)
    sort_idx = np.argsort(x)
    xs = x[sort_idx].astype(float)
    ys = y[sort_idx].astype(np.float32)

    if ys.size < win:
        return xs, ys, ys.copy(), sort_idx

    pad = win // 2
    padded = np.pad(ys, (pad, pad), mode="edge")
    y_med = np.array([np.median(padded[i:i + win]) for i in range(ys.size)], dtype=np.float32)
    return xs, ys, y_med, sort_idx


def plot_points_on_roi(roi_gray, x, y, mask, save_path, title):
    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H), constrained_layout=True)
    ax.imshow(roi_gray, cmap="gray", origin="upper", aspect="auto")
    ax.scatter(x[mask], y[mask], s=6, alpha=0.9, label="有效点")
    ax.scatter(x[~mask], y[~mask], s=14, alpha=0.9, label="离群点(剔除)", zorder=5)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, roi_gray.shape[1] - 1)
    ax.set_ylim(roi_gray.shape[0] - 1, 0)
    ax.set_xlabel("X (px)", fontsize=11)
    ax.set_ylabel("Y (px)", fontsize=11)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.8)
    fig.savefig(save_path, transparent=True, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cleaning_overview(x, y, mask, win, T, save_path, title_prefix, half_width=250):
    xs, ys, y_med, sort_idx = _sliding_median_baseline(x, y, win)
    mask_sorted = mask[sort_idx]

    # 取中间 500px 窗口
    cx = float(np.median(xs))
    x0, x1 = cx - half_width, cx + half_width
    sel = (xs >= x0) & (xs <= x1)
    x_all = x.astype(float)
    y_all = y.astype(float)
    in_win = (x_all >= x0) & (x_all <= x1)

    fig, axes = plt.subplots(1, 2, figsize=(_FIG_W2, _FIG_H), constrained_layout=True)
    for ax in axes:
        ax.invert_yaxis()
        ax.set_xlabel("X (px)", fontsize=11)
        ax.set_ylabel("Y (px)", fontsize=11)
        ax.set_xlim(x0, x1)

    axes[0].set_title(f"{title_prefix} — 清洗前", fontsize=12, fontweight="bold")
    axes[0].scatter(x_all[in_win &  mask], y_all[in_win &  mask], s=8, alpha=0.85, label="有效点")
    axes[0].scatter(x_all[in_win & ~mask], y_all[in_win & ~mask], s=18, alpha=0.9,  label="离群点(剔除)", zorder=5)
    axes[0].plot(xs[sel], y_med[sel], linewidth=2.0, label="滑动中位数", alpha=0.9)
    axes[0].fill_between(xs[sel], y_med[sel] - T, y_med[sel] + T, alpha=0.15, label="阈值带 ±T")

    axes[1].set_title(f"{title_prefix} — 清洗后", fontsize=12, fontweight="bold")
    axes[1].scatter(x_all[in_win & mask], y_all[in_win & mask], s=8, alpha=0.85, label="有效点")
    axes[1].plot(xs[sel], y_med[sel], linewidth=2.0, label="滑动中位数", alpha=0.9)
    axes[1].fill_between(xs[sel], y_med[sel] - T, y_med[sel] + T, alpha=0.15)

    for ax in axes:
        ax.legend(fontsize=10, loc="upper right", framealpha=0.8)

    fig.savefig(save_path, transparent=True, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_local_window(x, y, mask, win, T, save_path, title, center_x=None, half_width=300):
    xs, ys, y_med, sort_idx = _sliding_median_baseline(x, y, win)
    mask_sorted = mask[sort_idx]

    if xs.size == 0:
        return

    if center_x is None:
        center_x = float(np.median(xs))

    x0, x1 = center_x - half_width, center_x + half_width
    sel = (xs >= x0) & (xs <= x1)

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H), constrained_layout=True)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("X (px)", fontsize=11)
    ax.set_ylabel("Y (px)", fontsize=11)

    ax.axvspan(x0, x1, alpha=0.07, label="示例局部窗口")
    ax.scatter(xs[sel & mask_sorted], ys[sel & mask_sorted], s=18, alpha=0.85, label="窗口内有效点")
    ax.scatter(xs[sel & (~mask_sorted)], ys[sel & (~mask_sorted)], s=28, alpha=0.9, label="窗口内离群点", zorder=5)
    ax.plot(xs[sel], y_med[sel], linewidth=2.2, label="窗口内滑动中位数")
    ax.fill_between(xs[sel], y_med[sel] - T, y_med[sel] + T, alpha=0.15, label="阈值带 ±T")

    ax.text(
        0.02, 0.04,
        "判定：|y − y_med| < T → 保留；否则剔除\nW 影响平滑程度，T 控制剔除力度",
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
    )

    ax.legend(fontsize=10, loc="upper right", framealpha=0.8)
    fig.savefig(save_path, transparent=True, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_flowchart(save_path: str):
    fig = plt.figure(figsize=(9.5, 3.8), dpi=_FIG_DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    container = Rectangle((0.02, 0.12), 0.96, 0.72, fill=False, linewidth=1.8,
                           linestyle=(0, (5, 3)), edgecolor="#888888")
    ax.add_patch(container)

    def add_box(x, y, w, h, text, fc, ec, fontsize=11):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.8, edgecolor=ec, facecolor=fc
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, linespacing=1.5)

    def add_arrow(x0, x1, y_mid):
        arr = FancyArrowPatch((x0, y_mid), (x1, y_mid),
                              arrowstyle="-|>", mutation_scale=16, linewidth=1.8,
                              color="#444444")
        ax.add_patch(arr)

    blue_fc, blue_ec     = "#EAF2FF", "#2F6FDB"
    yellow_fc, yellow_ec = "#FFF8E7", "#D39B1E"

    y_mid, h_box = 0.40, 0.26
    gap = 0.04
    # 五个框等宽，均匀分布
    w_each = (0.96 - 2 * 0.03 - 4 * gap) / 5
    xs = [0.03 + i * (w_each + gap) for i in range(5)]

    colors = [(blue_fc, blue_ec), (yellow_fc, yellow_ec),
              (blue_fc, blue_ec), (yellow_fc, yellow_ec), (blue_fc, blue_ec)]
    labels = ["输入 ROI\n灰度图", "粗定位\nSobel(Y)", "亚像素细化\n1D 剖面插值",
              "滑动中位数\n求 y_med", "阈值判断\n保留/剔除"]

    for i, (x, (fc, ec), lbl) in enumerate(zip(xs, colors, labels)):
        add_box(x, y_mid, w_each, h_box, lbl, fc, ec)
        if i < 4:
            add_arrow(x + w_each + 0.002, xs[i + 1] - 0.002, y_mid + h_box / 2)

    ax.text(0.03, 0.90, "离群点剔除流程（粗定位 → 细化 → 滑动中位数 + 阈值）",
            fontsize=13, fontweight="bold", color="#222222")
    ax.text(0.03, 0.08, "关键参数：窗口长度 W（median_window）  |  阈值 T（diff_thresh）",
            fontsize=11, color="#444444")

    fig.savefig(save_path, transparent=False, dpi=_FIG_DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def run_one_edge(img, x, y, clean_cfg, out_dir, tag):
    x2, y2, mask = ht.clean_edge_points(x, y, clean_cfg)

    plot_points_on_roi(
        img, x.astype(float), y.astype(float), mask,
        save_path=os.path.join(out_dir, f"{tag}_roi_overlay.png"),
        title=f"{tag} edge on ROI (valid vs outliers)"
    )
    plot_cleaning_overview(
        x.astype(float), y.astype(float), mask,
        win=int(clean_cfg.median_window),
        T=float(clean_cfg.diff_thresh),
        save_path=os.path.join(out_dir, f"{tag}_cleaning_overview.png"),
        title_prefix=f"{tag} edge"
    )
    plot_local_window(
        x.astype(float), y.astype(float), mask,
        win=int(clean_cfg.median_window),
        T=float(clean_cfg.diff_thresh),
        save_path=os.path.join(out_dir, f"{tag}_local_window.png"),
        title=f"{tag} 局部窗口示意",
        center_x=None,
        half_width=300
    )

    return x2, y2, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="ROI 灰度图路径（单通道）")
    ap.add_argument("--out", default="out_clean_assets", help="输出目录")
    ap.add_argument("--edge", default="both", choices=["top", "bottom", "both"], help="输出哪条边的图")
    ap.add_argument("--W", type=int, default=199, help="clean.median_window")
    ap.add_argument("--T", type=float, default=2.0, help="clean.diff_thresh")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1) 读图（复用 h_test）
    img = ht.load_grayscale_image(args.input)

    # 2) 粗定位（复用 h_test）
    pts_coarse = ht.find_edges_gradient_coarse(img, ht.EdgeDetectConfig())

    # 3) 亚像素细化（复用 h_test）
    refine_cfg = ht.SubpixelRefineConfig()
    xt, yt = ht.refine_edge_subpixel(img, pts_coarse.x_top, pts_coarse.y_top, is_top_edge=True, cfg=refine_cfg)
    xb, yb = ht.refine_edge_subpixel(img, pts_coarse.x_bottom, pts_coarse.y_bottom, is_top_edge=False, cfg=refine_cfg)

    # 4) 清洗（复用 h_test 的 clean_edge_points）
    clean_cfg = ht.CleanConfig(median_window=_ensure_odd(args.W), diff_thresh=float(args.T))

    if args.edge in ("top", "both"):
        run_one_edge(img, xt, yt, clean_cfg, args.out, "top")
    if args.edge in ("bottom", "both"):
        run_one_edge(img, xb, yb, clean_cfg, args.out, "bottom")

    # 5) 流程图
    plot_flowchart(os.path.join(args.out, "flowchart_outlier_removal.png"))

    print(f"[OK] assets saved to: {args.out}")
    print(" - top_roi_overlay / top_cleaning_overview / top_local_window (if enabled)")
    print(" - bottom_roi_overlay / bottom_cleaning_overview / bottom_local_window (if enabled)")
    print(" - flowchart_outlier_removal.png")


if __name__ == "__main__":
    main()