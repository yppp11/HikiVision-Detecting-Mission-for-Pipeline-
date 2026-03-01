#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

import h_test as ht

# PPT 友好的全局样式（必须在字体设置之前，否则 style.use 会重置字体设置）
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("seaborn-whitegrid")


def set_chinese_font():
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "KaiTi",
        "Noto Sans CJK SC", "Source Han Sans SC"
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for c in candidates:
        if c in available:
            plt.rcParams["font.sans-serif"] = [c]
            plt.rcParams["axes.unicode_minus"] = False
            return


set_chinese_font()

_FIG_DPI = 200          # 输出分辨率
_FIG_W   = 10.0         # 图宽度（英寸）
_FIG_H   = 4.6          # 单行图高度
_FIG_H2  = 4.2          # ROI 叠加图高度（单边）
_FIG_HD  = 6.2          # 缺陷双行图高度


def _white_ax(ax):
    """统一设置白色背景，适合 PPT。"""
    ax.set_facecolor("white")


# ---------- 图1：baseline + inliers/outliers + ±kσ ----------
def plot_baseline_inliers_band(x, y, fit: ht.FitResult, fit_cfg: ht.FitConfig,
                                save_path: str, title: str,
                                x0: float = None, x1: float = None):
    if fit.baseline_poly is None:
        raise ValueError("baseline_poly is None，无法画 baseline 图。")

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.asarray(fit.inlier_mask, bool)

    # 按 x 排序
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    ms = mask[order]

    # 裁剪到显示窗口
    if x0 is not None and x1 is not None:
        sel = (xs >= x0) & (xs <= x1)
        xs, ys, ms = xs[sel], ys[sel], ms[sel]

    # 计算 sigma（对 inliers 的残差算 std，用原始全局数据）
    resid = y - fit.baseline_poly(x)
    sigma = float(np.std(resid[mask])) if mask.any() else float(np.std(resid))
    band = float(fit_cfg.sigma_thresh) * sigma

    y_base = fit.baseline_poly(xs)

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H), constrained_layout=True)
    _white_ax(ax)
    ax.invert_yaxis()

    ax.scatter(xs[ms],  ys[ms],  s=6,  alpha=0.9, label="内点(inliers)")
    ax.scatter(xs[~ms], ys[~ms], s=18, alpha=0.9, label="外点(outliers)", zorder=5)
    ax.plot(xs, y_base, linewidth=2.0, alpha=0.9, label=f"baseline poly(deg={fit_cfg.poly_degree})")
    ax.fill_between(xs, y_base - band, y_base + band, alpha=0.15, label=f"±kσ (k={fit_cfg.sigma_thresh:.1f})")

    if x0 is not None and x1 is not None:
        ax.set_xlim(x0, x1)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("X (px)", fontsize=11)
    ax.set_ylabel("Y (px)", fontsize=11)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.8)

    fig.savefig(save_path, transparent=True, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------- 图2：ROI 叠加单边 spline（每条边单独一张图） ----------
def plot_roi_overlay_single(img_gray, x_pts, y_pts, fit: ht.FitResult,
                             save_path: str, title: str,
                             x0: float = None, x1: float = None,
                             y_margin: int = 60):
    """单边 ROI 叠加图：灰度图背景 + 边缘点（青色）+ spline 拟合曲线（黄色）。"""
    h, w = img_gray.shape[:2]
    ix0 = int(max(0, x0)) if x0 is not None else 0
    ix1 = int(min(w - 1, x1)) if x1 is not None else w - 1
    x_eval = np.arange(ix0, ix1 + 1, dtype=np.float64)

    yc  = float(np.median(y_pts))
    iy0 = int(max(0,     yc - y_margin))
    iy1 = int(min(h - 1, yc + y_margin))

    fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H2), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax.imshow(img_gray[iy0:iy1 + 1, ix0:ix1 + 1],
              cmap="gray", origin="upper",
              extent=[ix0, ix1, iy1, iy0],
              aspect="auto", interpolation="nearest")
    ax.set_xlim(ix0, ix1)
    ax.set_ylim(iy1, iy0)

    # 青色散点 + 黄色拟合曲线，在灰度图上清晰可见
    ax.scatter(x_pts, y_pts, s=10, color="cyan",   alpha=0.9, label="边缘点", zorder=3)
    if fit.spline is not None:
        y_fit = ht._eval_spline_clipped(fit.spline, x_eval, fit.x_min, fit.x_max)
        ax.plot(x_eval, y_fit, color="yellow", linewidth=2.2, alpha=0.95,
                label="spline 拟合", zorder=4)

    ax.set_xlabel("X (px)", fontsize=11)
    ax.set_ylabel("Y (px)", fontsize=11)
    # 标题加白色衬底，防止与深色图像混淆
    ax.set_title(title, fontsize=13, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9,
              facecolor="white", edgecolor="gray")

    fig.savefig(save_path, transparent=False, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)


# ---------- 图3：baseline vs spline 的差值（阴影表示 |spline-baseline|） ----------
def plot_defect_fill(x_eval, fit: ht.FitResult, label: str, ax):
    _white_ax(ax)
    if fit.spline is None or fit.baseline_poly is None:
        ax.text(0.5, 0.5, f"{label}: fit failed", ha="center", va="center",
                transform=ax.transAxes)
        return

    y_spl  = ht._eval_spline_clipped(fit.spline, x_eval, fit.x_min, fit.x_max)
    y_base = fit.baseline_poly(x_eval)

    # 仅在有效 x 范围内展示（外推区间置 NaN）
    valid  = (x_eval >= fit.x_min) & (x_eval <= fit.x_max)
    defect = np.full_like(x_eval, np.nan, dtype=float)
    defect[valid] = np.abs(y_spl[valid] - y_base[valid])

    ax.plot(x_eval, defect, linewidth=2.0, alpha=0.9, label=f"{label} |spline−baseline|")
    ax.fill_between(x_eval, 0, defect, alpha=0.15)
    ax.set_ylabel("偏离 (px)", fontsize=11)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",    required=True,  help="ROI 灰度图路径")
    ap.add_argument("--out",      default="out_fit_assets", help="输出目录")
    ap.add_argument("--cleanW",   type=int,   default=199)
    ap.add_argument("--cleanT",   type=float, default=2.0)
    ap.add_argument("--deg",      type=int,   default=3)
    ap.add_argument("--k",        type=float, default=3.0)
    ap.add_argument("--max_iter", type=int,   default=5)
    ap.add_argument("--s_factor", type=float, default=0.1)

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    img = ht.load_grayscale_image(args.input)
    h, w = img.shape[:2]

    # 1) 点集（复用 h_test）
    pts_coarse = ht.find_edges_gradient_coarse(img, ht.EdgeDetectConfig())
    xt, yt = ht.refine_edge_subpixel(img, pts_coarse.x_top,    pts_coarse.y_top,    is_top_edge=True,  cfg=ht.SubpixelRefineConfig())
    xb, yb = ht.refine_edge_subpixel(img, pts_coarse.x_bottom, pts_coarse.y_bottom, is_top_edge=False, cfg=ht.SubpixelRefineConfig())

    # 2) 清洗
    clean_cfg = ht.CleanConfig(median_window=int(args.cleanW), diff_thresh=float(args.cleanT))
    xt2, yt2, _ = ht.clean_edge_points(xt, yt, clean_cfg)
    xb2, yb2, _ = ht.clean_edge_points(xb, yb, clean_cfg)
    pts_clean = ht.EdgePoints(x_top=xt2, y_top=yt2, x_bottom=xb2, y_bottom=yb2)

    # 3) 拟合
    fit_cfg = ht.FitConfig(
        poly_degree=int(args.deg),
        sigma_thresh=float(args.k),
        max_iter=int(args.max_iter),
        spline_s_factor=float(args.s_factor),
    )
    fit_top    = ht.fit_edge(pts_clean.x_top,    pts_clean.y_top,    fit_cfg)
    fit_bottom = ht.fit_edge(pts_clean.x_bottom, pts_clean.y_bottom, fit_cfg)

    if (fit_top.spline is None or fit_bottom.spline is None
            or fit_top.baseline_poly is None or fit_bottom.baseline_poly is None):
        raise RuntimeError("拟合失败：点太少/参数不合适/ROI 有问题。建议先放宽 cleanT 或增大点数覆盖。")

    # ---------- 计算显示窗口（中间 500px） ----------
    all_x = np.concatenate([pts_clean.x_top, pts_clean.x_bottom])
    cx = float(np.median(all_x))
    half_width = 250
    x0 = max(0.0, cx - half_width)
    x1 = min(float(w - 1), cx + half_width)
    x_eval = np.arange(int(x0), int(x1) + 1, dtype=np.float64)

    # ---------- 输出三类图 ----------
    # 图1：baseline + inliers/outliers + band（top/bottom 各一张）
    plot_baseline_inliers_band(
        pts_clean.x_top, pts_clean.y_top, fit_top, fit_cfg,
        save_path=os.path.join(args.out, "A_top_baseline_inliers_band.png"),
        title="Top：baseline + inliers/outliers + ±kσ",
        x0=x0, x1=x1,
    )
    plot_baseline_inliers_band(
        pts_clean.x_bottom, pts_clean.y_bottom, fit_bottom, fit_cfg,
        save_path=os.path.join(args.out, "A_bottom_baseline_inliers_band.png"),
        title="Bottom：baseline + inliers/outliers + ±kσ",
        x0=x0, x1=x1,
    )

    # 图2：ROI 叠加 spline（top/bottom 各单独一张）
    plot_roi_overlay_single(
        img, pts_clean.x_top, pts_clean.y_top, fit_top,
        save_path=os.path.join(args.out, "B_top_roi_overlay_spline.png"),
        title="Top edge ROI：spline 拟合结果（放大）",
        x0=x0, x1=x1,
    )
    plot_roi_overlay_single(
        img, pts_clean.x_bottom, pts_clean.y_bottom, fit_bottom,
        save_path=os.path.join(args.out, "B_bottom_roi_overlay_spline.png"),
        title="Bottom edge ROI：spline 拟合结果（放大）",
        x0=x0, x1=x1,
    )

    # 图3：|spline-baseline| 缺陷（top/bottom 两行一张）
    fig, axes = plt.subplots(2, 1, figsize=(_FIG_W, _FIG_HD), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("white")
    plot_defect_fill(x_eval, fit_top,    "Top",    axes[0])
    plot_defect_fill(x_eval, fit_bottom, "Bottom", axes[1])
    axes[1].set_xlabel("X (px)", fontsize=11)
    axes[1].set_xlim(x0, x1)
    fig.suptitle("缺陷示意：|spline − baseline|（阴影表示偏离幅度）", fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(args.out, "C_defect_fill_top_bottom.png"),
                transparent=True, dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    print("[OK] saved to:", args.out)
    print(" - A_top_baseline_inliers_band.png")
    print(" - A_bottom_baseline_inliers_band.png")
    print(" - B_top_roi_overlay_spline.png")
    print(" - B_bottom_roi_overlay_spline.png")
    print(" - C_defect_fill_top_bottom.png")


if __name__ == "__main__":
    main()
