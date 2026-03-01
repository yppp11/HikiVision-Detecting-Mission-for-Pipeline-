import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Optional
# 直接复用你 h_test.py 里的配置与函数
from h_test import EdgeDetectConfig, load_grayscale_image, find_edges_gradient_coarse, _ensure_odd


def _compute_grad_y(img: np.ndarray, cfg: EdgeDetectConfig) -> np.ndarray:
    """完全按你代码：median -> Sobel(Y)"""
    median_ksize = _ensure_odd("edge.median_ksize", cfg.median_ksize, min_value=3)
    img_smooth = cv2.medianBlur(img, median_ksize)
    grad_y = cv2.Sobel(img_smooth, cv2.CV_64F, 0, 1, ksize=3)
    return grad_y


def _pick_representative_column(grad_y: np.ndarray, cfg: EdgeDetectConfig) -> int:
    """自动挑一列：上区正梯度强 + 下区负梯度强，便于PPT展示"""
    h, w = grad_y.shape
    lim_top = int(h * float(cfg.search_ratio_top))
    start_bottom = int(h * (1.0 - float(cfg.search_ratio_bottom)))
    lim_top = int(np.clip(lim_top, 1, h))
    start_bottom = int(np.clip(start_bottom, 0, h - 1))

    top_score = np.max(grad_y[:lim_top, :], axis=0)  # 正梯度峰值
    bot_score = np.abs(np.min(grad_y[start_bottom:, :], axis=0))  # 负梯度谷值绝对值
    score = top_score + bot_score
    return int(np.argmax(score))


def _coarse_find_on_column(grad_col: np.ndarray, h: int, cfg: EdgeDetectConfig):
    """
    单列复刻你 find_edges_gradient_coarse 的逻辑
    返回：lim_top, start_bottom, th_top, th_bottom, y_top, y_bottom_raw, y_bottom_refined
    """
    lim_top = int(h * float(cfg.search_ratio_top))
    start_bottom = int(h * (1.0 - float(cfg.search_ratio_bottom)))
    lim_top = int(np.clip(lim_top, 1, h))
    start_bottom = int(np.clip(start_bottom, 0, h - 1))

    # top threshold
    reg_top = grad_col[:lim_top]
    max_top = float(np.max(reg_top)) if reg_top.size else 0.0
    th_top = max(float(cfg.min_abs_thresh), max_top * float(cfg.rel_thresh_top))

    y_top = None
    for y in range(2, lim_top):
        if grad_col[y] > th_top:
            y_top = int(y)
            break

    # bottom threshold
    reg_bottom = grad_col[start_bottom:]
    min_bottom = float(np.min(reg_bottom)) if reg_bottom.size else 0.0
    th_bottom = min(-float(cfg.min_abs_thresh), min_bottom * float(cfg.rel_thresh_bottom))

    y_bottom_raw = None
    y_bottom_refined = None
    for y in range(h - 3, start_bottom, -1):
        if grad_col[y] < th_bottom:
            y_bottom_raw = int(y)
            peek_start = max(0, y - int(cfg.look_ahead_win))
            local = grad_col[peek_start:y + 1]
            if local.size > 0:
                y_bottom_refined = int(peek_start + np.argmin(local))
            else:
                y_bottom_refined = int(y)
            break

    return lim_top, start_bottom, th_top, th_bottom, y_top, y_bottom_raw, y_bottom_refined


# ---------------------------
# 图1：ROI + 上下搜索区间
# ---------------------------
def _crop_center(arr: np.ndarray, crop_w: int) -> tuple:
    """裁取中心 crop_w 列，返回 (裁剪后数组, x_offset)"""
    w = arr.shape[1]
    if w <= crop_w:
        return arr, 0
    x0 = (w - crop_w) // 2
    return arr[:, x0:x0 + crop_w], x0


def plot_roi_search_regions(img: np.ndarray, cfg: EdgeDetectConfig, save_path: str,
                            crop_w: int = 400):
    h, w = img.shape
    lim_top = int(np.clip(int(h * float(cfg.search_ratio_top)), 1, h))
    start_bottom = int(np.clip(int(h * (1.0 - float(cfg.search_ratio_bottom))), 0, h - 1))

    img_show, x0 = _crop_center(img, crop_w)
    wc = img_show.shape[1]

    fig, ax = plt.subplots(figsize=(3.0, 1.8))
    ax.imshow(img_show, cmap="gray", extent=[x0, x0 + wc, h, 0],
              interpolation="antialiased", aspect="auto")
    ax.set_title("ROI + Search Regions (Top/Bottom)", fontsize=8)
    ax.set_xlabel("x (px)", fontsize=7)
    ax.set_ylabel("y (px)", fontsize=7)
    ax.tick_params(labelsize=6)

    # 上搜索区
    ax.add_patch(Rectangle((x0, 0), wc, lim_top, fill=True, alpha=0.15))
    ax.text(x0 + 5, max(5, lim_top * 0.2), f"Top: y<{lim_top}", va="top", fontsize=7)

    # 下搜索区
    ax.add_patch(Rectangle((x0, start_bottom), wc, h - start_bottom, fill=True, alpha=0.15))
    ax.text(x0 + 5, min(h - 5, start_bottom + 10), f"Bot: y>{start_bottom}", va="top", fontsize=7)

    # 分界线
    ax.axhline(lim_top, linewidth=0.8)
    ax.axhline(start_bottom, linewidth=0.8)

    ax.set_xlim(x0, x0 + wc - 1)
    ax.set_ylim(h - 1, 0)
    fig.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# 图2：单列 Gy 曲线 + 阈值 + 越界点
# ---------------------------
def plot_single_column_gradient(img: np.ndarray, cfg: EdgeDetectConfig, save_path: str, x0=None):
    grad_y = _compute_grad_y(img, cfg)
    h, w = grad_y.shape

    if x0 is None:
        x0 = _pick_representative_column(grad_y, cfg)

    col = grad_y[:, x0]
    lim_top, start_bottom, th_top, th_bottom, y_top, yb_raw, yb_ref = _coarse_find_on_column(col, h, cfg)

    y = np.arange(h)

    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    ax.plot(col, y, linewidth=0.9)
    ax.set_title(f"Single Column Gy (x={x0})", fontsize=8)
    ax.set_xlabel("Gy (Sobel Y)", fontsize=7)
    ax.set_ylabel("y (px)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_ylim(h - 1, 0)

    # 区间线
    ax.axhline(lim_top, linewidth=0.8)
    ax.axhline(start_bottom, linewidth=0.8)

    # 阈值线
    ax.axvline(th_top, linestyle="--", linewidth=0.8)
    ax.axvline(th_bottom, linestyle="--", linewidth=0.8)

    # 点标注
    if y_top is not None:
        ax.scatter([col[y_top]], [y_top], s=20)
        ax.text(col[y_top], y_top, "  top", va="center", fontsize=6.5)

    if yb_raw is not None:
        ax.scatter([col[yb_raw]], [yb_raw], s=20)
        ax.text(col[yb_raw], yb_raw, "  bot", va="center", fontsize=6.5)

    if yb_ref is not None:
        ax.scatter([col[yb_ref]], [yb_ref], s=20)
        ax.text(col[yb_ref], yb_ref, "  refine", va="center", fontsize=6.5)

    # 角落写参数
    info = (
        f"med={cfg.median_ksize}, Sobel k=3\n"
        f"T_top={th_top:.1f}, T_bot={th_bottom:.1f}"
    )
    ax.text(0.02, 0.02, info, transform=ax.transAxes, va="bottom", ha="left", fontsize=6)

    fig.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# 图3：全列粗定位结果 y_top(x), y_bottom(x)
# ---------------------------
def plot_edges_over_x(img: np.ndarray, cfg: EdgeDetectConfig, save_path: str,
                      crop_w: int = 400):
    pts = find_edges_gradient_coarse(img, cfg)
    h, w = img.shape

    # 只展示中心 crop_w 列
    x_off = (w - crop_w) // 2 if w > crop_w else 0
    x_end = x_off + min(crop_w, w)
    mask_t = (pts.x_top >= x_off) & (pts.x_top < x_end)
    mask_b = (pts.x_bottom >= x_off) & (pts.x_bottom < x_end)

    fig, ax = plt.subplots(figsize=(3.0, 1.8))
    ax.scatter(pts.x_top[mask_t], pts.y_top[mask_t], s=3, label="top coarse")
    ax.scatter(pts.x_bottom[mask_b], pts.y_bottom[mask_b], s=3, label="bottom coarse")
    ax.set_title("Coarse Edges over X", fontsize=8)
    ax.set_xlabel("x (px)", fontsize=7)
    ax.set_ylabel("y (px)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xlim(x_off, x_end - 1)
    ax.set_ylim(h - 1, 0)
    ax.legend(fontsize=6.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# 可选图4：Gy 热力图（正/负梯度带）
# ---------------------------
def plot_grad_y_heatmap(img: np.ndarray, cfg: EdgeDetectConfig, save_path: str,
                        crop_w: int = 400):
    grad_y = _compute_grad_y(img, cfg)
    h, w = grad_y.shape

    grad_show, x0 = _crop_center(grad_y, crop_w)
    wc = grad_show.shape[1]

    v = np.percentile(np.abs(grad_show), 99)
    v = max(v, 1e-6)

    lim_top = int(np.clip(int(h * float(cfg.search_ratio_top)), 1, h))
    start_bottom = int(np.clip(int(h * (1.0 - float(cfg.search_ratio_bottom))), 0, h - 1))

    fig, ax = plt.subplots(figsize=(3.0, 1.8))
    im = ax.imshow(grad_show, cmap="seismic", vmin=-v, vmax=v,
                   extent=[x0, x0 + wc, h, 0],
                   interpolation="antialiased", aspect="auto")
    ax.set_title("Gy (Sobel Y) Heatmap", fontsize=8)
    ax.set_xlabel("x (px)", fontsize=7)
    ax.set_ylabel("y (px)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_ylim(h - 1, 0)

    ax.axhline(lim_top, linewidth=0.8)
    ax.axhline(start_bottom, linewidth=0.8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    # 你把路径换成自己的 ROI 灰度图即可
    img = load_grayscale_image("roi_pipe.png")

    cfg = EdgeDetectConfig(
        median_ksize=5,
        search_ratio_top=0.55,
        rel_thresh_top=0.15,
        search_ratio_bottom=0.45,
        rel_thresh_bottom=0.55,
        look_ahead_win=5,
        min_abs_thresh=4.0,
    )

    plot_roi_search_regions(img, cfg, "fig1_roi_search_regions.png")
    plot_single_column_gradient(img, cfg, "fig2_single_col_gy.png", x0=None)  # x0=None 会自动挑列
    plot_edges_over_x(img, cfg, "fig3_edges_over_x.png")
    plot_grad_y_heatmap(img, cfg, "fig4_gy_heatmap.png")