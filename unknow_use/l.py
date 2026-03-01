import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 直接复用你的 h_test.py
from h_test import (
    EdgeDetectConfig,
    SubpixelRefineConfig,
    load_grayscale_image,
    find_edges_gradient_coarse,
    refine_edge_subpixel,
    _ensure_odd,
)

# -------------------------
# 跟 h_test 一致的预处理：median + Sobel(Y)
# （用于挑一个"好看的列 x0"，以及可选的梯度可视化）
# -------------------------
def compute_grad_y_like_htest(img, edge_cfg):
    k = _ensure_odd("edge.median_ksize", int(edge_cfg.median_ksize), min_value=3)
    img_smooth = cv2.medianBlur(img, k)
    grad_y = cv2.Sobel(img_smooth, cv2.CV_64F, 0, 1, ksize=3)
    return grad_y

def _crop_center(arr: np.ndarray, crop_w: int) -> tuple:
    """裁取中心 crop_w 列，返回 (裁剪后数组, x_offset)"""
    w = arr.shape[1]
    if w <= crop_w:
        return arr, 0
    x0 = (w - crop_w) // 2
    return arr[:, x0:x0 + crop_w], x0


def pick_x0_with_both_edges(grad_y, pts_coarse):
    """
    优先选一个同时有 top & bottom 的列，并且上下梯度都明显的列，PPT更直观。
    """
    x_top = pts_coarse.x_top
    y_top = pts_coarse.y_top
    x_bot = pts_coarse.x_bottom
    y_bot = pts_coarse.y_bottom

    if x_top.size == 0 and x_bot.size == 0:
        return None

    top_map = {int(x): int(y) for x, y in zip(x_top, y_top)}
    bot_map = {int(x): int(y) for x, y in zip(x_bot, y_bot)}
    inter = sorted(set(top_map.keys()) & set(bot_map.keys()))

    # 如果没有交集，就退化：选 top 或 bottom 任意一边
    candidates = inter if len(inter) > 0 else sorted(set(top_map.keys()) | set(bot_map.keys()))
    if len(candidates) == 0:
        return None

    h, w = grad_y.shape
    scores = []
    for x in candidates:
        # 在该列上下各取一个强度评价：上区最大正梯度 + 下区最小负梯度绝对值
        col = grad_y[:, x]
        # 粗点附近更靠谱：用粗点位置作为中心取小窗评分（更符合"这列确实有边"）
        yt = top_map.get(x, None)
        yb = bot_map.get(x, None)

        s = 0.0
        if yt is not None:
            y1 = max(0, yt - 10)
            y2 = min(h, yt + 10)
            s += float(np.max(col[y1:y2]))
        else:
            s += float(np.max(col[: h // 2]))

        if yb is not None:
            y1 = max(0, yb - 10)
            y2 = min(h, yb + 10)
            s += float(np.abs(np.min(col[y1:y2])))
        else:
            s += float(np.abs(np.min(col[h // 2 :])))

        scores.append(s)

    return int(candidates[int(np.argmax(scores))])

# -------------------------
# 单点剖面：完全按 refine_edge_subpixel 的规则复刻（便于画图标注）
# -------------------------
def compute_profile_and_intersection(img, x0, y0, refine_cfg, is_top_edge):
    h, w = img.shape
    r = int(refine_cfg.window_radius)
    ratio = float(np.clip(float(refine_cfg.intensity_ratio), 0.0, 1.0))

    if x0 < 0 or x0 >= w or y0 - r < 0 or y0 + r >= h:
        return None

    prof = img[y0 - r : y0 + r + 1, x0].astype(np.float32)
    yy = np.arange(y0 - r, y0 + r + 1, dtype=np.int32)

    # h_test：头2个 vs 尾2个（上下边缘规则相反）
    if is_top_edge:
        bg_idx = slice(0, 2)
        fg_idx = slice(-2, None)
        val_bg = float(np.mean(prof[bg_idx]))
        val_fg = float(np.mean(prof[fg_idx]))
    else:
        bg_idx = slice(-2, None)
        fg_idx = slice(0, 2)
        val_bg = float(np.mean(prof[bg_idx]))
        val_fg = float(np.mean(prof[fg_idx]))

    target = val_bg + (val_fg - val_bg) * ratio

    # 按 refine_edge_subpixel：找交点并线性插值
    hit_j = None
    y_ref = float(y0)
    for j in range(len(prof) - 1):
        v1, v2 = float(prof[j]), float(prof[j + 1])
        if (v1 <= target <= v2) or (v2 <= target <= v1):
            denom = (v2 - v1)
            offset = (target - v1) / denom if abs(denom) > 1e-6 else 0.5
            y_ref = float((y0 - r) + j + offset)
            hit_j = int(j)
            break

    return {
        "r": r,
        "prof": prof,
        "yy": yy,
        "bg_idx": bg_idx,
        "fg_idx": fg_idx,
        "val_bg": val_bg,
        "val_fg": val_fg,
        "ratio": ratio,
        "target": target,
        "hit_j": hit_j,
        "y_ref": y_ref,
    }

def plot_profile(profile_info, x0, y0, is_top_edge, save_path):
    prof = profile_info["prof"]
    yy = profile_info["yy"]
    bg_idx = profile_info["bg_idx"]
    fg_idx = profile_info["fg_idx"]
    target = profile_info["target"]
    y_ref = profile_info["y_ref"]
    hit_j = profile_info["hit_j"]

    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    ax.plot(prof, yy, linewidth=1.2)
    ax.invert_yaxis()
    ax.set_xlabel("Intensity", fontsize=8)
    ax.set_ylabel("y (px)", fontsize=8)
    ax.set_title(f"{'Top' if is_top_edge else 'Bottom'} Edge Profile (x={x0}, y0={y0})", fontsize=8)
    ax.tick_params(labelsize=7)

    # bg/fg 样本点
    ax.scatter(prof[bg_idx], yy[bg_idx], s=30, label="bg samples")
    ax.scatter(prof[fg_idx], yy[fg_idx], s=30, label="fg samples")

    # target
    ax.axvline(target, linestyle="--", linewidth=1.0, label="target intensity")

    # coarse y0 & refined y
    ax.axhline(y0, linestyle=":", linewidth=1.0, label="coarse y0")
    ax.scatter([target], [y_ref], s=60, label="refined (interp)")

    # 命中的线段端点
    if hit_j is not None:
        ax.scatter([prof[hit_j], prof[hit_j + 1]], [yy[hit_j], yy[hit_j + 1]], s=30)

    # 角落信息
    info = (
        f"bg={profile_info['val_bg']:.1f}, fg={profile_info['val_fg']:.1f}\n"
        f"ratio={profile_info['ratio']:.2f}, target={profile_info['target']:.1f}\n"
        f"refined y={profile_info['y_ref']:.3f}"
    )
    ax.text(0.02, 0.02, info, transform=ax.transAxes, va="bottom", ha="left", fontsize=6.5)

    ax.legend(fontsize=6.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# -------------------------
# ROI 叠加：粗点 vs 细点 vs 清洗后细点
# -------------------------
def plot_overlay_points(img, x_top_ref, y_top_ref, x_bot_ref, y_bot_ref,
                        save_path, crop_w: int = 400):
    h, w = img.shape
    img_show, x_off = _crop_center(img, crop_w)
    wc = img_show.shape[1]
    x_end = x_off + wc

    def _mask(xarr):
        return (xarr >= x_off) & (xarr < x_end)

    fig, ax = plt.subplots(figsize=(3.0, 1.8))
    ax.imshow(img_show, cmap="gray", extent=[x_off, x_end, h, 0],
              interpolation="antialiased", aspect="auto")
    ax.set_title("Refined Edge Points", fontsize=8)
    ax.set_xlabel("x (px)", fontsize=7)
    ax.set_ylabel("y (px)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xlim(x_off, x_end - 1)
    ax.set_ylim(h - 1, 0)

    mt = _mask(x_top_ref); mb = _mask(x_bot_ref)
    ax.scatter(x_top_ref[mt], y_top_ref[mt], s=3, label="refined top")
    ax.scatter(x_bot_ref[mb], y_bot_ref[mb], s=3, label="refined bot")

    ax.legend(ncol=2, fontsize=6, loc="upper right")
    fig.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# -------------------------
# 在 ROI 上标出细定位窗口（对应"截取该列的灰度窗口"）
# -------------------------
def plot_windows_on_roi(img, x0, y0_top, y0_bottom, refine_cfg, save_path,
                        crop_w: int = 400):
    h, w = img.shape
    r = int(refine_cfg.window_radius)

    # 以 x0 为中心裁取 crop_w 列
    x_off = max(0, x0 - crop_w // 2)
    x_end = min(w, x_off + crop_w)
    x_off = max(0, x_end - crop_w)   # 靠近右边界时左移补足
    img_show = img[:, x_off:x_end]
    wc = img_show.shape[1]

    fig, ax = plt.subplots(figsize=(3.0, 1.8))
    ax.imshow(img_show, cmap="gray", extent=[x_off, x_off + wc, h, 0],
              interpolation="antialiased", aspect="auto")
    ax.set_title(f"Fine Localization Windows (x={x0}, r={r})", fontsize=8)
    ax.set_xlabel("x (px)", fontsize=7)
    ax.set_ylabel("y (px)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xlim(x_off, x_off + wc - 1)
    ax.set_ylim(h - 1, 0)

    band_w = 5
    x_left = max(x_off, x0 - band_w // 2)

    if y0_top is not None:
        y1 = max(0, y0_top - r)
        y2 = min(h - 1, y0_top + r)
        ax.add_patch(Rectangle((x_left, y1), band_w, (y2 - y1), fill=False, linewidth=1.2))
        ax.scatter([x0], [y0_top], s=30, label="coarse top y0")

    if y0_bottom is not None:
        y1 = max(0, y0_bottom - r)
        y2 = min(h - 1, y0_bottom + r)
        ax.add_patch(Rectangle((x_left, y1), band_w, (y2 - y1), fill=False, linewidth=1.2))
        ax.scatter([x0], [y0_bottom], s=30, label="coarse bot y0")

    ax.legend(fontsize=6.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="ROI grayscale image path")
    ap.add_argument("--outdir", type=str, default="fine_figs", help="output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    img = load_grayscale_image(args.input)

    # 你可以在这里改成你实际在 h_test 里用的参数
    edge_cfg = EdgeDetectConfig(
        median_ksize=5,
        search_ratio_top=0.55,
        rel_thresh_top=0.15,
        search_ratio_bottom=0.45,
        rel_thresh_bottom=0.55,
        look_ahead_win=5,
        min_abs_thresh=4.0,
    )
    refine_cfg = SubpixelRefineConfig(window_radius=10, intensity_ratio=0.5)

    # 1) 粗定位（htest）
    pts = find_edges_gradient_coarse(img, edge_cfg)

    # 2) 细定位（htest）
    x_top_ref, y_top_ref = refine_edge_subpixel(img, pts.x_top, pts.y_top, is_top_edge=True, cfg=refine_cfg)
    x_bot_ref, y_bot_ref = refine_edge_subpixel(img, pts.x_bottom, pts.y_bottom, is_top_edge=False, cfg=refine_cfg)

    # 图B：细定位叠加
    plot_overlay_points(
        img,
        x_top_ref, y_top_ref, x_bot_ref, y_bot_ref,
        os.path.join(args.outdir, "B_overlay_points.png")
    )

    # 选一个"好看列"画剖面（图A、图D）
    grad_y = compute_grad_y_like_htest(img, edge_cfg)
    x0 = pick_x0_with_both_edges(grad_y, pts)
    if x0 is None:
        print("No valid x0 found (no edges detected).")
        return

    # 找该列对应的 coarse y0（优先 top&bottom）
    top_map = {int(x): int(y) for x, y in zip(pts.x_top, pts.y_top)}
    bot_map = {int(x): int(y) for x, y in zip(pts.x_bottom, pts.y_bottom)}
    y0_top = top_map.get(int(x0), None)
    y0_bot = bot_map.get(int(x0), None)

    # 图D：窗口在 ROI 上的位置
    plot_windows_on_roi(
        img, x0, y0_top, y0_bot, refine_cfg,
        os.path.join(args.outdir, "D_windows_on_roi.png")
    )

    # 图A：单列剖面（上/下分别一张）
    if y0_top is not None:
        info_top = compute_profile_and_intersection(img, x0, y0_top, refine_cfg, is_top_edge=True)
        if info_top is not None:
            plot_profile(info_top, x0, y0_top, True, os.path.join(args.outdir, "A_profile_top.png"))

    if y0_bot is not None:
        info_bot = compute_profile_and_intersection(img, x0, y0_bot, refine_cfg, is_top_edge=False)
        if info_bot is not None:
            plot_profile(info_bot, x0, y0_bot, False, os.path.join(args.outdir, "A_profile_bottom.png"))

    print(f"Saved figures to: {args.outdir}")

if __name__ == "__main__":
    main()