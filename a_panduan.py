import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# SECTION 1: 全局配置参数 (Configuration)
# ==============================================================================

# --- [1.1] 输入与基本设置 ---
IMAGE_PATH = "roi_pipe.png"        # 输入图像路径
IMREAD_FLAG = cv2.IMREAD_UNCHANGED # 读取模式

# --- [1.2] 绘图布局参数 ---
FOUR_PLOTS_FIGSIZE = (12, 9)       # 画布大小
GRAY_HEATMAP_CMAP = "jet"          # 灰度图配色
GRAD_HEATMAP_CMAP = "jet"          # 梯度图配色
SCATTER_SIZE = 2                   # 散点大小
SCATTER_ALPHA = 0.8                # 散点透明度

# --- [1.3] 步骤1: 梯度粗定位参数 ---
MEDIAN_KSIZE = 5                   # 预处理中值滤波大小
SEARCH_RATIO_TOP = 0.55            # 上边缘搜索范围 (0 ~ h * ratio)
REL_THRESH_TOP = 0.20              # 上边缘相对阈值 (max * ratio)

SEARCH_RATIO_BOT = 0.45            # 下边缘搜索范围 (h * (1-ratio) ~ h)
REL_THRESH_BOT = 0.25              # 下边缘相对阈值 (min * ratio)
LOOK_AHEAD_WIN = 5                 # 下边缘抗虚影的前瞻窗口大小
MIN_ABS_THRESH = 4.0               # 最小梯度绝对值门限

# --- [1.4] 步骤2: 亚像素细化参数 (新增) ---
REFINE_WIN_SIZE = 10               # 细化分析窗口半径 (pixel)
REFINE_INTENSITY_RATIO = 0.5       # 亚像素插值阈值比例 (0.5 表示取亮暗均值处)

# --- [1.5] 步骤3: 数据清洗参数 (新增) ---
CLEAN_WINDOW = 21                  # 清洗噪点的滑动窗口大小
CLEAN_DIFF_THRESH = 5.0            # 允许的最大突变偏差 (pixel)

# --- [1.6] 梯度热度图显示参数 ---
GRAD_SOBEL_KSIZE = 3               # 用于绘图分析的 Sobel 核大小
GRAD_CLIP_PERCENTILE = 99          # 梯度显示亮度截断百分位


# ==============================================================================
# SECTION 2: 核心算法函数 (移植自 gui_test.py)
# ==============================================================================

def find_edges_gradient_coarse(
    img, 
    median_ksize, 
    search_ratio_top, 
    rel_thresh_top, 
    search_ratio_bot, 
    rel_thresh_bot, 
    min_abs_thresh, 
    look_ahead_win
):
    """
    [步骤1] 使用梯度法进行边缘的粗略定位。
    """
    h, w = img.shape
    # 1. 预处理
    img_smooth = cv2.medianBlur(img, median_ksize)
    # 2. 垂直梯度
    grad_y = cv2.Sobel(img_smooth, cv2.CV_64F, 0, 1, ksize=3)

    x_top, y_top = [], []
    x_bot, y_bot = [], []

    for x in range(w):
        col_grads = grad_y[:, x]
        
        # === A. 上边缘搜索 ===
        lim_t = int(h * search_ratio_top)
        reg_t = col_grads[0:lim_t]
        th_t = max(min_abs_thresh, (np.max(reg_t) if reg_t.size else 0) * rel_thresh_top)
        
        for y in range(2, lim_t):
            if col_grads[y] > th_t:
                x_top.append(x)
                y_top.append(y)
                break
        
        # === B. 下边缘搜索 (含 Look-Ahead) ===
        start_b = int(h * (1 - search_ratio_bot))
        reg_b = col_grads[start_b:h]
        min_val = np.min(reg_b) if reg_b.size else 0
        th_bot = min(-min_abs_thresh, min_val * rel_thresh_bot)
        
        for y in range(h - 3, start_b, -1):
            if col_grads[y] < th_bot:
                # 前瞻机制：往内部多看一段，寻找局部最强梯度
                peek_start = max(0, y - look_ahead_win)
                peek_end = y + 1 
                local_window = col_grads[peek_start : peek_end]
                
                if local_window.size > 0:
                    best_local_idx = np.argmin(local_window) # 找最负的点
                    real_y = peek_start + best_local_idx
                    x_bot.append(x)
                    y_bot.append(real_y)
                else:
                    x_bot.append(x)
                    y_bot.append(y)
                break 

    return np.array(x_top), np.array(y_top), np.array(x_bot), np.array(y_bot)


def refine_edge_subpixel(img, x_list, y_list, is_top_edge, win_size, intensity_ratio):
    """
    [步骤2] 基于局部灰度分析的亚像素边缘细化。
    """
    h, w = img.shape
    y_refined = []
    x_kept = []
    r = win_size
    
    for i in range(len(x_list)):
        x = int(x_list[i])
        y0 = int(y_list[i])
        
        if y0 - r < 0 or y0 + r >= h: continue

        local_profile = img[y0-r : y0+r+1, x].astype(np.float32)
        
        if is_top_edge:
            val_bg = np.mean(local_profile[0:2])     
            val_fg = np.mean(local_profile[-2:])     
        else:
            val_fg = np.mean(local_profile[0:2])     
            val_bg = np.mean(local_profile[-2:])     

        target_val = val_bg + (val_fg - val_bg) * intensity_ratio
        
        found_subpixel = False
        for j in range(len(local_profile) - 1):
            v1 = local_profile[j]
            v2 = local_profile[j+1]
            if (v1 <= target_val <= v2) or (v2 <= target_val <= v1):
                if abs(v2 - v1) < 1e-3: offset = 0.5
                else: offset = (target_val - v1) / (v2 - v1)
                
                real_y = (y0 - r) + j + offset
                y_refined.append(real_y)
                x_kept.append(x)
                found_subpixel = True
                break
        
        if not found_subpixel:
            y_refined.append(y0)
            x_kept.append(x)

    return np.array(x_kept), np.array(y_refined)


def clean_edge_points(x, y, window_size, diff_thresh):
    """
    [步骤3] 数据清洗，去除突变离群点。
    """
    if len(x) < window_size: return x, y
    y_vals = np.asarray(y, dtype=np.float32)
    pad_width = window_size // 2
    padded = np.pad(y_vals, (pad_width, pad_width), mode='edge')
    y_smooth = np.zeros_like(y_vals)
    for i in range(len(y_vals)):
        y_smooth[i] = np.median(padded[i : i + window_size])
    mask = np.abs(y_vals - y_smooth) < diff_thresh
    return x[mask], y[mask]


# ==============================================================================
# SECTION 3: 绘图与主流程
# ==============================================================================

def inspect_image(path):
    img = cv2.imread(path, IMREAD_FLAG)
    if img is None:
        raise FileNotFoundError(f"找不到图像文件: {path}")

    print("===== 图像处理分析 =====")
    print(f"File: {path}, Shape: {img.shape}")
    
    # 统一转灰度用于分析
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # --- 算法流水线 ---
    print("\n[Step 1] 梯度粗定位...")
    xt, yt, xb, yb = find_edges_gradient_coarse(
        img=gray,
        median_ksize=MEDIAN_KSIZE,
        search_ratio_top=SEARCH_RATIO_TOP,
        rel_thresh_top=REL_THRESH_TOP,
        search_ratio_bot=SEARCH_RATIO_BOT,
        rel_thresh_bot=REL_THRESH_BOT,
        min_abs_thresh=MIN_ABS_THRESH,
        look_ahead_win=LOOK_AHEAD_WIN
    )
    print(f"  > 粗定位: Top={len(xt)}, Bot={len(xb)}")

    print("[Step 2] 亚像素细化...")
    xt, yt = refine_edge_subpixel(gray, xt, yt, True, REFINE_WIN_SIZE, REFINE_INTENSITY_RATIO)
    xb, yb = refine_edge_subpixel(gray, xb, yb, False, REFINE_WIN_SIZE, REFINE_INTENSITY_RATIO)

    print("[Step 3] 离群点清洗...")
    xt, yt = clean_edge_points(xt, yt, CLEAN_WINDOW, CLEAN_DIFF_THRESH)
    xb, yb = clean_edge_points(xb, yb, CLEAN_WINDOW, CLEAN_DIFF_THRESH)
    print(f"  > 最终有效点: Top={len(xt)}, Bot={len(xb)}")

    # --- 绘图 ---
    fig, axes = plt.subplots(2, 2, figsize=FOUR_PLOTS_FIGSIZE)

    # 1. 左上：灰度热度图
    plot_gray_heatmap(gray, ax=axes[0, 0], title="[1] Gray Heatmap")

    # 2. 右上：最终边缘散点分布图 (Refined & Cleaned)
    plot_edge_scatter(gray, xt, yt, xb, yb, ax=axes[0, 1], title="[2] Final Refined Edge Points")

    # 3. 左下 & 右下：梯度分析
    plot_gradient_heatmap(gray, ax_gy=axes[1, 0], ax_mag=axes[1, 1])

    plt.tight_layout()
    plt.show()


def plot_gray_heatmap(gray, ax, title="Gray Heatmap"):
    """左上：纯灰度热度图"""
    im = ax.imshow(gray, cmap=GRAY_HEATMAP_CMAP, aspect='auto', origin='upper')
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Gray Value')


def plot_edge_scatter(gray, xt, yt, xb, yb, ax, title="Edge Scatter"):
    """右上：在原图上叠加边缘散点 (浮点坐标)"""
    # 背景显示为灰度，稍微暗一点以便看清散点
    ax.imshow(gray, cmap='gray', aspect='auto', origin='upper', alpha=0.8)
    
    # 绘制上边缘 (红色点)
    if len(xt) > 0:
        ax.scatter(xt, yt, s=SCATTER_SIZE, c='red', alpha=SCATTER_ALPHA, edgecolors='none')
    
    # 绘制下边缘 (蓝色点)
    if len(xb) > 0:
        ax.scatter(xb, yb, s=SCATTER_SIZE, c='cyan', alpha=SCATTER_ALPHA, edgecolors='none')

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # ax.legend(loc='upper right', fontsize='small')
    ax.grid(False) 


def plot_gradient_heatmap(gray, ax_gy, ax_mag):
    """下方两图：梯度分析"""
    gray_f = gray.astype(np.float32)
    
    # 计算梯度
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=GRAD_SOBEL_KSIZE)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=GRAD_SOBEL_KSIZE)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # 截断显示，避免极值影响
    gy_abs = np.abs(gy)
    gy_clip = np.clip(gy_abs, 0, np.percentile(gy_abs, GRAD_CLIP_PERCENTILE))
    mag_clip = np.clip(grad_mag, 0, np.percentile(grad_mag, GRAD_CLIP_PERCENTILE))

    # 左下：垂直梯度 |Gy|
    im1 = ax_gy.imshow(gy_clip, cmap=GRAD_HEATMAP_CMAP, aspect='auto', origin='upper')
    ax_gy.set_title("[3] Vertical Gradient |Gy|")
    ax_gy.set_xlabel("X")
    plt.colorbar(im1, ax=ax_gy, fraction=0.046, pad=0.04, label='|Gy|')

    # 右下：梯度幅值 |Grad|
    im2 = ax_mag.imshow(mag_clip, cmap=GRAD_HEATMAP_CMAP, aspect='auto', origin='upper')
    ax_mag.set_title("[4] Gradient Magnitude |∇I|")
    ax_mag.set_xlabel("X")
    plt.colorbar(im2, ax=ax_mag, fraction=0.046, pad=0.04, label='|∇I|')


if __name__ == "__main__":
    inspect_image(IMAGE_PATH)