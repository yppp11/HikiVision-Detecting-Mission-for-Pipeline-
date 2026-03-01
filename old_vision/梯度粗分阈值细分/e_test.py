import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ========== 1. 全局配置参数 ==========

ROI_IMAGE_PATH = "roi_pipe.png"

# --- 预处理 ---
MEDIAN_KSIZE = 5

# --- [Step 1] 梯度搜索 (粗定位) ---
RELATIVE_THRESH_TOP = 0.20
RELATIVE_THRESH_BOT = 0.25
MIN_ABS_THRESHOLD = 4.0
SEARCH_LIMIT_RATIO_TOP = 0.55
SEARCH_LIMIT_RATIO_BOT = 0.45

# --- [Step 2] 局部灰度细化 (精定位 - 新增) ---
REFINE_WIN_SIZE = 10         # 在粗边缘上下取多少像素做分析 (半径)
REFINE_INTENSITY_RATIO = 0.5 # 0.5 表示取(亮+暗)/2的位置，0.6表示更偏向亮部

# --- 数据清洗 ---
CLEAN_WINDOW_SIZE = 21
CLEAN_PIXEL_DIFF = 5.0

# --- 拟合 ---
POLY_DEGREE = 2             # 基准拟合阶数
RANSAC_SIGMA = 2.0

# --- 绘图 ---
VISUAL_SAVE_PATH = "final_refined_result.png"

# =================================

# 1. 滤波器函数 (按你要求先提供在这里，暂时不用)
def apply_savgol_filter(y, window_length=31, polyorder=3):
    """
    Savitzky-Golay 滤波器：用于保留波峰波谷细节的平滑。
    window_length: 必须是奇数，越大越平滑。
    polyorder: 拟合多项式阶数，通常 2 或 3。
    """
    if len(y) < window_length:
        return y
    if window_length % 2 == 0:
        window_length += 1
    return savgol_filter(y, window_length=window_length, polyorder=polyorder)

# 2. 亚像素细化函数 (核心改进)
def refine_edge_subpixel(img, x_list, y_list, is_top_edge):
    """
    结合了梯度定位(输入)和局部阈值(优化)。
    原理：在梯度边缘附近，寻找灰度值跨越 "局部动态阈值" 的精确位置。
    """
    h, w = img.shape
    y_refined = []
    x_kept = []

    # 半径
    r = REFINE_WIN_SIZE
    
    for i in range(len(x_list)):
        x = int(x_list[i])
        y0 = int(y_list[i])
        
        # 边界保护
        if y0 - r < 0 or y0 + r >= h:
            continue

        # 1. 提取局部灰度切片 (竖直方向)
        # 注意：这里我们不用中值滤波后的图，而是用原图(或轻微高斯)，保留灰度层次
        # 但为了抗噪，建议还是基于 img (已经做过预处理的图) 或者原图
        local_profile = img[y0-r : y0+r+1, x].astype(np.float32)
        
        # 2. 确定局部背景和前景
        # 上边缘：上方是背景(索引小)，下方是前景(索引大)
        # 下边缘：上方是前景，下方是背景
        if is_top_edge:
            val_bg = np.mean(local_profile[0:2])     # 局部最外侧的暗部
            val_fg = np.mean(local_profile[-2:])     # 局部最内侧的亮部
        else:
            val_fg = np.mean(local_profile[0:2])     # 局部最内侧的亮部
            val_bg = np.mean(local_profile[-2:])     # 局部最外侧的暗部

        # 3. 计算局部动态阈值 (50% 处)
        target_val = val_bg + (val_fg - val_bg) * REFINE_INTENSITY_RATIO
        
        # 4. 线性插值寻找 target_val 的位置
        # 我们在 local_profile 里找 val[j] < target <= val[j+1] (或反之)
        found_subpixel = False
        
        for j in range(len(local_profile) - 1):
            v1 = local_profile[j]
            v2 = local_profile[j+1]
            
            # 判断是否跨越了阈值 (不管是上升还是下降)
            if (v1 <= target_val <= v2) or (v2 <= target_val <= v1):
                if abs(v2 - v1) < 1e-3: 
                    offset = 0.5
                else:
                    # 线性插值公式
                    offset = (target_val - v1) / (v2 - v1)
                
                # y_refined = (y0 - r) + j + offset
                real_y = (y0 - r) + j + offset
                y_refined.append(real_y)
                x_kept.append(x)
                found_subpixel = True
                break
        
        # 如果局部对比度太差找不到过零点，就保留原梯度点
        if not found_subpixel:
            y_refined.append(y0)
            x_kept.append(x)

    return np.array(x_kept), np.array(y_refined)

# 3. 梯度搜索 (First Strike)
def find_edges_gradient_coarse(img):
    h, w = img.shape
    # 使用中值滤波去噪，利于梯度搜索
    img_smooth = cv2.medianBlur(img, MEDIAN_KSIZE)
    grad_y = cv2.Sobel(img_smooth, cv2.CV_64F, 0, 1, ksize=3)

    x_top, y_top = [], []
    x_bot, y_bot = [], []

    for x in range(w):
        col_grads = grad_y[:, x]
        
        # Top
        lim_t = int(h * SEARCH_LIMIT_RATIO_TOP)
        reg_t = col_grads[0:lim_t]
        th_t = max(MIN_ABS_THRESHOLD, (np.max(reg_t) if reg_t.size else 0) * RELATIVE_THRESH_TOP)
        for y in range(2, lim_t):
            if col_grads[y] > th_t:
                x_top.append(x); y_top.append(y); break
        
        # Bot
        start_b = int(h * (1 - SEARCH_LIMIT_RATIO_BOT))
        reg_b = col_grads[start_b:h]
        th_b = min(-MIN_ABS_THRESHOLD, (np.min(reg_b) if reg_b.size else 0) * RELATIVE_THRESH_BOT)
        for y in range(h - 3, start_b, -1):
            if col_grads[y] < th_b:
                x_bot.append(x); y_bot.append(y); break

    return np.array(x_top), np.array(y_top), np.array(x_bot), np.array(y_bot)

# 4. 数据清洗
def clean_edge_points(x, y, window=21, thresh=5.0):
    if len(x) < window: return x, y
    y_vals = np.asarray(y, dtype=np.float32)
    # 手动实现中值滤波
    pad_width = window // 2
    padded = np.pad(y_vals, (pad_width, pad_width), mode='edge')
    y_smooth = np.zeros_like(y_vals)
    for i in range(len(y_vals)):
        y_smooth[i] = np.median(padded[i : i + window])
    
    mask = np.abs(y_vals - y_smooth) < thresh
    return x[mask], y[mask]

# 5. 拟合
def robust_polyfit(x, y, deg, sigma_thresh=2.0, max_iter=5):
    x = np.asarray(x); y = np.asarray(y)
    mask = np.ones_like(x, dtype=bool)
    if len(x) < deg + 2: return None, mask, 0.0
    best_poly = None; final_std = 0.0
    
    for i in range(max_iter):
        if mask.sum() < deg + 2: break
        try:
            coef = np.polyfit(x[mask], y[mask], deg)
            best_poly = np.poly1d(coef)
        except: break
        y_pred = best_poly(x)
        resid = y - y_pred
        std = resid[mask].std()
        final_std = std
        if std < 1e-5: break 
        new_mask = np.abs(resid) < sigma_thresh * std
        if new_mask.sum() == mask.sum(): break
        mask = new_mask
    return best_poly, mask, final_std

def main():
    # 读图
    img = cv2.imread(ROI_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None: print("Err"); return
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Step 1: 梯度粗定位
    print("Step 1: 梯度粗定位...")
    xt, yt, xb, yb = find_edges_gradient_coarse(img)
    print(f"  粗定位点数: Top={len(xt)}, Bot={len(xb)}")

    # Step 2: 局部灰度细化 (新功能)
    # 这里我们传入 img，因为细化需要真实的灰度值
    print("Step 2: 亚像素级灰度细化...")
    xt, yt = refine_edge_subpixel(img, xt, yt, is_top_edge=True)
    xb, yb = refine_edge_subpixel(img, xb, yb, is_top_edge=False)
    print(f"  细化后点数: Top={len(xt)}, Bot={len(xb)}")

    # Step 3: 清洗突变噪点
    print("Step 3: 数据清洗...")
    xt, yt = clean_edge_points(xt, yt, CLEAN_WINDOW_SIZE, CLEAN_PIXEL_DIFF)
    xb, yb = clean_edge_points(xb, yb, CLEAN_WINDOW_SIZE, CLEAN_PIXEL_DIFF)

    # Step 4: 拟合
    poly_top, _, std_top = robust_polyfit(xt, yt, POLY_DEGREE, RANSAC_SIGMA)
    poly_bot, _, std_bot = robust_polyfit(xb, yb, POLY_DEGREE, RANSAC_SIGMA)

    # Step 5: 绘图
    # 画清洗后的有效点 (绿色)
    for x, y in zip(xt, yt):
        cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 0), -1)
    for x, y in zip(xb, yb):
        cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 0), -1)

    x_eval = np.arange(img.shape[1])
    if poly_top:
        print(f"上边缘 RMS: {std_top:.4f}")
        pts = np.column_stack((x_eval, poly_top(x_eval))).astype(np.int32)
        cv2.polylines(vis, [pts], False, (0, 0, 255), 2)
    if poly_bot:
        print(f"下边缘 RMS: {std_bot:.4f}")
        pts = np.column_stack((x_eval, poly_bot(x_eval))).astype(np.int32)
        cv2.polylines(vis, [pts], False, (255, 0, 0), 2)

    cv2.imwrite(VISUAL_SAVE_PATH, vis)
    print(f"完成，结果: {VISUAL_SAVE_PATH}")

if __name__ == "__main__":
    main()