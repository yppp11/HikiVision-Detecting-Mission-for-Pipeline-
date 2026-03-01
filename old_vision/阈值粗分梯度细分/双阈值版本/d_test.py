import cv2
import numpy as np
import matplotlib.pyplot as plt

# ========== 可配置参数开始 ==========

# 输入图像
ROI_IMAGE_PATH = "roi_pipe.png"
IMREAD_FLAG = cv2.IMREAD_GRAYSCALE

# 归一化到 8bit 的范围
NORM_MIN = 0
NORM_MAX = 255

# 分割相关（固定阈值 + 形态学）
SEG_THRESHOLD_TOP = 18       # 上边缘固定阈值
SEG_THRESHOLD_BOT = 27       # 下边缘固定阈值

# 高斯模糊
GAUSS_KERNEL_SIZE = (7, 7)
GAUSS_SIGMA = 1.0

# 形态学操作
MORPH_KERNEL_SIZE = (15, 15)
MORPH_ITER_CLOSE = 1        # 闭运算迭代次数
MORPH_ITER_OPEN = 1         # 开运算迭代次数

# 列投影 + 平滑
COL_MIN_WHITE_RATIO = 0.5   # 每列白点（前景）数量至少占整幅高度的比例，过小认为是噪声列
COL_MEDIAN_KSIZE = 3        # 列方向中值滤波窗口（奇数，<=1 表示不做）这里调大一点似乎是让曲率的变化来着，滤波值越大曲率变化越平缓
COL_GAUSS_SIGMA = 3.0       # 列方向高斯平滑 sigma（<=0 表示不做）

# 局部梯度细化
REFINE_USE_GRADIENT = False  # 是否启用局部梯度细化
REFINE_HALF_WIN_TOP = 3     # 上边缘：在粗边缘附近 ±N 像素内搜索梯度极值
REFINE_HALF_WIN_BOT = 3     # 下边缘：在粗边缘附近 ±N 像素内搜索梯度极值
REFINE_GAUSS_SIGMA =  2.0    # 对竖直梯度做一点高斯平滑，0 表示不平滑

# 多项式拟合
POLY_DEGREE = 3             # 上下边缘拟合多项式阶数
ROI_EDGES_FIT_PATH = "roi_edges_fit.png"

# 拟合误差可视化
ERR_VISUAL_PATH = "roi_edge_fit_error_visual.png"
ERR_SCALE = 5.0             # 误差绘制放大因子

# 轮廓重建 + 交互测厚度绘图参数
MPL_STYLE = "dark_background"
RECON_FIGSIZE = (10, 6)

PIPE_FILL_COLOR = "#00CED1"
TOP_EDGE_COLOR = "#FF6347"
BOTTOM_EDGE_COLOR = "#1E90FF"
CENTER_LINE_COLOR = "#32CD32"
CENTER_LINE_STYLE = "--"
CENTER_LINE_ALPHA = 0.7

VLINE_COLOR = "yellow"
POINT_COLOR = "yellow"
TEXT_BOX_FACE_COLOR = "#333333"
GRID_LINESTYLE = ":"
GRID_ALPHA = 0.4

PLOT_TITLE = "Pipe Deformation Reconstruction & Measurement"
PLOT_XLABEL = "X (pixels)"
PLOT_YLABEL = "Y (pixels)"

# ========== 可配置参数结束 ==========


def print_err_stats(name, err):
    """
    打印单条边缘的拟合误差统计信息。

    参数
    ----
    name : str
        边缘名称，用于区分“上边缘 / 下边缘”等。
    err : np.ndarray
        残差数组 (y_meas - y_fit)，允许为任意形状，内部按一维处理。
    """
    err = np.asarray(err, dtype=np.float64).ravel()
    err_abs = np.abs(err)

    print(f"\n{name}:")
    print(f"  均值误差           = {err.mean():.4f} 像素")
    print(f"  均方根(RMS)        = {np.sqrt(np.mean(err ** 2)):.4f} 像素")
    print(f"  标准差             = {err.std():.4f} 像素")
    print(f"  最大绝对误差       = {err_abs.max():.4f} 像素")
    print(f"  95%分位绝对误差    = {np.percentile(err_abs, 95):.4f} 像素")

def robust_polyfit(x, y, deg, sigma_thresh=3.0, max_iter=3):
    """
    简单的迭代 sigma 剔除异常点的多项式拟合。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.ones_like(x, dtype=bool)

    for _ in range(max_iter):
        coef = np.polyfit(x[mask], y[mask], deg)
        y_pred = np.polyval(coef, x)
        resid = y - y_pred
        sigma = resid[mask].std()

        new_mask = np.abs(resid) < sigma_thresh * sigma
        if new_mask.sum() == mask.sum():
            break  # 没有新的点被剔除

        mask = new_mask

    return np.poly1d(coef), mask

# ========== 步骤1：预处理 + 分割，得到管道主掩膜 ==========

def segment_and_get_mask(
    roi_gray,
    thr_val,
    gauss_ksize,
    gauss_sigma,
    morph_kernel_size,
    morph_iter_close,
    morph_iter_open,
    norm_min,
    norm_max,
):
    """
    对 ROI 做归一化、模糊、固定阈值分割、形态学，
    然后保留最大连通区域作为管道掩膜。

    参数
    ----
    roi_gray : np.ndarray
        输入灰度图，任意 bit 深度。
    thr_val : float or int
        固定阈值，>thr_val 认为是前景（管体）。
    gauss_ksize : (int, int)
        高斯模糊核大小。
    gauss_sigma : float
        高斯模糊 sigma。
    morph_kernel_size : (int, int)
        形态学操作核大小。
    morph_iter_close : int
        闭运算迭代次数。
    morph_iter_open : int
        开运算迭代次数。
    norm_min, norm_max : int
        归一化输出灰度范围，一般是 [0, 255]。

    返回
    ----
    binary_main : np.ndarray
        uint8 0/255 掩膜，仅保留最大连通区域。
    """
    h, w = roi_gray.shape[:2]

    # 归一化到 [norm_min, norm_max]，保证 8bit
    roi_8u = cv2.normalize(
        roi_gray, None,
        norm_min, norm_max,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # 高斯模糊去噪
    blur = cv2.GaussianBlur(roi_8u, gauss_ksize, gauss_sigma)

    # 固定阈值分割
    _, binary = cv2.threshold(blur, thr_val, 255, cv2.THRESH_BINARY)
    print(f"使用阈值: {thr_val}")

    # 闭运算 + 开运算，去孔洞和孤立噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
    binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=morph_iter_close)
    binary_clean = cv2.morphologyEx(binary_close, cv2.MORPH_OPEN, kernel, iterations=morph_iter_open)

    # 找最大连通轮廓，作为“管体”
    contours, _ = cv2.findContours(
        binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if len(contours) == 0:
        raise RuntimeError("分割后没有找到任何轮廓，请检查阈值 / 形态学参数。")

    areas = [cv2.contourArea(c) for c in contours]
    main_cnt = contours[int(np.argmax(areas))]
    print("轮廓数量:", len(contours), "最大轮廓面积:", max(areas))

    # 根据最大轮廓生成填充掩膜
    binary_main = np.zeros_like(binary_clean)
    cv2.drawContours(binary_main, [main_cnt], contourIdx=-1, color=255, thickness=-1)

    return binary_main


# ========== 辅助：1D 平滑函数 ==========

def _smooth_1d_median(arr, ksize):
    """
    简单 1D 中值滤波

    参数
    ----
    arr : np.ndarray
        一维数据。
    ksize : int
        滤波窗口长度，必须为奇数，<=1 时不做处理。

    返回
    ----
    out : np.ndarray
        滤波后的结果，float32。
    """
    if ksize is None or ksize <= 1:
        return np.asarray(arr, dtype=np.float32)

    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1

    arr = np.asarray(arr, dtype=np.float32)
    pad = ksize // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    out = np.empty_like(arr, dtype=np.float32)

    for i in range(arr.shape[0]):
        out[i] = np.median(padded[i : i + ksize])

    return out


def _smooth_1d_gauss(arr, sigma):
    """
    简单 1D 高斯平滑

    参数
    ----
    arr : np.ndarray
        一维数据。
    sigma : float
        高斯核 sigma，<=0 时不做处理。

    返回
    ----
    out : np.ndarray
        平滑后的结果，float32。
    """
    if sigma is None or sigma <= 0:
        return np.asarray(arr, dtype=np.float32)

    arr = np.asarray(arr, dtype=np.float32)
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    padded = np.pad(arr, (radius, radius), mode="edge")
    out = np.empty_like(arr, dtype=np.float32)

    ksize = kernel.size
    for i in range(arr.shape[0]):
        out[i] = np.sum(padded[i : i + ksize] * kernel)

    return out


# ========== 步骤2：列投影提取上下边缘粗略位置 ==========

def extract_edges_by_column(
    binary_main,
    col_min_white_ratio,
    median_ksize,
    gauss_sigma,
):
    """
    使用“二值掩膜 + 按列扫描”的方式提取粗略的上/下边缘。

    参数
    ----
    binary_main : np.ndarray
        uint8 0/255，前景像素表示管子区域。
    col_min_white_ratio : float
        每列前景像素至少占整幅高度多少比例，否则认为该列是噪声，直接丢弃。
    median_ksize : int
        对 y_top / y_bot 进行 1D 中值滤波的窗口长度。
    gauss_sigma : float
        对 y_top / y_bot 做 1D 高斯平滑的 sigma。

    返回
    ----
    x_top, y_top, x_bot, y_bot : np.ndarray
        粗略上下边缘点集，全部为 float32。
    """
    if binary_main.ndim == 3:
        binary_main = cv2.cvtColor(binary_main, cv2.COLOR_BGR2GRAY)

    h, w = binary_main.shape[:2]

    x_list = []
    y_top_list = []
    y_bot_list = []

    for x in range(w):
        # 找这一列里所有前景 y 坐标
        ys = np.where(binary_main[:, x] > 0)[0]
        if ys.size == 0:
            continue

        # 如果这一列前景太少（可能是噪声），直接跳过
        if ys.size < col_min_white_ratio * h:
            continue

        x_list.append(x)
        y_top_list.append(ys.min())
        y_bot_list.append(ys.max())

    if len(x_list) == 0:
        raise RuntimeError("按列投影没有找到任何有效列，请检查阈值 / COL_MIN_WHITE_RATIO。")

    x_arr = np.array(x_list, dtype=np.float32)
    y_top_arr = np.array(y_top_list, dtype=np.float32)
    y_bot_arr = np.array(y_bot_list, dtype=np.float32)

    # 1D 平滑（先中值再高斯）
    y_top_arr = _smooth_1d_median(y_top_arr, median_ksize)
    y_bot_arr = _smooth_1d_median(y_bot_arr, median_ksize)

    y_top_arr = _smooth_1d_gauss(y_top_arr, gauss_sigma)
    y_bot_arr = _smooth_1d_gauss(y_bot_arr, gauss_sigma)

    # x_top / x_bot 共用同一份 x_arr
    return x_arr, y_top_arr, x_arr.copy(), y_bot_arr


# ========== 步骤2.5：利用局部灰度梯度细化上下边缘点集 ==========

def refine_edges_with_local_gradient(
    roi_gray,
    x_top,
    y_top,
    x_bot,
    y_bot,
    half_win_top,
    half_win_bot,
    gauss_sigma,
):
    """
    在粗边缘附近用竖直梯度做局部细化（列方向独立处理）。

    上边缘：期望从外暗到内亮，优先选梯度为正的最大值；
    下边缘：期望从内亮到外暗，优先选梯度为负的最小值；
    如局部没有满足符号要求的点，则退而求其次选 |g| 最大。

    参数
    ----
    roi_gray : np.ndarray
        原始灰度图。
    x_top, y_top : np.ndarray
        粗略上边缘点集。
    x_bot, y_bot : np.ndarray
        粗略下边缘点集。
    half_win_top : int
        上边缘局部搜索窗半宽（像素）。
    half_win_bot : int
        下边缘局部搜索窗半宽（像素）。
    gauss_sigma : float
        对竖直梯度做高斯平滑的 sigma，<=0 时不平滑。

    返回
    ----
    y_top_ref, y_bot_ref : np.ndarray
        细化后的 y 坐标，float32。
    """
    roi_gray = roi_gray.astype(np.uint8)
    h, w = roi_gray.shape[:2]

    # 竖直方向梯度
    grad_y = cv2.Sobel(roi_gray, cv2.CV_32F, 0, 1, ksize=3)

    # 对竖直梯度做一点竖直方向的高斯平滑，降噪
    if gauss_sigma is not None and gauss_sigma > 0:
        ksize = int(3 * gauss_sigma) * 2 + 1
        if ksize < 3:
            ksize = 3
        # 只在 y 方向平滑，x 方向 kernel 大小为 1
        grad_y = cv2.GaussianBlur(grad_y, (1, ksize), gauss_sigma)

    y_top_ref = np.asarray(y_top, dtype=np.float32).copy()
    y_bot_ref = np.asarray(y_bot, dtype=np.float32).copy()

    half_win_top = int(max(1, half_win_top))
    half_win_bot = int(max(1, half_win_bot))

    # ---- 细化上边缘 ----
    for i in range(len(x_top)):
        x = int(round(x_top[i]))
        if x < 0 or x >= w:
            continue

        yc = int(round(y_top[i]))
        y0 = max(0, yc - half_win_top)
        y1 = min(h - 1, yc + half_win_top)

        line = grad_y[y0 : y1 + 1, x]

        # 寻找正梯度最大的位置
        idx_pos = int(np.argmax(line))
        if line[idx_pos] > 0:
            y_top_ref[i] = y0 + idx_pos
        else:
            # 若无明显正梯度，则选 |g| 最大
            idx_abs = int(np.argmax(np.abs(line)))
            y_top_ref[i] = y0 + idx_abs

    # ---- 细化下边缘 ----
    for i in range(len(x_bot)):
        x = int(round(x_bot[i]))
        if x < 0 or x >= w:
            continue

        yc = int(round(y_bot[i]))
        y0 = max(0, yc - half_win_bot)
        y1 = min(h - 1, yc + half_win_bot)

        line = grad_y[y0 : y1 + 1, x]

        # 寻找负梯度最小的位置
        idx_neg = int(np.argmin(line))
        if line[idx_neg] < 0:
            y_bot_ref[i] = y0 + idx_neg
        else:
            idx_abs = int(np.argmax(np.abs(line)))
            y_bot_ref[i] = y0 + idx_abs

    return y_top_ref, y_bot_ref



# ========== 步骤2.75：平滑化 ==========
def postprocess_edge(y_ref, y_base, max_shift, med_ksize, gauss_sigma):
    """
    y_ref   : 细化后的边缘
    y_base  : 细化前的粗边缘
    max_shift : 允许细化相对粗边缘的最大偏移（像素）
    """
    y_ref = np.asarray(y_ref, dtype=np.float32)
    y_base = np.asarray(y_base, dtype=np.float32)

    # 1) 限制细化偏移
    shift = y_ref - y_base
    shift = np.clip(shift, -max_shift, max_shift)
    y_lim = y_base + shift

    # 2) 再做一次 1D 平滑
    y_smooth = _smooth_1d_median(y_lim, med_ksize)
    y_smooth = _smooth_1d_gauss(y_smooth, gauss_sigma)

    return y_smooth

# ========== 步骤3：拟合 + 画 roi_edges_fit ==========

def fit_edges_and_draw(
    roi_gray,
    x_top,
    y_top,
    x_bot,
    y_bot,
    poly_degree,
    save_path,
):
    """
    对上下边缘做多项式拟合，并在原 ROI 上绘制：
      - 上下边缘散点；
      - 上下拟合曲线；
      - 中心线。

    参数
    ----
    roi_gray : np.ndarray
        原始灰度图。
    x_top, y_top, x_bot, y_bot : np.ndarray
        上下边缘点集。
    poly_degree : int
        多项式阶数。
    save_path : str
        结果可视化图像保存路径。

    返回
    ----
    poly_top, poly_bot : np.poly1d
        上下边缘拟合多项式。
    x_fit, y_top_fit, y_bot_fit, y_center : np.ndarray
        在每个整列 x 上评价后的浮点结果。
    """
    h_roi, w_roi = roi_gray.shape[:2]

    # 拟合（多项式：y = a_n x^n + ... + a_0）
    poly_top, mask_top = robust_polyfit(x_top, y_top, deg=poly_degree)
    poly_bot, mask_bot = robust_polyfit(x_bot, y_bot, deg=poly_degree)
    print("上边缘保留点数:", mask_top.sum(), "/", len(mask_top))
    print("下边缘保留点数:", mask_bot.sum(), "/", len(mask_bot))

    print("\n=== 多项式拟合系数 ===")
    print("上边缘:", poly_top.c)
    print("下边缘:", poly_bot.c)

    # 在每一列上评价多项式，得到浮点坐标
    x_fit = np.arange(0, w_roi, dtype=np.float32)
    y_top_fit = poly_top(x_fit).astype(np.float32)
    y_bot_fit = poly_bot(x_fit).astype(np.float32)
    y_center = 0.5 * (y_top_fit + y_bot_fit)

    # 可视化时需要裁剪并转成 int
    y_top_fit_clip = np.clip(y_top_fit, 0, h_roi - 1).astype(np.int32)
    y_bot_fit_clip = np.clip(y_bot_fit, 0, h_roi - 1).astype(np.int32)
    y_center_clip = np.clip(y_center, 0, h_roi - 1).astype(np.int32)

    vis = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

    # 原始散点
    for x, y in zip(x_top, y_top):
        cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 255), -1)   # 黄：上边粗点
    for x, y in zip(x_bot, y_bot):
        cv2.circle(vis, (int(x), int(y)), 1, (255, 255, 0), -1)   # 青：下边粗点

    # 拟合曲线
    pts_top_fit = np.stack(
        [x_fit.astype(np.int32), y_top_fit_clip], axis=1
    ).reshape(-1, 1, 2)
    pts_bot_fit = np.stack(
        [x_fit.astype(np.int32), y_bot_fit_clip], axis=1
    ).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts_top_fit], False, (0, 0, 255), 1)      # 红：上边拟合
    cv2.polylines(vis, [pts_bot_fit], False, (255, 0, 0), 1)      # 蓝：下边拟合

    # 中心线（绿）
    pts_center = np.stack(
        [x_fit.astype(np.int32), y_center_clip], axis=1
    ).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts_center], False, (0, 255, 0), 1)

    cv2.imwrite(save_path, vis)

    return poly_top, poly_bot, x_fit, y_top_fit, y_bot_fit, y_center


# ========== 步骤4：计算拟合误差并可视化 ==========

def compute_and_draw_error(
    roi_gray,
    x_top,
    y_top,
    x_bot,
    y_bot,
    poly_top,
    poly_bot,
    scale,
    save_path,
):
    """
    计算上下边缘的拟合残差，并在 ROI 中画出误差竖线。

    参数
    ----
    roi_gray : np.ndarray
        原始灰度图。
    x_top, y_top, x_bot, y_bot : np.ndarray
        上下边缘点集。
    poly_top, poly_bot : np.poly1d
        上下边缘拟合多项式。
    scale : float
        误差放大倍率，仅用于可视化。
    save_path : str
        保存路径。
    """
    h_roi, w_roi = roi_gray.shape[:2]

    # 浮点预测值
    y_top_pred = poly_top(x_top)
    y_bot_pred = poly_bot(x_bot)

    # 浮点残差
    err_top = y_top - y_top_pred
    err_bot = y_bot - y_bot_pred

    print("\n=== 拟合误差统计（单位：像素） ===")
    print_err_stats("上边缘", err_top)
    print_err_stats("下边缘", err_bot)

    vis_err = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

    # 上边缘误差竖线（红）
    for x, e in zip(x_top.astype(int), err_top):
        y = int(poly_top(x))
        y_draw = int(np.clip(y + e * scale, 0, h_roi - 1))
        cv2.line(vis_err, (x, y), (x, y_draw), (0, 0, 255), 1)

    # 下边缘误差竖线（蓝）
    for x, e in zip(x_bot.astype(int), err_bot):
        y = int(poly_bot(x))
        y_draw = int(np.clip(y + e * scale, 0, h_roi - 1))
        cv2.line(vis_err, (x, y), (x, y_draw), (255, 0, 0), 1)

    cv2.imwrite(save_path, vis_err)


# ========== 步骤5：轮廓重建 + 交互测厚度 ==========

def reconstruct_pipe_and_measure(
    roi_gray,
    x_fit,
    y_top_fit,
    y_bot_fit,
    mpl_style,
    figsize,
    pipe_fill_color,
    top_edge_color,
    bottom_edge_color,
    center_line_color,
    center_line_style,
    center_line_alpha,
    vline_color,
    point_color,
    text_box_face_color,
    grid_linestyle,
    grid_alpha,
    plot_title,
    plot_xlabel,
    plot_ylabel,
):
    """
    使用已经拟合好的上下边缘，重建管道轮廓，并在 Matplotlib 中
    以交互方式显示任意 x 位置处的厚度。

    参数
    ----
    roi_gray : np.ndarray
        原始灰度图（仅用于获取尺寸）。
    x_fit, y_top_fit, y_bot_fit : np.ndarray
        多项式在每一列 x 对应的浮点坐标。
    其余参数：绘图风格、颜色、标题等。
    """
    h_roi, w_roi = roi_gray.shape[:2]

    # 中心线（完全 float）
    y_center = (y_top_fit + y_bot_fit) / 2.0

    # Matplotlib 样式
    plt.style.use(mpl_style)
    fig, ax = plt.subplots(figsize=figsize)

    # 坐标系跟图像一致，y 轴向下
    ax.set_xlim(0, w_roi)
    ax.set_ylim(h_roi, 0)
    ax.set_title(plot_title, color="white", fontsize=14)
    ax.set_xlabel(plot_xlabel)
    ax.set_ylabel(plot_ylabel)

    # 填充管道区域
    ax.fill_between(x_fit, y_top_fit, y_bot_fit, color=pipe_fill_color, alpha=0.3, label="Pipe Body")

    # 上下边缘线
    ax.plot(x_fit, y_top_fit, color=top_edge_color, linewidth=1.5, label="Top Edge")
    ax.plot(x_fit, y_bot_fit, color=bottom_edge_color, linewidth=1.5, label="Bottom Edge")

    # 中心线
    ax.plot(
        x_fit,
        y_center,
        color=center_line_color,
        linestyle=center_line_style,
        linewidth=1,
        alpha=center_line_alpha,
        label="Center Line",
    )

    ax.legend(loc="upper right", facecolor="#333333", edgecolor="none")
    ax.grid(True, linestyle=grid_linestyle, alpha=grid_alpha)

    # 交互元素
    v_line = ax.axvline(x=0, color=vline_color, linestyle="-", linewidth=1, alpha=0.0)
    point_top, = ax.plot([], [], "o", color=point_color, markersize=4)
    point_bot, = ax.plot([], [], "o", color=point_color, markersize=4)

    text_template = "X: {:.0f}\nY_top: {:.3f}\nY_bot: {:.3f}\nThickness: {:.4f} px"
    text_annot = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        color="white",
        bbox=dict(
            facecolor=text_box_face_color,
            alpha=0.8,
            edgecolor="none",
            boxstyle="round,pad=0.5",
        ),
        verticalalignment="top",
    )

    # 鼠标移动回调：用 float 数组做精确计算，只用 int 做下标
    def on_mouse_move(event):
        if not event.inaxes or event.xdata is None:
            return

        x_mouse = event.xdata
        idx = int(np.clip(round(x_mouse), 0, len(x_fit) - 1))

        x_curr = float(x_fit[idx])
        curr_y_top = float(y_top_fit[idx])
        curr_y_bot = float(y_bot_fit[idx])
        thickness = abs(curr_y_bot - curr_y_top)

        # 更新垂直线和点
        v_line.set_xdata([x_curr])
        v_line.set_alpha(0.8)

        point_top.set_data([x_curr], [curr_y_top])
        point_bot.set_data([x_curr], [curr_y_bot])

        # 文本用高精度
        text_annot.set_text(text_template.format(x_curr, curr_y_top, curr_y_bot, thickness))

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

    print("\n交互窗口已打开。")
    print("  - 使用 Matplotlib 工具栏可以进行【缩放】和【平移】")
    print("  - 鼠标在图表中移动即可查看任意位置的厚度")

    plt.show()


# ========== 主流程 ==========

def main():
    """整条处理流水线入口。"""
    roi_gray = cv2.imread(ROI_IMAGE_PATH, IMREAD_FLAG)
    if roi_gray is None:
        print("读图失败:", ROI_IMAGE_PATH)
        return

    # ---- 1) 上边缘用较低阈值分割 ----
    binary_top = segment_and_get_mask(
        roi_gray=roi_gray,
        thr_val=SEG_THRESHOLD_TOP,
        gauss_ksize=GAUSS_KERNEL_SIZE,
        gauss_sigma=GAUSS_SIGMA,
        morph_kernel_size=MORPH_KERNEL_SIZE,
        morph_iter_close=MORPH_ITER_CLOSE,
        morph_iter_open=MORPH_ITER_OPEN,
        norm_min=NORM_MIN,
        norm_max=NORM_MAX,
    )

    # ---- 2) 下边缘用较高阈值分割 ----
    binary_bot = segment_and_get_mask(
        roi_gray=roi_gray,
        thr_val=SEG_THRESHOLD_BOT,
        gauss_ksize=GAUSS_KERNEL_SIZE,
        gauss_sigma=GAUSS_SIGMA,
        morph_kernel_size=MORPH_KERNEL_SIZE,
        morph_iter_close=MORPH_ITER_CLOSE,
        morph_iter_open=MORPH_ITER_OPEN,
        norm_min=NORM_MIN,
        norm_max=NORM_MAX,
    )

    # ---- 3) 列投影：上边缘只用 binary_top，下边缘只用 binary_bot ----
    x_top, y_top, _, _ = extract_edges_by_column(
        binary_main=binary_top,
        col_min_white_ratio=COL_MIN_WHITE_RATIO,
        median_ksize=COL_MEDIAN_KSIZE,
        gauss_sigma=COL_GAUSS_SIGMA,
    )

    _, _, x_bot, y_bot = extract_edges_by_column(
        binary_main=binary_bot,
        col_min_white_ratio=COL_MIN_WHITE_RATIO,
        median_ksize=COL_MEDIAN_KSIZE,
        gauss_sigma=COL_GAUSS_SIGMA,
    )

    # ---- 4) 梯度细化（可选） ----
    if REFINE_USE_GRADIENT:
        y_top_ref, y_bot_ref = refine_edges_with_local_gradient(
        roi_gray,
        x_top, y_top,
        x_bot, y_bot,
        half_win_top=REFINE_HALF_WIN_TOP,
        half_win_bot=REFINE_HALF_WIN_BOT,
        gauss_sigma=REFINE_GAUSS_SIGMA
    )

    # 上边缘一般很稳，限制 ±3 像素就够
        y_top_ref = postprocess_edge(y_top_ref, y_top, max_shift=3,
                                 med_ksize=5, gauss_sigma=0.5)

    # 下边缘允许稍大一点，比如 ±5 像素
        y_bot_ref = postprocess_edge(y_bot_ref, y_bot, max_shift=5,
                                 med_ksize=5, gauss_sigma=0.5)
    else:
        y_top_ref, y_bot_ref = y_top, y_bot


    # ---- 5) 多项式拟合 + 边缘/中心线绘制 ----
    poly_top, poly_bot, x_fit, y_top_fit, y_bot_fit, y_center = fit_edges_and_draw(
        roi_gray=roi_gray,
        x_top=x_top,
        y_top=y_top_ref,
        x_bot=x_bot,
        y_bot=y_bot_ref,
        poly_degree=POLY_DEGREE,
        save_path=ROI_EDGES_FIT_PATH,
    )

    # ---- 6) 拟合误差统计 + 误差可视化 ----
    compute_and_draw_error(
        roi_gray=roi_gray,
        x_top=x_top,
        y_top=y_top_ref,
        x_bot=x_bot,
        y_bot=y_bot_ref,
        poly_top=poly_top,
        poly_bot=poly_bot,
        scale=ERR_SCALE,
        save_path=ERR_VISUAL_PATH,
    )

    # ---- 7) 轮廓重建 + 交互测厚度 ----
    reconstruct_pipe_and_measure(
        roi_gray=roi_gray,
        x_fit=x_fit,
        y_top_fit=y_top_fit,
        y_bot_fit=y_bot_fit,
        mpl_style=MPL_STYLE,
        figsize=RECON_FIGSIZE,
        pipe_fill_color=PIPE_FILL_COLOR,
        top_edge_color=TOP_EDGE_COLOR,
        bottom_edge_color=BOTTOM_EDGE_COLOR,
        center_line_color=CENTER_LINE_COLOR,
        center_line_style=CENTER_LINE_STYLE,
        center_line_alpha=CENTER_LINE_ALPHA,
        vline_color=VLINE_COLOR,
        point_color=POINT_COLOR,
        text_box_face_color=TEXT_BOX_FACE_COLOR,
        grid_linestyle=GRID_LINESTYLE,
        grid_alpha=GRID_ALPHA,
        plot_title=PLOT_TITLE,
        plot_xlabel=PLOT_XLABEL,
        plot_ylabel=PLOT_YLABEL,
    )


if __name__ == "__main__":
    main()
