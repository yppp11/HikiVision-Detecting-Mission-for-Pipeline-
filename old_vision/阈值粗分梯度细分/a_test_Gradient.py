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

# 分割相关
SEG_THRESHOLD = 18          # 固定阈值
SEG_ROW_RATIO = 0.8       # 行投影过滤阈值 = ratio * 图像宽度

# 高斯模糊
GAUSS_KERNEL_SIZE = (5, 5)
GAUSS_SIGMA = 1.0

# 形态学操作
MORPH_KERNEL_SIZE = (15, 15)
MORPH_ITER_CLOSE = 1        # 第一次闭运算迭代次数
MORPH_ITER_OPEN = 1         # 开运算迭代次数

MORPH_KERNEL2_SIZE = (5, 5)
MORPH2_ITER_CLOSE = 1       # 行投影之后的小核闭运算迭代次数

# 边缘拟合
POLY_DEGREE = 3             # 上下边缘拟合多项式阶数


# 厚度先验（软约束）
USE_THICKNESS_PRIOR = True   # 是否启用厚度先验
THICKNESS_PRIOR_PIX = None   # 如果为 None，则自动用当前平均厚度
# 梯度阈值（在 0~1 归一化灰度上）
GRAD_ABS_THR = 0.01          # 绝对梯度阈值，大约相当于 2~3 个灰度级
GRAD_REL_THR = 0.3           # 相对阈值：至少达到该列梯度最大值的 30%
# 先验惩罚系数：score = grad - PRIOR_LAMBDA*g_max*|y - y_prior|
PRIOR_LAMBDA = 0.02


# 拟合结果可视化
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

# ========== 公共函数 ==========

def collapse_edge(pts_in, is_top=True):
    """按 x 列压缩成单值边缘：上边取最小 y，下边取最大 y。"""
    xs_in = pts_in[:, 0].astype(int)
    ys_in = pts_in[:, 1].astype(float)
    x_unique = np.unique(xs_in)

    x_list, y_list = [], []
    for x in x_unique:
        ys_x = ys_in[xs_in == x]
        if is_top:
            y_sel = ys_x.min()
        else:
            y_sel = ys_x.max()
        x_list.append(x)
        y_list.append(y_sel)

    return np.array(x_list), np.array(y_list)


def print_err_stats(name, e):
    e_abs = np.abs(e)
    print(f"{name}:")
    print(f"  均值误差         = {e.mean():.4f} 像素")
    print(f"  均方根(RMS)     = {np.sqrt(np.mean(e ** 2)):.4f} 像素")
    print(f"  标准差           = {e.std():.4f} 像素")
    print(f"  最大绝对误差     = {e_abs.max():.4f} 像素")
    print(f"  95%分位绝对误差   = {np.percentile(e_abs, 95):.4f} 像素")


# ========== 步骤1：预处理 + 分割 + 主轮廓 ==========

def segment_and_find_contour(
    roi_gray,
    thr_val=SEG_THRESHOLD,
    ratio=SEG_ROW_RATIO,
    gauss_ksize=GAUSS_KERNEL_SIZE,
    gauss_sigma=GAUSS_SIGMA,
    morph_kernel_size=MORPH_KERNEL_SIZE,
    morph_iter_close=MORPH_ITER_CLOSE,
    morph_iter_open=MORPH_ITER_OPEN,
    morph_kernel2_size=MORPH_KERNEL2_SIZE,
    morph2_iter_close=MORPH2_ITER_CLOSE,
):
    """对 ROI 做分割、形态学、行投影过滤，返回主轮廓。"""
    h, w = roi_gray.shape[:2]

    # 归一化到 0~255，保证 8bit
    roi_8u = cv2.normalize(roi_gray, None, NORM_MIN, NORM_MAX, cv2.NORM_MINMAX).astype(np.uint8)

    # 高斯模糊去噪
    blur = cv2.GaussianBlur(roi_8u, gauss_ksize, gauss_sigma)

    # 固定阈值分割
    _, binary = cv2.threshold(blur, thr_val, 255, cv2.THRESH_BINARY)
    print("使用固定阈值:", thr_val)

    # 闭运算 + 开运算
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
    binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=morph_iter_close)
    binary_clean = cv2.morphologyEx(binary_close, cv2.MORPH_OPEN, kernel, iterations=morph_iter_open)

    # 行投影过滤，只保留“几乎整行是白”的行
    row_sums = binary_clean.sum(axis=1) / 255.0
    threshold = ratio * w
    row_mask = row_sums >= threshold

    binary_main = np.zeros_like(binary_clean)
    binary_main[row_mask, :] = binary_clean[row_mask, :]

    # 小核再闭一下，让边界更平顺
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel2_size)
    binary_main = cv2.morphologyEx(binary_main, cv2.MORPH_CLOSE, kernel2, iterations=morph2_iter_close)

    # 找主轮廓
    contours, _ = cv2.findContours(
        binary_main, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if len(contours) == 0:
        raise RuntimeError("行投影过滤后没有找到轮廓，请检查阈值/ratio。")

    areas = [cv2.contourArea(c) for c in contours]
    main_cnt = contours[int(np.argmax(areas))]
    print("轮廓数量:", len(contours), "最大轮廓面积:", max(areas))

    return main_cnt, binary_main


# ========== 步骤2：提取上下边缘点集 ==========

def extract_top_bottom_edges(main_cnt):
    """从主轮廓中拆出上边缘和下边缘的 y(x) 点集。"""
    pts = main_cnt.reshape(-1, 2)
    xs = pts[:, 0]
    ys = pts[:, 1]

    y_min, y_max = ys.min(), ys.max()
    y_mid = 0.5 * (y_min + y_max)
    print("y_min, y_mid, y_max:", y_min, y_mid, y_max)

    top_pts = pts[ys < y_mid]
    bottom_pts = pts[ys >= y_mid]
    print("上半圈点数:", top_pts.shape[0], "下半圈点数:", bottom_pts.shape[0])

    x_top, y_top = collapse_edge(top_pts, is_top=True)
    x_bot, y_bot = collapse_edge(bottom_pts, is_top=False)
    print("上边缘采样列数:", len(x_top), "下边缘采样列数:", len(x_bot))

    return x_top, y_top, x_bot, y_bot


# ========== 步骤2.5：利用局部灰度梯度细化上下边缘点集 ==========
def refine_edges_with_local_gradient(
    roi_gray,
    x_top, y_top,
    x_bot, y_bot,
    half_window_top=4,
    half_window_bot=4,
    use_thickness_prior=False,
    thickness_prior_pix=None,
):
    """
    利用局部灰度梯度，对上下边缘做一次细化。
    - 上边缘：只找“由暗到亮”的正梯度峰；
    - 下边缘：只找“由亮到暗”的负梯度峰；
    - 可选：使用厚度先验，把下边缘搜索窗口中心约束在 y_top + thickness_prior_pix 附近。
    输入/输出 y_* 都是 float，可达亚像素。
    """
    h, w = roi_gray.shape[:2]

    roi_f = cv2.normalize(roi_gray, None, 0, 1.0,
                          cv2.NORM_MINMAX).astype(np.float32)

    y_top_ref = y_top.astype(np.float32).copy()
    y_bot_ref = y_bot.astype(np.float32).copy()

    if use_thickness_prior:
        if thickness_prior_pix is None:
            thickness_prior_pix = float(np.mean(y_bot_ref - y_top_ref))
        print(f"使用厚度先验: {thickness_prior_pix:.3f} 像素")

    top_map = {int(x): float(y) for x, y in zip(x_top.astype(int), y_top_ref)}

    # --- 基础版：老的 abs(diff) 搜索，中心在 y_center 一带 ---
    def refine_basic(col, y_center, half_window):
        y_center = int(round(y_center))
        y0 = max(1, y_center - half_window)
        y1 = min(h - 2, y_center + half_window)
        if y1 <= y0 + 2:
            return float(y_center)

        profile = col[y0:y1 + 1]
        grad = np.abs(np.diff(profile))

        k = int(np.argmax(grad))
        offset = 0.0
        if 0 < k < len(grad) - 1:
            g0, g1, g2 = grad[k - 1], grad[k], grad[k + 1]
            denom = (g0 - 2.0 * g1 + g2)
            if abs(denom) > 1e-6:
                offset = 0.5 * (g0 - g2) / denom

        return float(y0 + k + offset)

    # --- 带方向和先验的版本，失败时回退到 refine_basic ---
    def refine_oriented(col, y0_int, half_window, is_top, y_center_override=None):
        if y_center_override is not None:
            y_center = float(y_center_override)
        else:
            y_center = float(y0_int)

        y_center_i = int(round(y_center))
        y0 = max(1, y_center_i - half_window)
        y1 = min(h - 2, y_center_i + half_window)
        if y1 <= y0 + 2:
            return float(y_center_i)

        profile = col[y0:y1 + 1]
        diff = np.diff(profile)

        if is_top:
            grad = diff          # 上边缘：暗->亮
        else:
            grad = -diff         # 下边缘：亮->暗

        grad[grad < 0] = 0.0
        if np.all(grad <= 0):
            # 如果方向梯度根本找不到有效峰，就退回到旧逻辑
            return refine_basic(col, y_center, half_window)

        k = int(np.argmax(grad))
        offset = 0.0
        if 0 < k < len(grad) - 1:
            g0, g1, g2 = grad[k - 1], grad[k], grad[k + 1]
            denom = (g0 - 2.0 * g1 + g2)
            if abs(denom) > 1e-6:
                offset = 0.5 * (g0 - g2) / denom

        return float(y0 + k + offset)

    # 上边缘：中心用粗 y_top，主要靠正梯度
    for i, (x_i, y_i) in enumerate(zip(x_top.astype(int), y_top.astype(int))):
        if 0 <= x_i < w:
            col = roi_f[:, x_i]
            y_top_ref[i] = refine_oriented(col, y_i, half_window_top, is_top=True)

    # 下边缘：优先用 “y_top_ref + 厚度先验” 做中心，失败再回到粗 y_bot 一带
    for i, (x_i, y_i) in enumerate(zip(x_bot.astype(int), y_bot.astype(int))):
        if 0 <= x_i < w:
            col = roi_f[:, x_i]
            if use_thickness_prior and int(x_i) in top_map:
                y_center_prior = top_map[int(x_i)] + float(thickness_prior_pix)
            else:
                y_center_prior = float(y_i)

            # 先按先验中心找一次
            y_ref = refine_oriented(col, y_i, half_window_bot, is_top=False,
                                    y_center_override=y_center_prior)

            # 如果实在不放心，可以再检查一下厚度是否离先验太远，超出就再用粗 y_bot 重新 refine_basic 一次
            # 这里为了简单先不加第二重判断

            y_bot_ref[i] = y_ref

    return y_top_ref, y_bot_ref

# ========== 步骤3：拟合 + 画 roi_edges_fit ==========

def fit_edges_and_draw(
    roi_gray,
    x_top,
    y_top,
    x_bot,
    y_bot,
    deg=POLY_DEGREE,
    save_path=ROI_EDGES_FIT_PATH,
):
    """拟合上下边缘，多项式 y=f(x)，同时生成 roi_edges_fit 图像，并保存浮点拟合结果。"""
    h_roi, w_roi = roi_gray.shape[:2]

    # 拟合（多项式：y = a3 x^3 + a2 x^2 + a1 x + a0）
    coef_top = np.polyfit(x_top, y_top, deg=deg)
    coef_bot = np.polyfit(x_bot, y_bot, deg=deg)
    poly_top = np.poly1d(coef_top)
    poly_bot = np.poly1d(coef_bot)
    print("上边缘拟合系数:", coef_top)
    print("下边缘拟合系数:", coef_bot)

    # 在每一列上评价多项式，这里都是浮点数（亚像素级）
    x_fit = np.arange(0, w_roi, dtype=np.float32)
    y_top_fit = poly_top(x_fit).astype(np.float32)
    y_bot_fit = poly_bot(x_fit).astype(np.float32)

    # 中心线（同样先保留浮点）
    y_center = 0.5 * (y_top_fit + y_bot_fit).astype(np.float32)

    # ===== 以下是为了可视化，需要把坐标裁剪并转成 int =====
    y_top_fit_clip = np.clip(y_top_fit, 0, h_roi - 1).astype(np.int32)
    y_bot_fit_clip = np.clip(y_bot_fit, 0, h_roi - 1).astype(np.int32)
    y_center_clip = np.clip(y_center, 0, h_roi - 1).astype(np.int32)

    vis = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

    # 原始散点
    for x, y in zip(x_top, y_top):
        cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 255), -1)   # 黄：上边
    for x, y in zip(x_bot, y_bot):
        cv2.circle(vis, (int(x), int(y)), 1, (255, 255, 0), -1)   # 青：下边

    # 拟合曲线（用 int 坐标画出来，肉眼看会有一点锯齿很正常）
    pts_top_fit = np.stack(
        [x_fit.astype(np.int32), y_top_fit_clip], axis=1
    ).reshape(-1, 1, 2)
    pts_bot_fit = np.stack(
        [x_fit.astype(np.int32), y_bot_fit_clip], axis=1
    ).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts_top_fit], False, (0, 0, 255), 1)      # 红：上拟合
    cv2.polylines(vis, [pts_bot_fit], False, (255, 0, 0), 1)      # 蓝：下拟合

    # 中心线（绿）
    pts_center = np.stack(
        [x_fit.astype(np.int32), y_center_clip], axis=1
    ).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts_center], False, (0, 255, 0), 1)

    cv2.imwrite(save_path, vis)

    # 返回浮点结果用于后续计算
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
    scale=ERR_SCALE,
    save_path=ERR_VISUAL_PATH,
):
    """计算拟合残差并画误差竖线，输出 roi_edge_fit_error_visual，同时保存浮点残差数据。"""
    h_roi, w_roi = roi_gray.shape[:2]

    # 浮点预测值
    y_top_pred = poly_top(x_top)
    y_bot_pred = poly_bot(x_bot)

    # 浮点残差（亚像素）
    err_top = y_top - y_top_pred
    err_bot = y_bot - y_bot_pred

    print_err_stats("上边缘", err_top)
    print_err_stats("下边缘", err_bot)

    vis_err = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

    # 下面这部分只是为了画图，必须把坐标转成 int
    for x, e in zip(x_top.astype(int), err_top):
        y = int(poly_top(x))
        y_draw = int(np.clip(y + e * scale, 0, h_roi - 1))
        cv2.line(vis_err, (x, y), (x, y_draw), (0, 0, 255), 1)   # 红：上边误差

    for x, e in zip(x_bot.astype(int), err_bot):
        y = int(poly_bot(x))
        y_draw = int(np.clip(y + e * scale, 0, h_roi - 1))
        cv2.line(vis_err, (x, y), (x, y_draw), (255, 0, 0), 1)   # 蓝：下边误差

    cv2.imwrite(save_path, vis_err)

    # 如果后面需要，也可以把 err_top, err_bot 返回
    # return err_top, err_bot


# ========== 步骤5：轮廓重建 + 交互测厚度 ==========

def reconstruct_pipe_and_measure(
    roi_gray,
    x_fit,
    y_top_fit,
    y_bot_fit,
    mpl_style=MPL_STYLE,
    figsize=RECON_FIGSIZE,
):
    """
    使用已经计算好的浮点拟合结果进行轮廓重建与交互测厚度。
    x_fit, y_top_fit, y_bot_fit 均为 float（亚像素），画图时只在显示层面用 int。
    """
    h_roi, w_roi = roi_gray.shape[:2]

    # 中心线（完全 float）
    y_center = (y_top_fit + y_bot_fit) / 2.0

    # --- Matplotlib 绘图设置 ---
    plt.style.use(mpl_style)
    fig, ax = plt.subplots(figsize=figsize)

    # 坐标系跟图像一致，y 轴向下
    ax.set_xlim(0, w_roi)
    ax.set_ylim(h_roi, 0)
    ax.set_title(PLOT_TITLE, color='white', fontsize=14)
    ax.set_xlabel(PLOT_XLABEL)
    ax.set_ylabel(PLOT_YLABEL)

    # 填充管道区域
    ax.fill_between(x_fit, y_top_fit, y_bot_fit, color=PIPE_FILL_COLOR, alpha=0.3, label='Pipe Body')

    # 上下边缘线（浮点曲线）
    ax.plot(x_fit, y_top_fit, color=TOP_EDGE_COLOR, linewidth=1.5, label='Top Edge')
    ax.plot(x_fit, y_bot_fit, color=BOTTOM_EDGE_COLOR, linewidth=1.5, label='Bottom Edge')

    # 中心线
    ax.plot(
        x_fit,
        y_center,
        color=CENTER_LINE_COLOR,
        linestyle=CENTER_LINE_STYLE,
        linewidth=1,
        alpha=CENTER_LINE_ALPHA,
        label='Center Line'
    )

    ax.legend(loc='upper right', facecolor='#333333', edgecolor='none')
    ax.grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)

    # --- 交互元素 ---
    v_line = ax.axvline(x=0, color=VLINE_COLOR, linestyle='-', linewidth=1, alpha=0.0)
    point_top, = ax.plot([], [], 'o', color=POINT_COLOR, markersize=4)
    point_bot, = ax.plot([], [], 'o', color=POINT_COLOR, markersize=4)

    text_template = "X: {:.0f}\nY_top: {:.3f}\nY_bot: {:.3f}\nThickness: {:.4f} px"
    text_annot = ax.text(
        0.02, 0.95, "", transform=ax.transAxes, color='white',
        bbox=dict(
            facecolor=TEXT_BOX_FACE_COLOR,
            alpha=0.8,
            edgecolor='none',
            boxstyle='round,pad=0.5'
        ),
        verticalalignment='top'
    )

    # 鼠标移动回调：这里用 float 数组做精确计算，只用 int 做“索引下标”
    def on_mouse_move(event):
        if not event.inaxes or event.xdata is None:
            return

        x_mouse = event.xdata

        # 根据鼠标的 x 找到最近的索引
        idx = int(np.clip(round(x_mouse), 0, len(x_fit) - 1))

        x_curr = float(x_fit[idx])
        curr_y_top = float(y_top_fit[idx])
        curr_y_bot = float(y_bot_fit[idx])
        thickness = abs(curr_y_bot - curr_y_top)  # 完整的浮点厚度（亚像素）

        # 更新垂直线和点
        v_line.set_xdata([x_curr])
        v_line.set_alpha(0.8)

        point_top.set_data([x_curr], [curr_y_top])
        point_bot.set_data([x_curr], [curr_y_bot])

        # 文本用高精度
        text_annot.set_text(text_template.format(x_curr, curr_y_top, curr_y_bot, thickness))

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

    print("交互窗口已打开。")
    print("使用 Matplotlib 工具栏可以进行【缩放】和【平移】。")
    print("鼠标在图表中移动即可查看任意位置的厚度（亚像素精度）。")
    plt.show()


# ========== 主流程 ==========

def main():
    roi_gray = cv2.imread(ROI_IMAGE_PATH, IMREAD_FLAG)
    if roi_gray is None:
        raise FileNotFoundError(f"找不到 {ROI_IMAGE_PATH}，请确认路径。")

    # 1. 分割 + 主轮廓（使用前面的全局配置作为默认参数）
    main_cnt, _ = segment_and_find_contour(roi_gray)

    # 2. 上下边缘点集（整数像素级）
    x_top, y_top, x_bot, y_bot = extract_top_bottom_edges(main_cnt)

    # 2.5 局部梯度细化边缘（得到浮点 y）
    y_top_ref, y_bot_ref = refine_edges_with_local_gradient(
        roi_gray,
        x_top, y_top,
        x_bot, y_bot,
        use_thickness_prior=USE_THICKNESS_PRIOR,
        thickness_prior_pix=THICKNESS_PRIOR_PIX,  # None 时自动用平均厚度
    )
    # 3. 拟合并画 roi_edges_fit，这里用的是细化后的 y 值
    poly_top, poly_bot, x_fit, y_top_fit, y_bot_fit, y_center = \
        fit_edges_and_draw(roi_gray, x_top, y_top_ref, x_bot, y_bot_ref)

    # 4. 误差计算 + 可视化
    compute_and_draw_error(roi_gray, x_top, y_top_ref, x_bot, y_bot_ref, poly_top, poly_bot)

    # 5. 轮廓重建 + 鼠标测厚度（用 float 数组）
    reconstruct_pipe_and_measure(roi_gray, x_fit, y_top_fit, y_bot_fit)


if __name__ == "__main__":
    main()
