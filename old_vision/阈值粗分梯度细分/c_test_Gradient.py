import cv2
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 核心配置参数 (Configuration)
# ==============================================================================

# 1. 输入图像
ROI_IMAGE_PATH = "roi_pipe.png"

# 2. 预处理：双边滤波 (Bilateral Filter) —— 核心抗噪手段
# d: 像素邻域直径。越大越慢，但平滑范围越大。
# sigmaColor: 颜色空间标准差。越大，颜色差异大的像素也会互相混合（磨皮效果越强）。
# sigmaSpace: 坐标空间标准差。越大，越远的像素如果颜色相近也会参与平滑。
# 推荐值：(9, 75, 75) 是经典的磨皮参数，能把背景噪点抹平，同时保留强边缘。
BILATERAL_PARAMS = (9, 75, 75)

# 3. 梯度搜索范围 (Search Region)
# 为了避开管子内部极强的反光（通常在正中央），我们只在边缘附近搜索。
# 0.45 表示：上边缘只在前 45% 的高度找，下边缘只在后 45% 的高度找。
# 中间 10% 的区域直接被屏蔽，反光再强也干扰不到我们。
SEARCH_RATIO_TOP = 0.45 
SEARCH_RATIO_BOT = 0.45 

# 4. 梯度强度门槛 (Gradient Threshold)
# 即使是区域内的最大值，如果绝对值太小（比如纯黑背景里的微小波动），也不认为是边缘。
# 这个值设为 10.0 通常足够过滤掉双边滤波后的残留噪声。
MIN_GRADIENT_STRENGTH = 10.0

# 5. 拟合参数 (Fitting)
POLY_DEGREE = 2             # 2阶（抛物线）通常最适合弯管，3阶适合S型
RANSAC_SIGMA = 2.5          # 拟合时剔除离群点的宽容度（标准差倍数）

# 6. 可视化输出
VISUAL_SAVE_PATH = "result_final_bilateral.png"
MPL_STYLE = "dark_background"  # Matplotlib 绘图风格

# ==============================================================================
# 核心算法函数
# ==============================================================================

def robust_polyfit(x, y, deg, sigma_thresh=2.0, max_iter=5):
    """
    带抗噪能力的鲁棒多项式拟合 (RANSAC思路)
    自动剔除那些虽然被检测到、但明显偏离曲线轨迹的噪点。
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.ones_like(x, dtype=bool)
    
    # 点太少，不够拟合阶数，直接放弃
    if len(x) < deg + 2:
        return None, mask, 0.0

    best_poly = None
    best_std = 0.0
    
    for _ in range(max_iter):
        if mask.sum() < deg + 2:
            break
        try:
            # 拟合
            coef = np.polyfit(x[mask], y[mask], deg)
            best_poly = np.poly1d(coef)
        except:
            break

        # 计算残差
        y_pred = best_poly(x)
        resid = y - y_pred
        std = resid[mask].std()
        best_std = std
        
        # 如果拟合极好（甚至过拟合），就不用剔除了
        if std < 1e-5: break
            
        # 核心逻辑：保留误差在 N 倍标准差以内的点
        new_mask = np.abs(resid) < sigma_thresh * std
        
        # 没点被剔除，说明收敛了
        if new_mask.sum() == mask.sum():
            break
        mask = new_mask

    return best_poly, mask, best_std


def find_edges_argmax(img):
    """
    终极边缘提取逻辑：双边滤波 + 区域梯度最大值搜索 (Argmax)
    不依赖固定阈值，而是寻找局部最显著的变化。
    """
    h, w = img.shape
    
    # 1. 双边滤波：磨皮去噪，保留边缘锐度
    img_smooth = cv2.bilateralFilter(img, *BILATERAL_PARAMS)

    # 2. 计算竖直方向梯度 (Sobel Y)
    # CV_64F 允许负梯度存在（区分上边缘和下边缘的方向）
    grad_y = cv2.Sobel(img_smooth, cv2.CV_64F, 0, 1, ksize=3)

    x_top_list, y_top_list = [], []
    x_bot_list, y_bot_list = [], []

    # 计算搜索区域的截止行号
    limit_top = int(h * SEARCH_RATIO_TOP)
    limit_bot = int(h * (1 - SEARCH_RATIO_BOT))

    # 3. 逐列扫描
    for x in range(w):
        col_grads = grad_y[:, x]
        
        # === A. 找上边缘 (期望梯度 > 0) ===
        # 截取上半部分区域
        roi_top = col_grads[0:limit_top]
        
        if roi_top.size > 0:
            # 找该区域内梯度最大的位置 (Argmax)
            idx_max = np.argmax(roi_top) 
            val_max = roi_top[idx_max]
            
            # 只有当"最强梯度"也足够强时，才认为是边缘
            if val_max > MIN_GRADIENT_STRENGTH:
                x_top_list.append(x)
                y_top_list.append(idx_max) 

        # === B. 找下边缘 (期望梯度 < 0) ===
        # 截取下半部分区域
        roi_bot = col_grads[limit_bot:h]
        
        if roi_bot.size > 0:
            # 找该区域内梯度最"负"的位置 (Argmin)
            idx_min = np.argmin(roi_bot)
            val_min = roi_bot[idx_min]
            
            # 判断负梯度强度 (例如 val_min 为 -50，阈值为 10，-50 < -10 成立)
            if val_min < -MIN_GRADIENT_STRENGTH:
                x_bot_list.append(x)
                # 注意加上偏移量 limit_bot
                y_bot_list.append(limit_bot + idx_min)

    return (np.array(x_top_list), np.array(y_top_list), 
            np.array(x_bot_list), np.array(y_bot_list))


def show_interactive_plot(img, x_fit, y_top_fit, y_bot_fit):
    """
    交互式测量厚度显示 (Matplotlib)
    """
    h, w = img.shape[:2]
    plt.style.use(MPL_STYLE)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.set_title(f"Pipe Measurement (Degree={POLY_DEGREE})", color='white', fontsize=14)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0) # 图像坐标系 Y 轴向下
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    
    # 绘制区域
    ax.fill_between(x_fit, y_top_fit, y_bot_fit, color='#00CED1', alpha=0.3, label='Pipe Area')
    ax.plot(x_fit, y_top_fit, color='#FF6347', linewidth=2, label='Top Edge')
    ax.plot(x_fit, y_bot_fit, color='#1E90FF', linewidth=2, label='Bottom Edge')
    
    # 交互游标元素
    v_line = ax.axvline(x=w/2, color='yellow', linestyle='--', alpha=0.5)
    point_top, = ax.plot([], [], 'o', color='yellow', markersize=5)
    point_bot, = ax.plot([], [], 'o', color='yellow', markersize=5)
    
    # 信息框
    text_annot = ax.text(0.02, 0.95, "Move mouse to measure...", 
                         transform=ax.transAxes, color='white', verticalalignment='top',
                         bbox=dict(facecolor='#333333', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))

    def on_mouse_move(event):
        if not event.inaxes: return
        # 获取鼠标 X 坐标
        x_mouse = int(np.clip(event.xdata, 0, len(x_fit) - 1))
        
        # 获取该位置的拟合值
        curr_y_top = y_top_fit[x_mouse]
        curr_y_bot = y_bot_fit[x_mouse]
        thickness = abs(curr_y_bot - curr_y_top)
        
        # 更新绘图
        v_line.set_xdata([x_mouse])
        point_top.set_data([x_mouse], [curr_y_top])
        point_bot.set_data([x_mouse], [curr_y_bot])
        
        # 更新文字
        info_text = (f"X: {x_mouse}\n"
                     f"Y_Top: {curr_y_top:.2f}\n"
                     f"Y_Bot: {curr_y_bot:.2f}\n"
                     f"Thickness: {thickness:.3f} px")
        text_annot.set_text(info_text)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    plt.legend(loc='upper right', facecolor='#333333', edgecolor='none')
    print("\n[交互] 图表已打开，移动鼠标即可测量任意位置厚度。")
    plt.show()


# ==============================================================================
# 主流程 (Main)
# ==============================================================================

def main():
    print("=== 开始处理 ===")
    
    # 1. 读取图像
    img = cv2.imread(ROI_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[Error] 无法读取图像: {ROI_IMAGE_PATH}")
        return
    h, w = img.shape
    print(f"图像尺寸: {w}x{h}")

    # 2. 提取边缘点 (核心步骤)
    print("正在执行：双边滤波 + 梯度最大值搜索...")
    x_top, y_top, x_bot, y_bot = find_edges_argmax(img)
    print(f"  -> 上边缘检测点数: {len(x_top)}")
    print(f"  -> 下边缘检测点数: {len(x_bot)}")

    if len(x_top) < 10 or len(x_bot) < 10:
        print("[Warning] 检测到的边缘点过少，请检查图片或调整阈值！")
        return

    # 3. 多项式拟合
    print(f"正在拟合 (阶数={POLY_DEGREE})...")
    poly_top, mask_top, std_top = robust_polyfit(x_top, y_top, POLY_DEGREE, RANSAC_SIGMA)
    poly_bot, mask_bot, std_bot = robust_polyfit(x_bot, y_bot, POLY_DEGREE, RANSAC_SIGMA)
    
    print(f"  -> 上边缘 RMS 误差: {std_top:.4f} px (保留率 {mask_top.sum()/len(mask_top):.1%})")
    print(f"  -> 下边缘 RMS 误差: {std_bot:.4f} px (保留率 {mask_bot.sum()/len(mask_bot):.1%})")

    # 4. 生成拟合曲线坐标
    x_eval = np.arange(w)
    y_top_eval = poly_top(x_eval)
    y_bot_eval = poly_bot(x_eval)

    # 5. 绘制 OpenCV 结果图 (用于调试保存)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 画检测到的原始点 (小点)
    for x, y in zip(x_top, y_top):
        cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 255), -1) # 黄色：上边缘原始点
    for x, y in zip(x_bot, y_bot):
        cv2.circle(vis, (int(x), int(y)), 1, (255, 255, 0), -1) # 青色：下边缘原始点

    # 画拟合后的曲线 (粗线)
    pts_top = np.column_stack((x_eval, y_top_eval)).astype(np.int32)
    pts_bot = np.column_stack((x_eval, y_bot_eval)).astype(np.int32)
    cv2.polylines(vis, [pts_top], False, (0, 0, 255), 2) # 红色：上边缘拟合
    cv2.polylines(vis, [pts_bot], False, (255, 0, 0), 2) # 蓝色：下边缘拟合

    cv2.imwrite(VISUAL_SAVE_PATH, vis)
    print(f"结果图片已保存至: {VISUAL_SAVE_PATH}")

    # 6. 启动交互式界面
    show_interactive_plot(img, x_eval, y_top_eval, y_bot_eval)


if __name__ == "__main__":
    main()