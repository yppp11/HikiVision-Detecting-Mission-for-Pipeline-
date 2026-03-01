import cv2
import numpy as np

# 1. 读取 ROI（上一阶段保存的 roi_pipe.png）
roi = cv2.imread('roi_pipe.png', cv2.IMREAD_GRAYSCALE)

# 2. 归一化到 0~255，保证是 8bit
roi_8u = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 3. 轻微高斯模糊，去高频噪声
blur = cv2.GaussianBlur(roi_8u, (5, 5), 1.0)

# 4.  阈值分割：亮管子 -> 255，背景 -> 0
thr_val, binary = cv2.threshold(
    blur, 25, 255,
    cv2.THRESH_BINARY
)
print("使用固定阈值:", thr_val)

# 5. 形态学闭运算 + 开运算，填孔、去小斑点，让管子成一个干净大块
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
binary_clean = cv2.morphologyEx(binary_close, cv2.MORPH_OPEN, kernel, iterations=1)

binary = binary_clean
h, w = binary.shape

# 1. 按行统计白像素数量
row_sums = binary.sum(axis=1) / 255.0   # 每行里有多少个 255

# 2. 只保留“几乎整行都是白”的那些行
ratio = 0.8   # 可以试 0.85~0.95 调整
threshold = ratio * w
row_mask = row_sums >= threshold

# 3. 构造新的干净掩膜：只有满足条件的行保留，其他行清零
binary_main = np.zeros_like(binary)
binary_main[row_mask, :] = binary[row_mask, :]

# （可选）再做一次小核的闭运算，把上下边界补平滑一点
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
binary_main = cv2.morphologyEx(binary_main, cv2.MORPH_CLOSE, kernel2, iterations=1)

cv2.imwrite('roi_binary_main.png', binary_main)

# 4. 在这个干净掩膜上寻找主体轮廓
contours, _ = cv2.findContours(
    binary_main,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE
)

if len(contours) == 0:
    raise RuntimeError("行投影过滤后没有找到轮廓，请检查 ratio 阈值。")

# 理论上这里应该只剩 1 条轮廓，如果多了就取面积最大的
areas = [cv2.contourArea(c) for c in contours]
main_cnt = contours[int(np.argmax(areas))]
print("轮廓数量:", len(contours), "最大轮廓面积:", max(areas))

# 5. 画在原 ROI 上检查
roi = cv2.imread('roi_pipe.png', cv2.IMREAD_GRAYSCALE)
overlay = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
cv2.drawContours(overlay, [main_cnt], -1, (0, 0, 255), 2)
cv2.imwrite('roi_contour_overlay_main.png', overlay)

cv2.imwrite('roi_binary_mask.png', binary)


# ======================
# 步骤3：从主轮廓中分离上下边缘，并规整成 y(x)
# 假设 main_cnt 和 roi 已经按前面代码得到
# ======================

# 1. 将 main_cnt 展平成 (N,2) 的点集
pts = main_cnt.reshape(-1, 2)  # (N, 2)
xs = pts[:, 0]
ys = pts[:, 1]

# 2. 用中线 y_mid 把轮廓分成上、下两部分
y_min, y_max = ys.min(), ys.max()
y_mid = 0.5 * (y_min + y_max)
print("y_min, y_mid, y_max:", y_min, y_mid, y_max)

top_pts = pts[ys < y_mid]      # 上半圈点
bottom_pts = pts[ys >= y_mid]  # 下半圈点

print("上半圈点数:", top_pts.shape[0], "下半圈点数:", bottom_pts.shape[0])

# 3. 按 x 列“压缩”成单值边缘：上边取该列最小 y，下边取该列最大 y
def collapse_edge(pts_in, is_top=True):
    xs_in = pts_in[:, 0].astype(int)
    ys_in = pts_in[:, 1].astype(float)
    x_unique = np.unique(xs_in)

    x_list = []
    y_list = []
    for x in x_unique:
        ys_x = ys_in[xs_in == x]
        if is_top:
            y_sel = ys_x.min()   # 上边缘：最小 y
        else:
            y_sel = ys_x.max()   # 下边缘：最大 y
        x_list.append(x)
        y_list.append(y_sel)

    return np.array(x_list), np.array(y_list)

x_top, y_top = collapse_edge(top_pts, is_top=True)
x_bot, y_bot = collapse_edge(bottom_pts, is_top=False)

print("上边缘采样列数:", len(x_top), "下边缘采样列数:", len(x_bot))

# 4. 可视化：在 ROI 上画出上下边缘曲线
roi_vis = cv2.imread('roi_pipe.png', cv2.IMREAD_GRAYSCALE)
roi_vis = cv2.cvtColor(roi_vis, cv2.COLOR_GRAY2BGR)

# 组装成 polyline 需要的形状 (N,1,2)
pts_top_draw = np.stack([x_top, y_top], axis=1).astype(np.int32).reshape(-1, 1, 2)
pts_bot_draw = np.stack([x_bot, y_bot], axis=1).astype(np.int32).reshape(-1, 1, 2)

# 上边缘画成红色，下边缘画成蓝色
cv2.polylines(roi_vis, [pts_top_draw], isClosed=False, color=(0, 0, 255), thickness=1)
cv2.polylines(roi_vis, [pts_bot_draw], isClosed=False, color=(255, 0, 0), thickness=1)

cv2.imwrite('roi_top_bottom_edges.png', roi_vis)

# ======================
# 步骤4：曲线拟合
# ======================
# 1. 选择多项式阶数
deg = 3   # 先用 3 阶，多了容易过拟合噪声

# 2. 分别拟合上下边缘：y = f(x)
coef_top = np.polyfit(x_top, y_top, deg=deg)
coef_bot = np.polyfit(x_bot, y_bot, deg=deg)

poly_top = np.poly1d(coef_top)
poly_bot = np.poly1d(coef_bot)

# 3. 在整幅 ROI 宽度上取一组连续的 x，用来画平滑曲线
roi_gray = cv2.imread('roi_pipe.png', cv2.IMREAD_GRAYSCALE)
h_roi, w_roi = roi_gray.shape[:2]
x_fit = np.arange(0, w_roi)

y_top_fit = poly_top(x_fit)
y_bot_fit = poly_bot(x_fit)

# 4. 把拟合结果裁到图像范围内，并转 int
y_top_fit_clip = np.clip(y_top_fit, 0, h_roi - 1).astype(np.int32)
y_bot_fit_clip = np.clip(y_bot_fit, 0, h_roi - 1).astype(np.int32)

# 5. 在图像上画原始散点 + 拟合曲线
vis = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

# 原始散点（颜色稍淡一点，便于和拟合线区分）
for x, y in zip(x_top, y_top):
    cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 255), -1)   # 黄色点：原始上边
for x, y in zip(x_bot, y_bot):
    cv2.circle(vis, (int(x), int(y)), 1, (255, 255, 0), -1)   # 青色点：原始下边

# 拟合曲线
pts_top_fit = np.stack([x_fit, y_top_fit_clip], axis=1).reshape(-1, 1, 2)
pts_bot_fit = np.stack([x_fit, y_bot_fit_clip], axis=1).reshape(-1, 1, 2)

cv2.polylines(vis, [pts_top_fit], isClosed=False, color=(0, 0, 255), thickness=1)   # 红线：拟合上边
cv2.polylines(vis, [pts_bot_fit], isClosed=False, color=(255, 0, 0), thickness=1)   # 蓝线：拟合下边

cv2.imwrite('roi_edges_fit.png', vis)

print("上边缘拟合系数:", coef_top)
print("下边缘拟合系数:", coef_bot)

# ======================
# 步骤5：误差计算
# ======================
# 1. 计算残差（像素）
y_top_pred = poly_top(x_top)
y_bot_pred = poly_bot(x_bot)

err_top = y_top - y_top_pred
err_bot = y_bot - y_bot_pred

# 2. 定义一个打印统计量的小函数
def print_err_stats(name, e):
    e_abs = np.abs(e)
    print(f"{name}:")
    print(f"  均值误差       = {e.mean():.4f} 像素")
    print(f"  均方根(RMS)   = {np.sqrt(np.mean(e**2)):.4f} 像素")
    print(f"  标准差         = {e.std():.4f} 像素")
    print(f"  最大绝对误差   = {e_abs.max():.4f} 像素")
    print(f"  95%分位绝对误差 = {np.percentile(e_abs, 95):.4f} 像素")

print_err_stats("上边缘", err_top)
print_err_stats("下边缘", err_bot)

# 3. 可选：把误差在图像上画出来，直观看看哪里误差大
roi_gray = cv2.imread("roi_pipe.png", cv2.IMREAD_GRAYSCALE)
h_roi, w_roi = roi_gray.shape
vis_err = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

# 为了看得清，把误差放大若干倍画成小竖线
scale = 5.0  # 每 1 像素误差画成 5 像素长的线

# 上边缘误差（红色）
for x, e in zip(x_top.astype(int), err_top):
    y = int(poly_top(x))
    y_draw = int(np.clip(y + e * scale, 0, h_roi - 1))
    cv2.line(vis_err, (x, y), (x, y_draw), (0, 0, 255), 1)

# 下边缘误差（蓝色）
for x, e in zip(x_bot.astype(int), err_bot):
    y = int(poly_bot(x))
    y_draw = int(np.clip(y + e * scale, 0, h_roi - 1))
    cv2.line(vis_err, (x, y), (x, y_draw), (255, 0, 0), 1)

cv2.imwrite("roi_edge_fit_error_visual.png", vis_err)

# ======================
# 步骤6：中心线拟合
# ======================
# 1. 读入前一步生成的 roi_edges_fit 图像
img = cv2.imread('roi_edges_fit.png')   # 里面已经有上下边缘的拟合线和散点
h, w = img.shape[:2]

# 2. 在整幅宽度上采样 x，并计算上下边缘拟合值
x_c = np.arange(0, w, dtype=np.float32)

y_top_fit = poly_top(x_c)
y_bot_fit = poly_bot(x_c)

# 3. 中心线：上下拟合的平均
y_center = 0.5 * (y_top_fit + y_bot_fit)

# 4. 裁到图像范围并转 int
y_center_clip = np.clip(y_center, 0, h - 1).astype(np.int32)

# 5. 组装成 polyline 所需的形状 (N,1,2)，画到图上
pts_center = np.stack([x_c.astype(np.int32), y_center_clip], axis=1)
pts_center = pts_center.reshape(-1, 1, 2)

# 绿色画中心线
cv2.polylines(img, [pts_center], isClosed=False, color=(0, 255, 0), thickness=1)

cv2.imwrite('roi_edges_centerline.png', img)
# ======================
# 步骤7：轮廓重建
# ======================
# 1. 读取 ROI 尺寸，用它来确定重建画布大小
roi_gray = cv2.imread('roi_pipe.png', cv2.IMREAD_GRAYSCALE)
h_roi, w_roi = roi_gray.shape[:2]

# 2. 在整幅宽度上采样 x，并计算上下边缘拟合值
x_fit = np.arange(0, w_roi, dtype=np.float32)

y_top_fit = poly_top(x_fit)
y_bot_fit = poly_bot(x_fit)

# 裁到图像范围并转 int
y_top_clip = np.clip(y_top_fit, 0, h_roi - 1).astype(np.int32)
y_bot_clip = np.clip(y_bot_fit, 0, h_roi - 1).astype(np.int32)

# 3. 组装一个封闭多边形轮廓：上边从左到右，下边从右到左
pts_top = np.stack([x_fit,          y_top_clip], axis=1)          # (W,2)
pts_bot = np.stack([x_fit[::-1],    y_bot_clip[::-1]], axis=1)    # (W,2)
pts_poly = np.concatenate([pts_top, pts_bot], axis=0)             # (2W,2)
pts_poly = pts_poly.reshape(-1, 1, 2).astype(np.int32)

# 4. 创建一张干净画布并填充管道区域
canvas = np.zeros((h_roi, w_roi, 3), dtype=np.uint8)               # 黑底
cv2.fillPoly(canvas, [pts_poly], (255, 255, 255))                  # 管道区域刷成白色

# （可选）给轮廓描一下边，方便看
cv2.polylines(canvas, [pts_poly], isClosed=True, color=(0, 0, 0), thickness=1)

# 5. 同时在这张画布上画一条中心线（纯视觉用）
y_center_clip = ((y_top_clip + y_bot_clip) / 2.0).astype(np.int32)
pts_center = np.stack([x_fit.astype(np.int32), y_center_clip], axis=1).reshape(-1, 1, 2)
cv2.polylines(canvas, [pts_center], isClosed=False, color=(0, 255, 0), thickness=1)

cv2.imwrite('pipe_reconstruction.png', canvas)
print("轮廓重建已保存为 pipe_reconstruction.png")

# ============================
# 鼠标交互：测量指定列的厚度
# ============================

# 为了在回调里方便使用，把需要的数据放到一个 dict 里
data = {
    "base": canvas,            # 重建后的底图（彩色）
    "y_top": y_top_clip,       # 上边缘 y 数组
    "y_bot": y_bot_clip        # 下边缘 y 数组
}

def on_mouse(event, x, y, flags, param):
    base = param["base"]
    y_top_arr = param["y_top"]
    y_bot_arr = param["y_bot"]
    h, w = base.shape[:2]

    # 防止越界
    if x < 0 or x >= w:
        return

    if event == cv2.EVENT_MOUSEMOVE:
        # 当前列的上下边缘
        y_top = int(y_top_arr[x])
        y_bot = int(y_bot_arr[x])
        thickness = float(abs(y_bot - y_top))   # 像素厚度

        # 底图拷贝
        disp = base.copy()

        # 画红色测量竖线
        cv2.line(disp, (x, y_top), (x, y_bot), (0, 0, 255), 2)

        # --------- 白底标签区域 ----------
        tag_w, tag_h = 180, 50  # 标签宽高
        # 尝试放在竖线右上方，超出边界再调整
        x0 = x + 15
        y0 = y_top - tag_h - 10

        if x0 + tag_w > w - 5:
            x0 = w - tag_w - 5
        if y0 < 5:
            y0 = 5

        x1 = x0 + tag_w
        y1 = y0 + tag_h

        # 白底 + 黑边框
        cv2.rectangle(disp, (x0, y0), (x1, y1), (255, 255, 255), -1)
        cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 0, 0), 1)

        # --------- 标签文字 ----------
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8      # 字体放大
        thick = 2             # 字体描边加粗

        text1 = f"x = {x}"
        text2 = f"t = {thickness:.2f}px"

        # 两行文字位置
        cv2.putText(disp, text1, (x0 + 8, y0 + 20),
                    font, font_scale, (0, 0, 0), thick, cv2.LINE_AA)
        cv2.putText(disp, text2, (x0 + 8, y0 + 40),
                    font, font_scale, (0, 0, 0), thick, cv2.LINE_AA)

        cv2.imshow('pipe_measure', disp)


# 打开交互窗口
cv2.namedWindow('pipe_measure', cv2.WINDOW_NORMAL)
cv2.imshow('pipe_measure', canvas)
cv2.setMouseCallback('pipe_measure', on_mouse, data)

print("鼠标移动到管道上即可看到当前列厚度（px），按 ESC 或 q 退出。")

while True:
    key = cv2.waitKey(20) & 0xFF
    if key == 27 or key == ord('q'):
        break

cv2.destroyAllWindows()
