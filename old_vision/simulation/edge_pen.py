import cv2
import numpy as np
import matplotlib.pyplot as plt


def estimate_edge_subpixel_1d(profile, rough_index, search_radius=5):
    """
    一维亚像素边缘定位：
      profile      : 一维灰度数组（列或行）
      rough_index  : 粗略边缘位置
      search_radius: 在粗略位置左右多少像素范围内搜索梯度峰值
    返回：
      亚像素边缘位置（浮点）
    """
    prof = profile.astype(np.float32)
    n = prof.size
    if n < 3:
        return float(rough_index)

    # 中心差分梯度
    grad = np.zeros_like(prof, dtype=np.float32)
    grad[1:-1] = 0.5 * (prof[2:] - prof[:-2])
    abs_grad = np.abs(grad)

    # 搜索窗口
    start = max(1, rough_index - search_radius)
    end = min(n - 2, rough_index + search_radius)
    if end <= start:
        return float(rough_index)

    local = abs_grad[start:end + 1]
    i_rel = int(np.argmax(local))
    i = start + i_rel

    if i <= 0 or i >= n - 1:
        return float(i)

    g1, g2, g3 = abs_grad[i - 1], abs_grad[i], abs_grad[i + 1]
    denom = (g1 - 2.0 * g2 + g3)
    if abs(denom) < 1e-6:
        return float(i)

    delta = 0.5 * (g1 - g3) / denom
    if abs(delta) > 1.0:
        return float(i)

    return float(i) + float(delta)


def detect_pen_like_object(image_path, thresh_val=80, invert=True, debug_show=True):
    """
    用“管路式”亚像素边缘检测一根笔的厚度（像素单位）

    thresh_val: 手动阈值（0~255）
    invert    : True 表示笔比背景暗，用 THRESH_BINARY_INV
                False 表示笔比背景亮
    """
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        print("无法读取图像:", image_path)
        return

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.0)

    # 1. 手动阈值分割，得到大致前景区域
    if invert:
        _, mask0 = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY_INV)
    else:
        _, mask0 = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY)

    # 2. 形态学：填缝 + 去小噪声（粗掩膜即可，不要求精细）
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask0, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # 3. 连通域：保留面积较大的几个块（条码切断也没关系）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        print("没有找到前景连通域")
        return

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = areas.max()
    keep_labels = [i + 1 for i, a in enumerate(areas) if a > max_area * 0.1]

    clean_mask = np.zeros_like(mask)
    for lb in keep_labels:
        clean_mask[labels == lb] = 255

    # 4. 用 clean_mask 做 minAreaRect，得到笔的大致方向，旋转到“长轴水平”
    ys, xs = np.where(clean_mask > 0)
    pts = np.column_stack((xs, ys)).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    center, (w_rect, h_rect), angle = rect

    # 保证 w_rect 是长边
    if w_rect < h_rect:
        w_rect, h_rect = h_rect, w_rect
        angle += 90.0

    rot_angle = -angle

    h_img, w_img = gray.shape
    M = cv2.getRotationMatrix2D(center, rot_angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    nW = int(h_img * sin + w_img * cos)
    nH = int(h_img * cos + w_img * sin)

    M[0, 2] += nW / 2 - center[0]
    M[1, 2] += nH / 2 - center[1]

    rot_gray = cv2.warpAffine(gray,      M, (nW, nH), flags=cv2.INTER_LINEAR)
    rot_mask = cv2.warpAffine(clean_mask, M, (nW, nH), flags=cv2.INTER_NEAREST)

    # 5. 在旋转后的掩膜上取一个宽一点的 ROI（只用来限制“上下大概范围”）
    ys2, xs2 = np.where(rot_mask > 0)
    if ys2.size == 0:
        print("旋转后掩膜为空")
        return

    y_min_mask, y_max_mask = ys2.min(), ys2.max()
    x_min_mask, x_max_mask = xs2.min(), xs2.max()

    # 沿长度方向（x）多留一些，两头不至于被掩膜截掉
    pad_y = 600   # 上下多留背景，方便看边缘
    pad_x = 100   # 左右多留，避免尾部被掩膜裁掉

    y0 = max(0, y_min_mask - pad_y)
    y1 = min(rot_gray.shape[0] - 1, y_max_mask + pad_y)
    x0 = max(0, x_min_mask - pad_x)
    x1 = min(rot_gray.shape[1] - 1, x_max_mask + pad_x)

    roi_gray = rot_gray[y0:y1 + 1, x0:x1 + 1]
    roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 1.0)

    H, W = roi_gray.shape

    # 把“笔的大致上下边界”映射到 ROI 坐标系，用来限定搜索带
    top_band_center = y_min_mask - y0
    bot_band_center = y_max_mask - y0

    top_start = max(1, int(top_band_center - 30))
    top_end   = min(H - 2, int(top_band_center + 40))

    bot_start = max(1, int(bot_band_center - 40))
    bot_end   = min(H - 2, int(bot_band_center + 30))

    # 6. 逐列：在固定的上/下搜索带里用梯度找两条边（不再依赖列上的掩膜）
    top_edges = np.full(W, np.nan, dtype=np.float32)
    bot_edges = np.full(W, np.nan, dtype=np.float32)

    for x in range(W):
        col = roi_blur[:, x].astype(np.float32)
        # 梯度
        grad = np.zeros_like(col)
        grad[1:-1] = 0.5 * (col[2:] - col[:-2])
        abs_g = np.abs(grad)

        # 上边缘：在上搜索带找梯度峰值
        local_top = abs_g[top_start:top_end + 1]
        if local_top.size == 0:
            continue
        i_rel_top = int(np.argmax(local_top))
        i_top = top_start + i_rel_top

        # 下边缘：在下搜索带找梯度峰值
        local_bot = abs_g[bot_start:bot_end + 1]
        if local_bot.size == 0:
            continue
        i_rel_bot = int(np.argmax(local_bot))
        i_bot = bot_start + i_rel_bot

        # 亚像素 refinement
        top_edges[x] = estimate_edge_subpixel_1d(col, i_top, search_radius=3)
        bot_edges[x] = estimate_edge_subpixel_1d(col, i_bot, search_radius=3)

    valid = np.where(~np.isnan(top_edges) & ~np.isnan(bot_edges))[0]
    if valid.size == 0:
        print("没有有效列")
        return

    te = top_edges[valid]
    be = bot_edges[valid]
    thickness = be - te

    # 简单去极端离群值
    med = np.median(thickness)
    mad = np.median(np.abs(thickness - med)) + 1e-6
    good = np.abs(thickness - med) < 4 * mad

    x_good = valid[good]
    te_good = te[good]
    be_good = be[good]
    thickness_good = thickness[good]

    mean_t = float(thickness_good.mean())
    std_t = float(thickness_good.std())

    print("有效列数:", x_good.size)
    print("厚度均值: {:.3f} px, 标准差: {:.3f} px".format(mean_t, std_t))

    # 7. 可视化
    if debug_show:
        vis = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        for x, y_top, y_bot in zip(x_good, te_good, be_good):
            cv2.circle(vis, (int(x), int(round(y_top))), 2, (0, 0, 255), -1)   # 上边缘 红
            cv2.circle(vis, (int(x), int(round(y_bot))), 2, (0, 255, 0), -1)   # 下边缘 绿

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].set_title("Original gray")
        axes[0].imshow(gray, cmap='gray')
        axes[0].axis('off')

        axes[1].set_title("Rotated ROI + subpixel edges")
        axes[1].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        axes[1].axis('off')

        axes[2].set_title("Thickness profile")
        axes[2].plot(x_good, thickness_good, '.-')
        axes[2].set_xlabel("Position along pen (column index)")
        axes[2].set_ylabel("Thickness (pixels)")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 把这里换成你的笔图片路径
    detect_pen_like_object("test.jpg", thresh_val=60, invert=True)
