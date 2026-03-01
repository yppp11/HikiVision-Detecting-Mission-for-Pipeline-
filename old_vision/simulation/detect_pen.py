import cv2
import numpy as np
import matplotlib.pyplot as plt

# =================================================================
# 1. 全局配置区域
# =================================================================
CONFIG = {
    # 输入输出
    "image_path": "test.jpg",
    
    # 基础处理
    "binary_threshold": 100,         # 二值化阈值 (剔除阴影)
    "proj_threshold": 500,           # 投影阈值 (定位ROI)
    "roi_padding": 500,               # ROI 上下留白
    
    # 内部细节优化参数 (关键修改!)
    "mask_erode_iter": 6,            # 【新】去皮力度：数值越大，越往内部缩，边缘噪点越少 (建议 5-8)
    "clahe_clip_limit": 3.0,         # 【新】打光强度：增强对比度，让笔帽细节显形
    "canny_threshold1": 40,          # 弱边缘
    "canny_threshold2": 120,         # 强边缘
    "edge_thickness_iter": 1,        # 边缘加粗 (现在细节多了，设为1就够了，太粗会糊)
    
    # 可视化颜色
    "box_color": (0, 0, 255),        # 外框颜色 (红)
    "edge_color": (0, 255, 0),       # 内部细节颜色 (绿)
    "line_thickness": 5,             
    "font_scale": 3.0,               
    "font_thickness": 5              
}

def analyze_and_measure(cfg):
    # --- Step 1: 读取 ---
    original_gray = cv2.imread(cfg["image_path"], cv2.IMREAD_GRAYSCALE)
    if original_gray is None:
        print(f"错误：无法读取 {cfg['image_path']}")
        return
    img_h, img_w = original_gray.shape
    print(f"图像尺寸: {img_w}x{img_h}")

    # --- Step 2: 二值化 (制作 Mask) ---
    ret, binary = cv2.threshold(original_gray, cfg["binary_threshold"], 255, cv2.THRESH_BINARY_INV)

    # --- Step 3: 投影定位 ROI ---
    h_projection = np.sum(binary == 255, axis=1)
    valid_rows = np.where(h_projection > cfg["proj_threshold"])[0]
    if len(valid_rows) == 0: 
        print("未找到物体")
        return

    start_row = max(0, valid_rows[0] - cfg["roi_padding"])
    end_row = min(img_h, valid_rows[-1] + cfg["roi_padding"])
    print(f"ROI: {start_row} - {end_row}")

    # --- Step 4: 抠图 & 测量 ---
    roi_gray = original_gray[start_row:end_row, :]
    roi_binary = binary[start_row:end_row, :]

    # 4.1 闭运算：填补空洞，用于【测量】(保持原大小)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    measure_mask = cv2.morphologyEx(roi_binary, cv2.MORPH_CLOSE, kernel_close)

    # 4.2 测量 (外轮廓)
    contours, _ = cv2.findContours(measure_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_vis = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR) # 准备画布

    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(max_contour)
        (center_x, center_y), (rect_w, rect_h), angle = rect
        length_pixel = max(rect_w, rect_h)
        thickness_pixel = min(rect_w, rect_h)
        
        # 画测量红框
        box = np.int32(cv2.boxPoints(rect))
        cv2.drawContours(result_vis, [box], 0, cfg["box_color"], cfg["line_thickness"])
        
        # 标文字
        label = f"L: {length_pixel:.1f} px"
        cv2.putText(result_vis, label, (box[1][0], max(0, box[1][1]-40)), 
                    cv2.FONT_HERSHEY_SIMPLEX, cfg["font_scale"], cfg["box_color"], cfg["font_thickness"])
        print(f"测量长度: {length_pixel:.3f}, 直径: {thickness_pixel:.3f}")

    # --- Step 5: 内部细节检测 (核心优化) ---
    
    # 5.1 【去皮】：腐蚀 Mask，向内缩，避开边缘阴影
    kernel_erode = np.ones((3, 3), np.uint8)
    # 这里的 iterations 控制缩多少，缩得越多，边缘越干净
    shrunk_mask = cv2.erode(measure_mask, kernel_erode, iterations=cfg["mask_erode_iter"])

    # 5.2 【打光】：CLAHE 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=cfg["clahe_clip_limit"], tileGridSize=(8,8))
    enhanced_gray = clahe.apply(roi_gray)

    # 5.3 抠图：只看内部
    masked_pen = cv2.bitwise_and(enhanced_gray, enhanced_gray, mask=shrunk_mask)

    # 5.4 Canny 边缘检测
    raw_edges = cv2.Canny(masked_pen, cfg["canny_threshold1"], cfg["canny_threshold2"])

    # 5.5 过滤微小噪点 (可选优化)
    # 只保留稍微长一点的线条，去掉孤立的像素点
    edge_contours, _ = cv2.findContours(raw_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    clean_edges = np.zeros_like(raw_edges)
    for c in edge_contours:
        if cv2.arcLength(c, False) > 20: # 只有长度大于20像素的线才画出来
            cv2.drawContours(clean_edges, [c], -1, 255, 1)

    # 5.6 膨胀加粗
    if cfg["edge_thickness_iter"] > 0:
        kernel_dilate = np.ones((3, 3), np.uint8)
        final_edges = cv2.dilate(clean_edges, kernel_dilate, iterations=cfg["edge_thickness_iter"])
    else:
        final_edges = clean_edges

    # 5.7 叠加绿色边缘
    result_vis[final_edges > 0] = cfg["edge_color"]

    # --- Step 6: 显示 ---
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.title("1. Enhanced Gray (CLAHE)")
    plt.imshow(enhanced_gray, cmap='gray')

    plt.subplot(2, 2, 2)
    plt.title(f"2. Clean Edges (Eroded Mask + Filtered)")
    plt.imshow(final_edges, cmap='gray')

    plt.subplot(2, 1, 2) 
    plt.title("3. Final Result")
    plt.imshow(cv2.cvtColor(result_vis, cv2.COLOR_BGR2RGB))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_and_measure(CONFIG)