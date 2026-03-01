import cv2
import numpy as np

# 1. 读入灰度图
img = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("未找到图片，请检查路径")
h, w = img.shape[:2]

# 2. 轻微高斯滤波
blur = cv2.GaussianBlur(img, (5, 5), 1.0)

# ================= 修改开始 =================
# 3. 固定阈值分割
# 设置你想要的阈值 (范围 0-255)
# 如果管子很亮，背景很黑，建议设置在 150-220 之间
# 如果对比度不明显，可能需要设置在 100 左右
target_thresh = 220

# 注意：这里去掉了 cv2.THRESH_OTSU，只保留 cv2.THRESH_BINARY
_, binary = cv2.threshold(
    blur, 
    target_thresh,   # 这里使用你设置的固定值
    255,             # 超过阈值的像素设为白色(255)
    cv2.THRESH_BINARY 
)
# ================= 修改结束 =================

# 4. 形态学闭运算
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# 5. 找外部轮廓
contours, _ = cv2.findContours(
    binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

if len(contours) == 0:
    raise RuntimeError(f"未找到任何轮廓，当前阈值为 {target_thresh}，请尝试调低该值。")

main_cnt = max(contours, key=cv2.contourArea)

# 6. 外接矩形
x, y, w_box, h_box = cv2.boundingRect(main_cnt)

# 7. 适当扩边
margin_x = 400   
margin_y = 50 

x0 = max(0, x - margin_x)
y0 = max(0, y - margin_y)
x1 = min(w, x + w_box + margin_x)
y1 = min(h, y + h_box + margin_y)

roi = img[y0:y1, x0:x1]

# 8. 保存和可视化
cv2.imwrite('roi_pipe.png', roi)

vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
cv2.imwrite('roi_on_full.png', vis)

print(f"当前使用固定阈值：{target_thresh}")
print("ROI 坐标：", x0, y0, x1, y1, " 尺寸：", roi.shape[1], "x", roi.shape[0])