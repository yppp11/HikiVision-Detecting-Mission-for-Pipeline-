import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
import time

# --- 1. 物理与仿真参数 ---
PIXEL_SIZE_UM = 80.0 / 5120.0 * 1000.0  # 约 15.625 μm/pixel
DEFECT_HEIGHT_UM = 80.0                 # 目标缺陷高度
DEFECT_HEIGHT_PX = DEFECT_HEIGHT_UM / PIXEL_SIZE_UM
SIGMA_BLUR = 1.5                        
NOISE_LEVEL = 5.0                       
TRIALS = 200                            

print(f"--- 启动可视化对比仿真 ---")
print(f"目标缺陷: {DEFECT_HEIGHT_UM:.1f} μm (约 {DEFECT_HEIGHT_PX:.2f} 像素)")

# --- 2. 核心函数 ---

def generate_roi_image(width=512, height=64, defect_h_px=5.0, shift_x=0.0):
    """
    生成图像并返回真实边缘位置
    """
    scale = 5  
    H_hr, W_hr = height * scale, width * scale
    
    x_hr = np.arange(W_hr, dtype=np.float64)
    center_x = (W_hr / 2.0) + (shift_x * scale)
    
    base_y = H_hr / 2.0
    defect_width = 40.0 * scale 
    
    # 生成真实边缘曲线 (高分辨率)
    edge_profile_hr = np.zeros_like(x_hr) + base_y
    edge_profile_hr += defect_h_px * scale * np.exp(-0.5 * ((x_hr - center_x)/(defect_width/4))**2)
    
    # 渲染图像
    img_hr = np.zeros((H_hr, W_hr), dtype=np.float64)
    y_hr_grid = np.arange(H_hr)[:, None]
    mask = y_hr_grid > edge_profile_hr
    img_hr[mask] = 50.0
    img_hr[~mask] = 200.0
    
    # 模拟模糊
    img_blur = scipy.ndimage.gaussian_filter1d(img_hr, sigma=SIGMA_BLUR*scale, axis=0)
    img_blur = scipy.ndimage.gaussian_filter1d(img_blur, sigma=SIGMA_BLUR*scale, axis=1)
    
    # 下采样到相机分辨率
    img_out = img_blur[::scale, ::scale]
    
    # 加噪
    noise = np.random.normal(0, NOISE_LEVEL, img_out.shape)
    img_out += noise
    
    # 【关键修改】提取对应的低分辨率真实曲线用于对比
    # 坐标要在Y轴上除以scale，在X轴上每隔scale取一个点
    true_curve_lr = edge_profile_hr[::scale] / scale
    
    return np.clip(img_out, 0, 255), true_curve_lr

def detect_edge(image):
    height, width = image.shape
    edge_y = []
    for x in range(width):
        col = image[:, x]
        grad = np.gradient(col)
        idx = np.argmin(grad) 
        if 1 < idx < height - 2:
            y1, y2, y3 = np.abs(grad[idx-1]), np.abs(grad[idx]), np.abs(grad[idx+1])
            denom = y1 - 2*y2 + y3
            if abs(denom) > 1e-6:
                edge_pos = float(idx) + 0.5 * (y1 - y3) / denom
            else:
                edge_pos = float(idx)
        else:
            edge_pos = float(idx)
        edge_y.append(edge_pos)
    return np.array(edge_y)

def measure_bump_height(edge_curve):
    curve_smooth = scipy.signal.medfilt(edge_curve, kernel_size=11)
    sorted_vals = np.sort(curve_smooth)
    baseline = np.mean(sorted_vals[:int(len(sorted_vals)*0.3)])
    peak_val = np.max(curve_smooth)
    return (peak_val - baseline) * PIXEL_SIZE_UM, curve_smooth

# --- 3. 主程序 ---

def main():
    measured_heights = []
    
    # 用于最后绘图的数据容器
    last_img = None
    last_true_curve = None
    last_raw_curve = None
    last_smooth_curve = None
    
    print("正在运行仿真...")
    
    for i in range(TRIALS):
        # 随机位移
        random_shift = np.random.uniform(-80, 80)
        
        # 获取图像和真值
        img, true_curve = generate_roi_image(defect_h_px=DEFECT_HEIGHT_PX, shift_x=random_shift)
        
        # 算法检测
        raw_curve = detect_edge(img)
        h_um, smooth_curve = measure_bump_height(raw_curve)
        
        measured_heights.append(h_um)
        
        # 记录最后一次的数据
        if i == TRIALS - 1:
            last_img = img
            last_true_curve = true_curve
            last_raw_curve = raw_curve
            last_smooth_curve = smooth_curve

    # 统计分析
    data = np.array(measured_heights)
    mean_val = np.mean(data)
    std_val = np.std(data)
    repeatability = 3 * std_val
    
    print("\n" + "="*40)
    print(f"测量均值     : {mean_val:.2f} μm")
    print(f"重复性 (3σ)  : {repeatability:.2f} μm")
    print("="*40)

    # --- 绘图部分 ---
    plt.figure(figsize=(12, 10))
    
    # 子图1: 测量值分布直方图
    plt.subplot(2, 1, 1)
    plt.hist(data, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(DEFECT_HEIGHT_UM, color='red', linestyle='--', linewidth=2, label='True 80um')
    plt.title(f"Statistics: Mean={mean_val:.1f}um, 3-Sigma={repeatability:.1f}um")
    plt.xlabel("Measured Height (um)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 曲线对比 (核心可视化)
    plt.subplot(2, 1, 2)
    
    # 显示背景图像 (拉伸对比度以便看清)
    plt.imshow(last_img, cmap='gray', aspect='auto', alpha=0.6)
    
    # 1. 画真实曲线 (Ground Truth) - 绿色虚线
    plt.plot(last_true_curve, color='lime', linestyle='--', linewidth=2, label='True Profile (Physics)')
    
    # 2. 画原始测量值 (Raw Measurement) - 红色细线
    plt.plot(last_raw_curve, color='red', linewidth=0.8, alpha=0.6, label='Raw Detection (Noisy)')
    
    # 3. 画平滑后曲线 (Smoothed) - 蓝色粗线
    plt.plot(last_smooth_curve, color='blue', linewidth=2.5, label='Final Algorithm Result')
    
    plt.title("Visual Verification: True vs. Measured vs. Filtered")
    plt.xlabel("Pixel Position (X)")
    plt.ylabel("Edge Position (Y)")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()