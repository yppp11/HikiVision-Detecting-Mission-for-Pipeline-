import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

# ==============================================================================
# SECTION 1: 全局配置与模拟云图风格
# ==============================================================================
IMAGE_PATH = "roi_pipe.png"        
IMREAD_FLAG = cv2.IMREAD_UNCHANGED 

# 绘图风格设置
FIG_SIZE = (18, 5)                 # 横向长图，并排三张
HEATMAP_CMAP = "jet"               # Jet 配色模拟应力云图
TITLE_COLOR = "yellow"             # 标题高亮颜色
BG_STYLE = 'dark_background'       # 深色背景

# PC 计算参数
PC_MIN_WAVELENGTH = 3              
PC_N_SCALES = 4                    
PC_MULT = 2.1                      
PC_K = 10.0                        

# LoG 算子参数 (新增)
LOG_KERNEL_SIZE = (9, 9)           # 高斯模糊核大小，必须是奇数，(5,5)比较通用
LOG_SIGMA = 0                      # 0表示根据核大小自动计算标准差

# ==============================================================================
# SECTION 2: 核心计算函数
# ==============================================================================

def compute_phase_congruency_vertical(img, min_wavelength, n_scales, mult, k_thresh):
    """ 计算垂直方向的相位一致性 """
    img = img.astype(np.float32)
    rows, cols = img.shape
    img_fft = fftpack.fft(img, axis=0)
    freq = fftpack.fftfreq(rows)
    freq[0] = 1e-8 
    
    sum_an = np.zeros((rows, cols), dtype=np.float32)
    accum_real = np.zeros((rows, cols), dtype=np.float32)
    accum_imag = np.zeros((rows, cols), dtype=np.float32)
    
    for s in range(n_scales):
        wavelength = min_wavelength * (mult ** s)
        fo = 1.0 / wavelength
        sigma_on_f = 0.55
        log_gabor = np.exp( - (np.log(np.abs(freq) / fo))**2 / (2 * np.log(sigma_on_f)**2) )
        log_gabor[0] = 0 
        log_gabor_col = log_gabor.reshape(-1, 1)
        
        filt_fft = img_fft * log_gabor_col
        res = fftpack.ifft(filt_fft, axis=0)
        
        sum_an += np.abs(res)
        accum_real += res.real
        accum_imag += res.imag

    total_energy = np.sqrt(accum_real**2 + accum_imag**2)
    noise_est = np.mean(sum_an) * 0.05 
    pc = np.maximum(total_energy - (k_thresh * noise_est), 0) / (sum_an + 1e-6)
    return pc

def inspect_thermal_maps(path):
    # 1. 读取图像
    img = cv2.imread(path, IMREAD_FLAG)
    if img is None:
        print(f"提示: 使用模拟数据演示 (未找到 {path})")
        img = np.zeros((300, 500), dtype=np.uint8)
        # 模拟: 背景 + 两个不同频率的波纹 + 噪声
        for r in range(300):
            img[r, :] = 80 + 60 * np.sin(r/25) + 30 * np.cos(r/8)
        img = cv2.randn(img, img, 8) # 加一点噪点测试 LoG 的抗噪性
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2: gray = img
    else: gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 计算特征图
    print(">>> 计算一阶梯度 (Gradient)...")
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    print(">>> 计算 LoG (Laplacian of Gaussian)...")
    # 步骤 A: 高斯模糊 (平滑降噪)
    blurred = cv2.GaussianBlur(gray, LOG_KERNEL_SIZE, LOG_SIGMA)
    # 步骤 B: 拉普拉斯算子 (二阶微分提取边缘)
    # 使用 CV_64F 保持负值精度，因为 LoG 会产生负响应
    log_response = cv2.Laplacian(blurred, cv2.CV_64F)

    print(">>> 计算相位一致性 (Phase Congruency)...")
    pc_map = compute_phase_congruency_vertical(gray, PC_MIN_WAVELENGTH, PC_N_SCALES, PC_MULT, PC_K)

    # 3. 绘图流程
    plt.style.use(BG_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE)
    fig.suptitle("Edge Detection Thermal Analysis", fontsize=16, color='white')

    def plot_heatmap(ax, data, title):
        # 取绝对值：我们关心的是“变化的大小/能量”，而不是方向
        # 这样处理后，热力图看起来就像应力云图
        show_data = np.abs(data)
        
        # 动态截断：去掉极值噪点，让色彩分布更均匀
        vmin, vmax = np.percentile(show_data, 1), np.percentile(show_data, 99)
        
        im = ax.imshow(show_data, cmap=HEATMAP_CMAP, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(title, color=TITLE_COLOR, fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- 图1: Gradient Y (一阶) ---
    # 物理意义：坡度。反映亮度的变化速度。
    plot_heatmap(axes[0], grad_y, "[1] Gradient Y (Slope)\nRate of Intensity Change")

    # --- 图2: LoG (Laplacian of Gaussian) ---
    # 物理意义：曲率/突变。反映图像结构的凹凸变化。
    # 相比纯二阶导，它这里的热力图更干净，高亮区域准确对应边缘附近。
    plot_heatmap(axes[1], log_response, "[2] LoG (Laplacian of Gaussian)\nEdge Structure / Blob Detection")

    # --- 图3: Phase Congruency ---
    # 物理意义：频域结构。反映特征的一致性，不受对比度影响。
    plot_heatmap(axes[2], pc_map, "[3] Phase Congruency\nFrequency Consistency")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    inspect_thermal_maps(IMAGE_PATH)