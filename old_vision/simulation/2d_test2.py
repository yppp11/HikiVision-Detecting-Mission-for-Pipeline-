import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['DejaVu Sans']  # 只用默认英文字体即可
rcParams['axes.unicode_minus'] = False

# ===================== 基本物理参数 =====================
# 视野 80 mm、分辨率 5120 像素，对应像素尺寸：
pixel_size_mm = 80.0 / 5120.0             # mm / 像素
pixel_size_um = pixel_size_mm * 1000.0    # μm / 像素


# ===================== 一维亚像素边缘检测 =====================
def estimate_edge_subpixel_1d(profile, border=20):
    """
    对一条灰度剖面做亚像素边缘定位：
      - 用中心差分算梯度
      - 在忽略左右 border 像素后找梯度峰值
      - 用三点抛物线拟合求峰值位置
    返回：边缘位置（像素坐标，浮点）
    """
    prof = np.asarray(profile, dtype=np.float64)
    n = prof.size

    # 中心差分梯度
    g = np.zeros_like(prof)
    g[1:-1] = 0.5 * (prof[2:] - prof[:-2])

    abs_g = np.abs(g)

    # 忽略最左/最右边若干像素，防止边界假边缘
    if n > 2 * border + 2:
        abs_g[:border] = 0.0
        abs_g[-border:] = 0.0

    i_max = int(np.argmax(abs_g))

    # 安全检查：如果在边界附近不做亚像素拟合
    if i_max <= 0 or i_max >= n - 1:
        return float(i_max)

    g1, g2, g3 = abs_g[i_max - 1], abs_g[i_max], abs_g[i_max + 1]
    denom = (g1 - 2.0 * g2 + g3)
    if abs(denom) < 1e-12:
        return float(i_max)

    # 抛物线顶点位置（相对 i_max 的偏移）
    delta = 0.5 * (g1 - g3) / denom
    if abs(delta) > 1.0:
        return float(i_max)

    return float(i_max) + float(delta)


# ===================== 高斯卷积（带边界处理） =====================
def gaussian_kernel1d(sigma_pixels, radius=None):
    if sigma_pixels <= 0:
        raise ValueError("sigma_pixels must be > 0")
    if radius is None:
        radius = int(3.0 * sigma_pixels)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma_pixels) ** 2)
    k /= k.sum()
    return k


def gaussian_blur_2d_reflect(img, sigma_pixels):
    """
    分离卷积实现 2D 高斯模糊，边界使用 'edge' 填充，
    避免卷积时引入 0 灰度导致的假边缘。
    """
    k = gaussian_kernel1d(sigma_pixels)
    r = len(k) // 2
    H, W = img.shape

    # 先沿 x 方向
    tmp = np.zeros_like(img, dtype=np.float64)
    for y in range(H):
        row = img[y, :]
        row_pad = np.pad(row, (r, r), mode='edge')  # 左右用边缘值填充
        tmp[y, :] = np.convolve(row_pad, k, mode='valid')

    # 再沿 y 方向
    blurred = np.zeros_like(tmp, dtype=np.float64)
    for x in range(W):
        col = tmp[:, x]
        col_pad = np.pad(col, (r, r), mode='edge')
        blurred[:, x] = np.convolve(col_pad, k, mode='valid')

    return blurred


# ===================== 构造二维真实边缘曲线 =====================
def build_edge_curve(height, x0_pix=200.0, amp_pix=5.0):
    """
    构造二维边缘 x_true(y) = x0 + δx(y)，包含：
      - 轻微整体弯曲（二次项）
      - 上半部一个凸起（高斯 bump）
      - 下半部一个凹陷（高斯 dent）
    """
    y = np.arange(height, dtype=np.float64)
    H = float(height)

    # 1. 整体轻微弯曲
    y_norm = (y - 0.5 * H) / (0.5 * H)  # [-1, 1]
    curve = 0.3 * amp_pix * (y_norm ** 2 - 0.5)

    # 2. 上半部凸起
    center1 = 0.3 * H
    sigma1 = 0.08 * H
    bump = 0.7 * amp_pix * np.exp(-0.5 * ((y - center1) / sigma1) ** 2)

    # 3. 下半部凹陷
    center2 = 0.75 * H
    sigma2 = 0.06 * H
    dent = -0.6 * amp_pix * np.exp(-0.5 * ((y - center2) / sigma2) ** 2)

    x_true = x0_pix + curve + bump + dent
    return x_true


# ===================== 构造二维图像：二值 -> 模糊 -> 下采样 -> 加噪 =====================
def generate_2d_edge_image(height=512, width=1024,
                           amp_pix=5.0,
                           upsample=4,
                           sigma_blur=1.5,
                           fg_gray=230.0,
                           bg_gray=20.0,
                           noise_gain=0.5,
                           noise_read=1.0):
    """
    生成模拟相机图像：
      - height, width: 低分辨率图像尺寸
      - amp_pix: 边缘弯曲/凸起的幅度（像素）
      - upsample: 高分辨率过采样倍数
      - sigma_blur: 低分辨率坐标系下的高斯模糊 σ（像素）
      - fg_gray: 物体侧灰度
      - bg_gray: 背景侧灰度
      - noise_gain: 信号相关噪声增益系数（~ sqrt(signal)）
      - noise_read: 常数读出噪声（灰度级）

    返回:
      img_noisy: height x width 模拟观测图像
      x_true:    每一行的真实边缘位置（像素）
    """
    # 1. 真实边缘曲线（在低分辨率坐标系）
    x0_pix = width * 0.4  # 把边缘放在图像偏左的位置
    x_true = build_edge_curve(height, x0_pix=x0_pix, amp_pix=amp_pix)

    H_lr, W_lr = height, width
    H_hr, W_hr = H_lr * upsample, W_lr * upsample

    # 高分辨率 y 坐标
    y_hr = np.arange(H_hr, dtype=np.float64) / float(upsample)
    y_lr = np.arange(H_lr, dtype=np.float64)
    # 插值 x_true 到高分辨率每一行
    x_true_interp = np.interp(y_hr, y_lr, x_true)
    x_edge_hr = x_true_interp * upsample  # 转成高分辨率像素坐标

    # 2. 构造高分辨率二值图像（左侧物体、右侧背景）
    img_hr = np.full((H_hr, W_hr), bg_gray, dtype=np.float64)
    x_coords_hr = np.arange(W_hr, dtype=np.float64)[None, :]  # 1 x W_hr
    mask_obj = x_coords_hr <= x_edge_hr[:, None]
    img_hr[mask_obj] = fg_gray

    # 3. 高分辨率高斯模糊
    sigma_hr = sigma_blur * upsample
    img_blur_hr = gaussian_blur_2d_reflect(img_hr, sigma_pixels=sigma_hr)

    # 4. 下采样到相机分辨率（平均池化）
    img_blur_lr = img_blur_hr.reshape(H_lr, upsample, W_lr, upsample).mean(axis=(1, 3))

    # 5. 加噪声：信号相关 + 读出噪声
    signal = img_blur_lr
    noise_sigma_mat = noise_gain * np.sqrt(np.maximum(signal, 1.0)) + noise_read
    noise = noise_sigma_mat * np.random.randn(H_lr, W_lr)
    img_noisy = signal + noise
    img_noisy = np.clip(img_noisy, 0.0, 255.0)

    return img_noisy, x_true


# ===================== 二维评估：逐行检测 + 误差统计 =====================
def evaluate_2d_edge_detection(img, x_true, border=20):
    H, W = img.shape
    assert x_true.shape[0] == H

    x_est = np.zeros(H, dtype=np.float64)
    for y in range(H):
        profile = img[y, :]
        x_est[y] = estimate_edge_subpixel_1d(profile, border=border)

    errors = x_est - x_true

    mean_err = errors.mean()
    std_err = errors.std(ddof=1)
    max_err = np.max(np.abs(errors))

    amp_true = x_true.max() - x_true.min()
    amp_est = x_est.max() - x_est.min()
    amp_err = amp_est - amp_true

    stats = {
        "mean_err_pix": mean_err,
        "std_err_pix": std_err,
        "max_err_pix": max_err,
        "mean_err_um": mean_err * pixel_size_um,
        "std_err_um": std_err * pixel_size_um,
        "max_err_um": max_err * pixel_size_um,
        "amp_true_pix": amp_true,
        "amp_est_pix": amp_est,
        "amp_err_pix": amp_err,
        "amp_true_um": amp_true * pixel_size_um,
        "amp_est_um": amp_est * pixel_size_um,
        "amp_err_um": amp_err * pixel_size_um,
        "x_est": x_est,
    }
    return stats


# ===================== 可视化 =====================
def visualize_results(img, x_true, x_est, rows_to_show=None):
    H, W = img.shape
    y = np.arange(H)

    if rows_to_show is None:
        rows_to_show = [H // 4, H // 2, 3 * H // 4]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 图 1：图像 + 真边缘 + 估计边缘
    ax = axes[0]
    ax.imshow(img, cmap='gray', origin='upper')
    ax.plot(x_true, y, 'r-', linewidth=1, label='true edge')
    ax.plot(x_est, y, 'c--', linewidth=1, label='estimated edge')
    ax.invert_yaxis()
    ax.set_title("Simulated image & edges")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.legend()

    # 图 2：误差随 y
    ax = axes[1]
    errors = x_est - x_true
    ax.plot(errors, y, '.-')
    ax.invert_yaxis()
    ax.set_xlabel("error (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_title("Edge position error e(y)")

    # 图 3：几条扫描线的灰度剖面
    ax = axes[2]
    for r in rows_to_show:
        profile = img[r, :]
        ax.plot(profile, label=f"y={r}")
    x_min = np.min(x_true) - 40
    x_max = np.min(x_true) + 80
    ax.set_xlim(max(0, x_min), min(W, x_max))
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("gray level")
    ax.set_title("Intensity profiles")
    ax.legend()

    plt.tight_layout()
    plt.show()


# ===================== 主函数 =====================
def main():
    np.random.seed(0)  # 为了复现性

    height = 512      # 可以先用 512x1024 测试
    width = 1024
    amp_pix = 5.0     # 弯曲/凸起幅度 (像素)
    upsample = 4
    sigma_blur = 1.5  # 低分辨率下边缘模糊 σ (像素)
    noise_gain = 0.5
    noise_read = 1.0

    print(f"像素尺寸约为 {pixel_size_um:.3f} μm/像素\n")

    img, x_true = generate_2d_edge_image(
        height=height,
        width=width,
        amp_pix=amp_pix,
        upsample=upsample,
        sigma_blur=sigma_blur,
        fg_gray=230.0,
        bg_gray=20.0,
        noise_gain=noise_gain,
        noise_read=noise_read,
    )

    stats = evaluate_2d_edge_detection(img, x_true, border=20)

    print("二维仿真结果:")
    print("  误差均值:   {:+.4f} 像素 ({:+.2f} μm)".format(
        stats["mean_err_pix"], stats["mean_err_um"]))
    print("  误差标准差: {:.4f} 像素 ({:.2f} μm)".format(
        stats["std_err_pix"], stats["std_err_um"]))
    print("  最大误差:   {:.4f} 像素 ({:.2f} μm)".format(
        stats["max_err_pix"], stats["max_err_um"]))

    print("\n  真实弯曲振幅: {:.3f} 像素 ({:.2f} μm)".format(
        stats["amp_true_pix"], stats["amp_true_um"]))
    print("  估计弯曲振幅: {:.3f} 像素 ({:.2f} μm)".format(
        stats["amp_est_pix"], stats["amp_est_um"]))
    print("  振幅误差:     {:+.3f} 像素 ({:+.2f} μm)".format(
        stats["amp_err_pix"], stats["amp_err_um"]))

    # 画图看看真实/估计边缘与误差
    visualize_results(img, x_true, stats["x_est"])


if __name__ == "__main__":
    main()
