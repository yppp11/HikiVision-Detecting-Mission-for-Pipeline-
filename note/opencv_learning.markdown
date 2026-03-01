cv2.normalize(roi_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
dst = cv2.normalize(src, dst, alpha, beta, norm_type)
这个函数在对灰度图进行二值化，src是传进来的数组（图片实际上也是一个数组），dst是一个输出的数组，alpha, beta是归一化后的上下界，NORM_MINMAX这个表示线性拉伸，后面的
.astype(np.uint8)是将图片放在8bit灰度上做。

blur = cv2.GaussianBlur(roi_8u, (5, 5), 1.0)
dst = cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]])
这里是高斯模糊代码。主要是消除高频的噪声。ksize是高斯核的大小；sigmaX是X方向的标准差，sigmaY不写的话就是按照sigmaX自动计算

_, binary = cv2.threshold(blur, thr_val, 255, cv2.THRESH_BINARY)
ret, dst = cv2.threshold(src, thresh, maxval, type)
src代表输入的图像，可以是8bit或者float，thresh是阈值，maxval超过阈值就输出这个值，type二值化方式；ret是实际使用的阈值，一般和传入的 thresh 一样，这里不需要就用 _ 丢弃了。

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
kernel = cv2.getStructuringElement(shape, ksize)
shape表示结构元素形状，这里 MORPH_RECT 表示矩形核。ksize：核大小

binary_close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
dst = cv2.morphologyEx(src, op, kernel[, dst[, iterations[, …]]])
这个函数是做各种组合形态学运算，op是操作的类型，MORPH_CLOSE：先膨胀再腐蚀，填补小黑洞、连接近邻的白区域。MORPH_OPEN：先腐蚀再膨胀，去掉小的白噪点、平滑边界。kernel是结构元素，iterations是重复次数。

binary_clean.sum(axis=1)对每一行求和，因为是 0 或 255 的二值图，所以一行的和 /255 就是这一行“白像素的个数”。
threshold = ratio * w：如果一行的白像素数 ≥ ratio × 行宽 w，则认为“几乎整行都是的”。
row_mask：布尔数组，长度为 h，True 表示这一行保留。

binary_main = np.zeros_like(binary_clean)
binary_main[row_mask, :] = binary_clean[row_mask, :]
生成一个和 binary_clean 同大小、全 0 的图，再把满足 row_mask 的那些行binary_clean拷贝过来。zeros_like：根据给定数组的 shape 和 dtype 创建全 0 数组。布尔索引 row_mask：只在 True 的行上赋值。

contours, _ = cv2.findContours(binary_main, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours, hierarchy = cv2.findContours(image, mode, method)
这里主要是在二值图中提取轮廓。image是传入的二值化图，mode是轮廓检索模式，RETR_EXTERNAL表示只保留最外层轮廓。method是轮廓近似方式，CHAIN_APPROX_NONE表示保留轮廓上的所有点，不做压缩。返回值contours：列表，每个元素是一个轮廓（形状约为 (N,1,2) 的数组）。hierarchy：层级关系，这里用不到。

areas = [cv2.contourArea(c) for c in contours]
area = cv2.contourArea(contour)
这里主要是计算轮廓封闭区域的面积。contour是一个轮廓点集。