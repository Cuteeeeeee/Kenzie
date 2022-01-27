import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./recourse/house.jpg', 0)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)  # 将低频分量转移到频谱中心
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))  # 构建振幅图

rows, cols = img.shape
crow, ccol = rows / 2, cols / 2  # 获取中心位置
mask = np.ones((rows, cols, 2), np.uint8)  # 高通滤波
mask[int(crow - 10):int(crow + 10), int(ccol - 114):int(ccol - 4)] = 0
mask[int(crow - 10):int(crow + 10), int(ccol + 4):int(ccol + 114)] = 0
fshift = dft_shift * mask  # 将掩码与当前得到的结果结合在一起
f_ishift = np.fft.ifftshift(fshift)  # 将低频还原到左上角
img_back = cv2.idft(f_ishift)  # 傅里叶逆变换
img_back = cv2.magnitude(img_back[:, :, 1], img_back[:, :, 0])  # 将实部与虚部进行处理
plt.imshow(img_back, cmap='gray')
plt.title('result'), plt.xticks([]), plt.yticks([])
plt.show()
