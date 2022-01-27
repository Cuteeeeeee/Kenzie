import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./recourse/girl.png', 0)
equ = cv2.equalizeHist(img)  # 并排堆叠图像
cv2.imshow('image', equ)  # 显示均衡化后的图
img = cv2.imread('./recourse/girl.png')  # 重新读取图片
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 画2D直方图
hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
plt.imshow(hist, interpolation='nearest')
plt.show()

cv2.waitKey(0)  # 按任意键关闭窗口
cv2.destroyAllWindows()
