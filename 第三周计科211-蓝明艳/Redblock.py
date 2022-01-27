import cv2
import numpy as np

img = cv2.imread('./recourse/red block.png')  # 读图片

lower_red = np.array([156, 43, 46])  # 设置阈值
upper_red = np.array([180, 255, 255])
hsv_red = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 将BGR图像转换成HSV图像
mask = cv2.inRange(hsv_red, lower_red, upper_red)  # 提取
res = cv2.bitwise_and(img, img, mask=mask)  # 按位与，通过掩膜显示红色
print(hsv_red, [[[60, 255, 255]]])  # 打印红色的HSV值

gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)  # 变为灰度图
retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # 大津法二值化
dst = cv2.dilate(dst, None, iterations=1)  # 膨胀，白区域变大
dst = cv2.erode(dst, None, iterations=1)  # 腐蚀，白区域变小

rows, cols = dst.shape  # 旋转图片并将图片缩小为原来的一半
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 71, 0.5)
exc = cv2.warpAffine(dst, M, (2 * cols, 2 * rows))

cv2.imshow('image', exc)  # 显示最后成型的图片
cv2.waitKey(0)  # 按任意键关闭窗口
cv2.destroyAllWindows()
