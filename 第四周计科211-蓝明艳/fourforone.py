import cv2
import numpy as np

img = cv2.imread('./recourse/label.jpg')  # 读图片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 变为灰度图
retval, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)  # 大津法二值化
dst = cv2.dilate(dst, None, iterations=1)  # 膨胀，白区域变大
dst = cv2.erode(dst, None, iterations=1)  # 腐蚀，白区域变小

cv2.namedWindow('image', 0)  # 定义窗口名
cv2.resizeWindow('image', 500, 500)  # 设定窗口大小
cv2.imshow('image', dst)  # 显示最后成型的图片
cv2.waitKey(0)  # 按任意键关闭窗口
cv2.destroyAllWindows()
