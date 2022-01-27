import cv2
import numpy as np


def nothing(x):
    pass


def Exchange(img, pd):  # 创建回调函数
    r = cv2.getTrackbarPos('R', 'image') / 100  # 用滑动条上面显示的数值对两张图片的权重进行改变
    return cv2.addWeighted(img, 1 - r, pd, r, 0)


img = cv2.imread('./recourse/football.png')  # 读图片
pd = cv2.imread("./recourse/photoo.jpg")
pd = cv2.resize(pd, (750, 467))  # 将图片大小调整为与第一张大小相同
print(img.shape)  # 输出图片的信息
print(img.size)
print(img.dtype)
ball = img[390:460, 455:535]  # 用红色矩形框圈出足球
football = ball.copy()
img[385:465, 450:540] = [0, 0, 225]
img[390:460, 455:535] = football

cv2.namedWindow('image')  # 定义窗口名
cv2.createTrackbar('R', 'image', 0, 100, nothing)  # 获得滑动条

while 1:
    cv2.imshow('image', Exchange(img, pd))  # 显示image窗口
    k = cv2.waitKey(1) & 0xff  # 等待输出指令
    if k == 27:  # Esc退出
        break
cv2.destroyAllWindows()
