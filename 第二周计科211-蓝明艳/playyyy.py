import numpy as np
import cv2


def nothing(x):
    pass


drawing = False
x1, x2 = -1, -1


def draw(event, x, y, flags, param):
    global x1, x2, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, x2 = x, y
    if event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        pd = img.copy()
        if drawing == True:
            cv2.rectangle(pd, (x1, x2), (x, y), (0, 0, 0), 5)
            cv2.imshow('image', pd)
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            cv2.rectangle(img, (x1, x2), (x, y), (0, 0, 0), 5)
        drawing == False


img = np.zeros((512, 680, 3), np.uint8)
img[:] = 255
cv2.namedWindow('image')
while (1):
    cv2.setMouseCallback('image', draw)
    cv2.imshow('image', img)
    k = cv2.waitKey(0)
    if k == 27:
        break
cv2.destroyAllWindows()
