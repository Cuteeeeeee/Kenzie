import numpy as np
import cv2


def nothing(x):
    pass


drawing = False
x1, x2 = -1, -1
mode = True


def draw(event, x, y, flags, param):
    global x1, x2, drawing, mode
    r = cv2.getTrackbarPos('R', 'image')
    g = cv2.getTrackbarPos('G', 'image')
    b = cv2.getTrackbarPos('B', 'image')
    color = (b, g, r)
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, x2 = x, y
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if drawing == True:
            if mode == True:
                cv2.rectangle(img, (x1, x2), (x, y), color, -1)
    elif event == cv2.EVENT_LBUTTONUP and mode == False:
        cv2.rectangle(img, (x1, x2), (x, y), color, 5)
        drawing == False
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_RBUTTON:
        if drawing == True:
            cv2.circle(img, (x, y), 3, color, -1)


img = np.zeros((512, 680, 3), np.uint8)
img[:] = 255
cv2.namedWindow('image')
cv2.createTrackbar('R', 'image', 0, 255, nothing)
cv2.createTrackbar('G', 'image', 0, 255, nothing)
cv2.createTrackbar('B', 'image', 0, 255, nothing)
cv2.setMouseCallback('image', draw)
while (1):
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('1'):
        mode == True
    elif k == ord('2'):
        mode = False
    elif k == 27:
        break
cv2.destroyAllWindows()
