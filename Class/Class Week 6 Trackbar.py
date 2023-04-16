import numpy as np
import cv2 as cv

def thresh_lever(val):
    b_img=cv.threshold(img_b, val, 255, cv.THRESH_BINARY)[1]
    cv.imshow('threshold', b_img)

img= cv.imread('img2/rice.jpg',cv.IMREAD_COLOR)
img = img[: ,: ,2]

img_b = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray_fillter = cv.GaussianBlur(img_b, (4,4),0)
img= cv.adaptiveThreshold(gray_fillter, 255, cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,51,9)



cv.namedWindow('abc')
cv.createTrackbar('xyz', 'abc', 0, 255, thresh_lever)
cv.imshow('raw',img)
cv.imshow('color',img_b)
#cv.imshow('Gray',gray)
cv.waitKey(0)
cv.destroyAllWindows()

