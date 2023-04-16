import cv2 as cv
import numpy as np

img= cv.imread('img2/coin3.jpg', cv.IMREAD_COLOR)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray_filtered = cv.GaussianBlur(gray,(3,3),0) 

b_img= cv.threshold(gray,180,255,cv.THRESH_BINARY_INV)[1]

dist_img = cv.distanceTransform(b_img,cv.DIST_L2,3)
cv.normalize(dist_img,dist_img,0,1,cv.NORM_MINMAX) # mau den = 0 , vi tri xa nhat cua mau trang = 1

out= cv.threshold(dist_img,0.1,255,cv.THRESH_BINARY)[1]

cv.imshow('raw',img)
cv.imshow('raw2',dist_img)
cv.imshow('out',out)

cv.waitKey(0)
cv.destroyAllwindows()
