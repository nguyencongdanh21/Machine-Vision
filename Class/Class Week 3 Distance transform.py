import cv2 as cv
import numpy as np

img= cv.imread('img2/2cell.jpg', cv.IMREAD_GRAYSCALE)
b_img= cv.threshold(img,150,255,cv.THRESH_BINARY)[1]

dist_img = cv.distanceTransform(b_img,cv.DIST_L2,3) ##
cv.normalize(dist_img,dist_img,0,1,cv.NORM_MINMAX) # mau den = 0 , vi tri xa nhat cua mau trang = 1

out= cv.threshold(dist_img,0.8,255,cv.THRESH_BINARY)[1]

cv.imshow('raw',dist_img)
cv.imshow('out',out)
cv.waitKey(0)
cv.destroyAllwindows()

