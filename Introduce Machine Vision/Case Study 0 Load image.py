import cv2 as cv
import numpy as np

img = cv.imread('img3/halloween.jpg', cv.IMREAD_COLOR)
img2 = cv.imread('img3/genshin1.jpg', cv.IMREAD_COLOR)
img3 = cv.imread('img3/genshin2.jpg', cv.IMREAD_COLOR)
img4 = cv.imread('img3/genshin3.jpg', cv.IMREAD_COLOR)


cv.imshow('halloween',img)
cv.waitKey(10000)
cv.imshow('halloween',img2)
cv.waitKey(10000)
cv.imshow('halloween',img3)
cv.waitKey(10000)
cv.imshow('halloween',img4)
cv.waitKey(10000)

cv.destroyAllWindows()
