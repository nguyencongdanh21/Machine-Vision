#import library
import cv2 as cv
import numpy as np
#load img
img = cv.imread('img3/noise.jpg',cv.IMREAD_COLOR)
#smooth image
averaging = cv.blur(img,(5,5))
gaussian=cv.GaussianBlur(img,(5,5),0)
median = cv.medianBlur(img,5)

# show imgae
cv.imshow('noise',img)
cv.imshow('Averaging',averaging)
cv.imshow('Gaussian',gaussian)
cv.imshow('Median',median)

cv.waitKey(0)
cv.destroyAllWindows()