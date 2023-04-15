import cv2 as cv
import numpy as np
#-----------[LOAD IMAGE]-----------------------------------------------------------------------------------------------------------------------#
img = cv.imread('img/coin2.jpg',cv.IMREAD_COLOR)
img = cv.resize(img,(356,256))
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#-----------[PROCESSING IMAGE]-----------------------------------------------------------------------------------------------------------------#
# blur
Blur=cv.medianBlur(gray,11)
#dilated = cv.dilate(Blur,(7,7),iterations=1)
eroding=cv.erode(Blur,(7,7),iterations=3) 
# Simple Thresholding
threshold,thresh=cv.threshold(src=eroding,thresh=132,maxval=255,type=cv.THRESH_BINARY) #150 -> return to threshold ,thresh -> the picture bianry
threshold,thresh_inv=cv.threshold(src=eroding,thresh=132,maxval=255,type=cv.THRESH_BINARY_INV)
# Adaptive Thresholding
#adaptive_thresh = cv.adaptiveThreshold(src=gray,maxval=255,thresholdType=cv.ADAPTIVE_THRESH_MEAN_C,type=cv.THRESH_BINARY,blockSize=11,C=3)
adaptive_thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,13,3) # caculate optimal threshold
adaptive_thresh2 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,13,3) # caculate optimal threshold
#-----------[SHOW RESULT]----------------------------------------------------------------------------------------------------------------------#
cv.imshow('Gray',gray)
cv.imshow('Coin',img)
cv.imshow('Adaptive Theshold',adaptive_thresh)
cv.imshow('Adaptive Theshold Invert',adaptive_thresh2)
cv.imshow('Binary Threshold',thresh)
cv.imshow('Binary Invert Threshold',thresh_inv)
cv.imshow('Median Blur',Blur)
cv.waitKey(0)
cv.destroyAllWindows()