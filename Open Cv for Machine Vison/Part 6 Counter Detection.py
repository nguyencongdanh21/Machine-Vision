import cv2 as cv
import numpy as np
#-----------[LOAD IMAGE]-----------------------------------------------------------------------------#
# import Color img
img = cv.imread('img/coin2.jpg',cv.IMREAD_COLOR)
#-----------[PROCESSING IMAGE]-----------------------------------------------------------------------#
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
blur=cv.GaussianBlur(gray,(5,5),1)
canny = cv.Canny(blur,threshold1=40,threshold2=175)
ret,thresh_val= cv.threshold(blur,thresh=120,maxval=255,type=cv.THRESH_BINARY_INV)

#-----------[COUNTING OBJECT]------------------------------------------------------------------------#
#Find Contours
contours,hierarchies=cv.findContours(thresh_val,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
#Draw Contours
blank=np.zeros(img.shape, dtype ='uint8')
cv.drawContours(blank,contours,-1,(0,255,0)) # draw all contours =-1, green color 

#Contours by Canny
contours_2,hierarchies_2=cv.findContours(canny,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

#-----------[SHOW RESULT]----------------------------------------------------------------------------#
print("# Contour",len(contours))
print("# Contour by canny",len(contours_2))

cv.imshow('Coin',img)
cv.imshow('Gray',gray)
cv.imshow('Gaussian Blur',blur)
cv.imshow('Binary',thresh_val)
cv.imshow('Blank',blank)
cv.imshow('Canny',canny)

cv.waitKey(0)
cv.destroyAllWindows()