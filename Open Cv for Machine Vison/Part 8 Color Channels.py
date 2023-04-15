import cv2 as cv
import numpy as np

#-----------[LOAD IMAGE]-----------------------------------------------------------------------------#
img =cv.imread('img/coin.jpg',cv.IMREAD_COLOR)
#-----------[Color Chanel]-----------------------------------------------------------------------------#
b,g,r =cv.split(img)
blank=np.zeros(img.shape[:2], dtype ='uint8')
#background blue or green or black
#black = 0 
blue =cv.merge([b,blank,blank])
green=cv.merge([blank,g,blank])
red=cv.merge([blank,blank,r])

#-----------[SHOW RESULT]----------------------------------------------------------------------------#
print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

cv.imshow('Coin',img)
cv.imshow('Blue',b)  #-> màu xanh trong coin -> graysalce
cv.imshow('Green',g) #-> màu xanh lá trong coin -> graysalce
cv.imshow('Red',r)   #-> màu đỏ trong coin -> graysalce
print(img.shape)
print(blue.shape)
print(green.shape)
print(red.shape)

cv.imshow('Blue_CN',blue) #-> color img
cv.imshow('Green_CN',green) # -> color img
cv.imshow('Red_CN',red) #-> color img

cv.waitKey(0)
cv.destroyAllWindows()