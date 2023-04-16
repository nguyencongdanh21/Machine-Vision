import cv2 as cv
import numpy as np
#-----------[LOAD IMAGE]-----------------------------------------------------------------------------------------------------------------------#
img = cv.imread('img/tomato.jpg',cv.IMREAD_COLOR)
img = cv.resize(img,(456,356))
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#-----------[PROCESSING IMAGE]-----------------------------------------------------------------------------------------------------------------#
# Edge Detectrion 
# 1. Laplacian
lap=cv.Laplacian(gray, cv.CV_64F)
lap=np.uint8(np.absolute(lap))
# 2. Soble
soblex=cv.Sobel(gray,cv.CV_64F,1,0)
sobley=cv.Sobel(gray,cv.CV_64F,0,1)
combined_soble=cv.bitwise_and(soblex,sobley)
#3. Canny
canny= cv.Canny(gray,60,140)
#-----------[SHOW RESULT]----------------------------------------------------------------------------------------------------------------------#
cv.imshow('Gray',gray)
cv.imshow('Laplacian',lap)
cv.imshow('Soble X',soblex)
cv.imshow('Soble Y',sobley)
cv.imshow('Combiantion',combined_soble)
cv.imshow('Canny',canny)
cv.waitKey(0)
cv.destroyAllWindows()