import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#-----------[LOAD IMAGE]-----------------------------------------------------------------------------#
img= cv.imread('img/coin4.jpg',cv.IMREAD_COLOR)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
img= cv.resize(gray,(356,256))
#-----------[PROCESSING IMAGE]-----------------------------------------------------------------------#
# 1 Average Blur
average = cv.blur(img,(5,5))
# 2 Gassian Blur
gaussian_blur=cv.GaussianBlur(img,(5,5),1)
# 3 Median Blur
median_blur=cv.medianBlur(img,5) #-> open cv automatic know 3 is 3x3 matrix
# 4 Bilateral Blur ->giữ đc viền ngoài
bilateral_blur=cv.bilateralFilter(img,5,150,150) # ->sigma color =15 , sigma space =15

#Binary 
ret1,thresh_val= cv.threshold(img,thresh=245,maxval=255,type=cv.THRESH_BINARY_INV)
ret1,thresh_val1= cv.threshold(bilateral_blur,thresh=245,maxval=255,type=cv.THRESH_BINARY_INV)
ret2,thresh_val2= cv.threshold(average,thresh=245,maxval=255,type=cv.THRESH_BINARY_INV)
ret3,thresh_val3= cv.threshold(gaussian_blur,thresh=245,maxval=255,type=cv.THRESH_BINARY_INV)
ret4,thresh_val4= cv.threshold(median_blur,thresh=245,maxval=255,type=cv.THRESH_BINARY_INV)
#Canny
canny=cv.Canny(img,50,255)
canny1=cv.Canny(average,50,225) 
canny2=cv.Canny(gaussian_blur,50,225) 
canny3=cv.Canny(median_blur,50,225) 
canny4=cv.Canny(bilateral_blur,50,225) 
# GRAYSCALE HISTOGRAM #
gray_hist = cv.calcHist([gray],[0],None,[256],[0,256])
plt.figure()
plt.title('Gray Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()
#-----------[SHOW RESULT]----------------------------------------------------------------------------#

# Blur picture
cv.imshow('coin',img)
cv.imshow('Average Blur',average)
cv.imshow('Gassian Blur',gaussian_blur)
cv.imshow('Median Blur',median_blur)
cv.imshow('Bilateral Blur',bilateral_blur)

# Binary picture
cv.imshow('Coin Binary',thresh_val)
cv.imshow('Binary with Bilateral Blur',thresh_val1)
cv.imshow('Binary with Average Blur',thresh_val2)
cv.imshow('Binary with Gassian Blur',thresh_val3)
cv.imshow('Binary with Median Blur',thresh_val4)

#Canny
cv.imshow('Canny Image',canny)
cv.imshow('Canny Average',canny1)
cv.imshow('Canny Gaussian',canny2)
cv.imshow('Canny Mdedian',canny3)
cv.imshow('Canny Bilateral',canny4)

cv.waitKey(0)
cv.destroyAllWindows()