import cv2 as cv
import numpy as np

#img = cv.imread('noise1.jpg',cv.IMREAD_COLOR)
img = cv.imread('img3/rice.jpg',cv.IMREAD_COLOR)
#-------------------------------------------[Gray Image]-----------------------------------------------------------------------#
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#-------------------------------------------[Processing]----------------------------------------------------------------------#

#-------------------------------------------[Blur]----------------------------------------------------------------------------#
#blur/mean filter
blur= cv.blur(gray, (2,2))
#Gaussian blur 
gaussian=cv.GaussianBlur(gray,(5,5),0) # tang sigma gia tri lam mo giam
# Median blur
median=cv.medianBlur(gray,5) # khong nen di qua 5
#-------------------------------------------[Mophology]----------------------------------------------------------------------------#


#------------------------------------------[Binray image]----------------------------------------------------------------------#
thresh= 120
maxval= 255

binary_gray = cv.threshold(gray,thresh,maxval,cv.THRESH_BINARY_INV)[1]
binary_blur = cv.threshold(blur,thresh,maxval,cv.THRESH_BINARY_INV)[1]
binary_Gaussian = cv.threshold(gaussian,thresh,maxval,cv.THRESH_BINARY_INV)[1]
binary_Median = cv.threshold(median,thresh,maxval,cv.THRESH_BINARY_INV)[1]

#-------------------------------------------[Find Contour]------------------------------------------------------------------#

contrours1,hierarchy1 = cv.findContours(binary_gray,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
contrours2,hierarchy2= cv.findContours(binary_blur,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
contrours3,hierarchy3 = cv.findContours(binary_Median,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
contrours4,hierarchy4 = cv.findContours(binary_Gaussian,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

#-------------------------------------------[Counting Ojbect]------------------------------------------------------------------#

'''n = 1
for cnt in contrours2:
    print(cnt.shape)
    (x,y),radius = cv.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    cv.circle(img,center,radius,(0,255,0),2) # cv.circle(img,center,radius,(B,G,R),thickness)
    text= "#"+str(n) # giá trị in ra thay đổi sau mỗi vòng lặp, chuỗi + chuỗi nên phải ép kiểu n
    cv.putText(img, text,center,cv.FONT_HERSHEY_PLAIN,2,(0,0,255),1)
    n+=1'''

#-------------------------------------------[Show Image]-----------------------------------------------------------------------#
print('Gray # of object =',len(contrours1))
print('Blur # of object =',len(contrours2))
print('Median # of object =',len(contrours3))
print('Gaussian # of object =',len(contrours4))

cv.imshow('Origanal',img)
cv.imshow('Blur',blur)
cv.imshow('Gaussian',gaussian)
cv.imshow('Median',median)

cv.imshow('Binary Imgae',binary_gray)
cv.imshow('Binary Blur',binary_blur)
cv.imshow('Binary Gaussian',binary_Gaussian)
cv.imshow('Binary Median',binary_Median)

cv.waitKey(0)
cv.destroyAllWindows()
