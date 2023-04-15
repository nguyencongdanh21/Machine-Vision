import cv2 as cv
import numpy as np

#img = cv.imread('noise1.jpg',cv.IMREAD_COLOR)
img = cv.imread('img2/rice.jpg',cv.IMREAD_COLOR)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#------------------------------------------------------------------------------------------------------------------#
#blur/mean filter
#out1= cv.blur(gray, (3,3))
#out2= cv.blur(gray, (5,5))
out1= cv.blur(gray, (7,7))

#Gaussian blur 
out2=cv.GaussianBlur(gray,(7,7),0) # tang sigma gia tri lam mo giam
'''kernel = cv.getGaussianKernel(7,0)
kerne2 = cv.getGaussianKernel(7,5)
print("Kernel1 \n",kernel)
print("kernel2 \n",kerne2)'''


# Median blur
out3=cv.medianBlur(gray,5) # khong nen di qua 5



#------------------------------------------------------------------------------------------------------------------#
cv.imshow('o1',out1)
cv.imshow('o2',out2)
cv.imshow('o3',out3)
cv.waitKey(0)
cv.destroyAllWindows()

