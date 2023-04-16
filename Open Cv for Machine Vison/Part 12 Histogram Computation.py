import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#-----------[LOAD IMAGE]-------------------------------------------------------------------------------#
img = cv.imread('img/coin2.jpg',cv.IMREAD_COLOR)
blank= np.zeros(img.shape[:2],dtype='uint8')
#img= cv.resize(img,(456,356))
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# 1 Circle
mask=cv.circle(blank,(img.shape[1]//3-40,img.shape[0]//3+36)
               ,50,255,-1) 
# 50 = diameter , img.shape[1] -> x , img,shape[0] -> y, black backgound 255
mask_img=cv.bitwise_and(img,img,mask=mask)

# GRAYSCALE HISTOGRAM #
gray_hist = cv.calcHist([gray],[0],mask,[256],[0,256]) #-> none nghĩa là ko gửi , còn muốn gửi vào phần nào thì thêm biến cần gửi ở đây tôi gửi vào mask
plt.figure()
plt.title('Gray Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.xlim([0,256])
plt.show()

# COLOR HISTOGRAM
colors = ('b','g','r')
for i,col in enumerate(colors):
    hist= cv.calcHist([img],[i],mask,[256],[0,256])
    plt.plot(hist,color=col)
    plt.xlim([0,256])
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.show()
#-----------[SHOW RESULT]------------------------------------------------------------------------------#

cv.imshow('Coin',img)
cv.imshow('Gray',gray)
cv.imshow('Mask',mask_img)
cv.waitKey(0)
cv.destroyAllWindows()