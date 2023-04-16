import cv2 as cv

#-----------[LOAD IMAGE]-----------------------------------------------------------------------------#
# import Color img
img = cv.imread('img/coin2.jpg',cv.IMREAD_COLOR)
#-----------[PROCESSING IMAGE]-----------------------------------------------------------------------#
# Converting to grayscale
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

#1. Blur
#kernel = 3x3,sigma = 1 hoặc cv.BORDER_DEFAULT

blur = cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT) #-> need to learn about it 
blur2 = cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)

#2. Edge Cascade
canny=cv.Canny(img,40,175) # 2 threshold value -> need to learn about it 

#3. Dilating the image Tăng độ dầy
dilated = cv.dilate(canny,(7,7),iterations=2)

#4. Eroding the image Xói mòn
eroding=cv.erode(canny,(3,3),iterations=2) 

#5. Resize
resized = cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)

#6. Cropping
cropped=img[50:200,200:400]

#-----------[SHOW RESULT]----------------------------------------------------------------------------#

cv.imshow('Coin',img)
cv.imshow('Gray Picture',gray)
cv.imshow('Blur(3x3) picture',blur)
#cv.imshow('Blur(7x7) picture',blur)
cv.imshow('Canny Edges',canny)
cv.imshow('Dilated',dilated)
cv.imshow('Eroding',eroding)
#cv.imshow('Resize',resized)
cv.imshow('Cropped',cropped)

cv.waitKey(0)
cv.destroyAllWindows()