import cv2 as cv
import numpy as np
#-----------[LOAD IMAGE]-----------------------------------------------------------------------------#
img = cv.imread('img/coin2.jpg',cv.IMREAD_COLOR)
# 1. Translate
def translate(img,x,y): # ma trận dịch - Ma trận của biến đổi tuyến tính
    transMat =np.float32([[1,0,x],[0,1,y]])
    dimensions = (img.shape[1],img.shape[0])
    return cv.warpAffine(img,transMat,dimensions)
# -x -> Left
# -y -> Up
#  x -> Right
#  y -> Down
#Example 
translate=translate(img,-100,100) 

# 2.Rotation
def rotate(img,angle,rotPoint=None):
    (height,width) = img.shape[:2] # lấy 2 giá trị đầu (0,1) là y và x
    if rotPoint is None: # nếu điểm quay = none quay tại chỗ giữa tâm bức hình
        rotPoint = (width//2,height//2)
    rotMat = cv.getRotationMatrix2D(rotPoint,angle,1.0)
    dimensions= (width,height)
    return cv.warpAffine(img,rotMat,dimensions)
rotate=rotate(img,-45)

# 3. Resize
resized = cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)
# 4. Flipping
flipping= cv.flip(img,-1)
#6. Cropping
cropped=img[50:200,200:400]
 
#-----------[SHOW RESULT]----------------------------------------------------------------------------#
cv.imshow('Coin',img)
cv.imshow('Translation',translate)
cv.imshow('Rotate',rotate)
cv.imshow('Flipping',flipping)
cv.waitKey(0)
cv.destroyAllWindows()