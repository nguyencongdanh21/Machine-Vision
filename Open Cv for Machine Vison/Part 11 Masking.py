import cv2 as cv
import numpy as np
#-----------[LOAD IMAGE]-------------------------------------------------------------------------------#
img=cv.imread('img/coin2.jpg',cv.IMREAD_COLOR)
img_2=cv.imread('img/dice.jpg',cv.IMREAD_COLOR)
blank= np.zeros(img.shape[:2],dtype='uint8')
blank2= np.zeros(img_2.shape[:2],dtype='uint8')
#-----------[MASK]-------------------------------------------------------------------------------------#
# 1 Circle
mask=cv.circle(blank,(img.shape[1]//3-40,img.shape[0]//3+36)
               ,50,255,-1) 
# 50 = diameter , img.shape[1] -> x , img,shape[0] -> y, black backgound 255
mask_img=cv.bitwise_and(img,img,mask=mask)

#2 Rectangle 
mask_2=cv.rectangle(blank2,(img_2.shape[1]//2,img_2.shape[0]//2),
                           (img_2.shape[1]//2+100,img_2.shape[0]//2+100),
                           255,-1)
mask_img_2=cv.bitwise_and(img_2,img_2,mask=mask_2)
#-----------[SHOW RESULT]------------------------------------------------------------------------------#
cv.imshow('Coin',img)
cv.imshow('Mask Image',mask)
cv.imshow('Mask with img',mask_img)

#cv.imshow('Dice',img_2)
cv.imshow('Mask Image 2',mask_2)
#cv.imshow('Mask with img',mask_img_2)

cv.waitKey(0)
cv.destroyAllWindows()
