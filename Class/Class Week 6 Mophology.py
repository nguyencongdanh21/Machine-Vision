import cv2 as cv
import numpy as np
img = cv.imread('img2/Salt_noise.jpg',cv.IMREAD_GRAYSCALE)
b_img = cv.threshold(img,150,255,cv.THRESH_BINARY)[1]

# create kernel
kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5)) # lay kich thuoc nhieu la hinh chu nhat
#mophology
#out = cv.erode(b_img,kernel,iterations=1) # bi soi mon khi xoa cac hat nho >1 xoa nhieu hon
#out = cv.dilate(out,kernel,iterations=1) # bu dap lai phan bi xoa >1 bu dap nhieu hon
# rut ngan mophology
out = cv.morphologyEx(b_img,cv.MORPH_OPEN,kernel,iterations=1)


cv.imshow('before',img)
cv.imshow('after',out)
cv.waitKey(0)
cv.destroyAllWindows()