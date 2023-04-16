import cv2 as cv
import numpy as np
img = cv.imread('img2/Small_holes.jpg',cv.IMREAD_GRAYSCALE)
b_img = cv.threshold(img,150,255,cv.THRESH_BINARY)[1]

# create kernel
kernel = cv.getStructuringElement(cv.MORPH_RECT,(9,9)) # lay kich thuoc nhieu la hinh chu nhat
#mophology -> anh nhi phan khong on thi nen su dung morphology de xu li

#out = cv.erode(b_img,kernel,iterations=1) # bi soi mon khi xoa cac hat nho >1 xoa nhieu hon
#out = cv.dilate(out,kernel,iterations=1) # bu dap lai phan bi xoa >1 bu dap nhieu hon
# rut ngan mophology
out = cv.morphologyEx(b_img,cv.MORPH_CLOSE,kernel,iterations=1)
# open xu li ben ngoai
# close xu li cac lo ben trong
# tang kich thuoc kernel nhanh hon vong lap
out = cv.morphologyEx(b_img,cv.MORPH_GRADIENT,kernel,iterations=3) # lay duong vien

cv.imshow('before',img)
cv.imshow('after',out)
cv.waitKey(0)
cv.destroyAllWindows()
