import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#---------------------------------------------------------------------------------------------#
img = cv.imread('img2/rice.jpg',cv.IMREAD_COLOR)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#plt.imshow(gray, cmap='gray')
#---------------------------------------------------------------------------------------------#
blur = cv.GaussianBlur(gray,(7,7),0)
canny = cv.Canny(blur, 30, 150, 3)
dilated = cv.dilate(canny, (1, 1), iterations=0)
(cnt, hierarchy) = cv.findContours(
    dilated.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.drawContours(rgb, cnt, -1, (0, 255, 0), 2)


#print(img)

#---------------------------------------------------------------------------------------------#

print("rices in the image : ", len(cnt))
cv.imshow('output', img)
cv.waitKey(0)
cv.destroyAllWindows()
# homework lam lai bang code cua thay

