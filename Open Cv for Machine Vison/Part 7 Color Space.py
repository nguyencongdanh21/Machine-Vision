import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#-----------[LOAD IMAGE]-----------------------------------------------------------------------------#
# import Color img
img = cv.imread('img/coin.jpg',cv.IMREAD_COLOR)

# BGR to Gray imgae
# Note: Gray imgae can't covenrt into HSV color ->BGR->HSV 
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# BRG to HSV (for hue, saturation, value)
hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)

# BRG to L*a*b 
# Không gian màu CIE Lab* là không gian màu có sự đồng đều trong dải màu sắc
# a* và b*. Màu có giá trị a* dương thì ngả đỏ, màu có giá trị a* âm thì ngả lục. 
# Tương tự b* dương thì ngả vàng và b* âm thì ngả lam.
# Còn độ sáng của màu thì thay đổi theo trục dọc (L*).
lab= cv.cvtColor(img,cv.COLOR_BGR2LAB)
lab=cv.cvtColor(img,cv.COLOR_BGR2Lab)

#-----------[SHOW RESULT]----------------------------------------------------------------------------#
cv.imshow('Coin',img)
plt.imshow(img) # matplotlib -> inverse color of img
plt.show()

cv.imshow('Coin',img)
cv.imshow('Gray',gray)
cv.imshow('HSV',hsv)
cv.imshow('LAB',lab)

cv.waitKey(0)
cv.destroyAllWindows()