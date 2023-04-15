
import cv2 as cv
#from matplotlib import pyplot as plt


img = cv.imread('img2/coin3.jpg', cv.IMREAD_COLOR)#BGR
#img = cv.imread('rice.jpg', cv.IMREAD_COLOR)#BGR

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur= cv.GaussianBlur(gray,(5,5),3)

thresh = 125
b_img = cv.threshold(blur, thresh,255, cv.THRESH_BINARY_INV)[1]
countours, hierachy = cv.findContours(b_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
count = 0
for cnt in countours:
 area = cv.contourArea(cnt)
 if area > 300:
  count = count + 1
  (x,y),radius = cv.minEnclosingCircle(cnt)
  center = (int(x),int(y))
  radius = int(radius)
  cv.circle(img, center, radius,(0.210,0),2)
  text = '#'+str(count)
  cv.putText(img, text,center,fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(0,230.0))

print(len(countours))

cv.imshow('output', b_img)
cv.imshow('2output', img)
cv.waitKey(0)
cv.destroyAllWindows()

