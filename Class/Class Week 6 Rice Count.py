import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


#img = cv.imread('coin.jpg', cv.IMREAD_COLOR)#BGR
img = cv.imread('img2/rice.jpg', cv.IMREAD_COLOR)#BGR

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray,(5,5),1)
thresh = 125
b_img = cv.threshold(blur, thresh,255, cv.THRESH_BINARY)[1]
countour, hierachy = cv.findContours(b_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

count = 0

'''for cnt in countour:
 area = cv.contourArea(cnt)
 if area > 300:
  count = count + 1
  (x,y),radius = cv.minEnclosingCircle(cnt)
  center = (int(x),int(y))
  radius = int(radius)
  cv.circle(img, center, radius,(0.210,0),2)
  text = '#'+str(count)
  cv.putText(img, text,center,fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(0,230.0))
print(img)'''

for cnt in countour:
    #print (cv.contourArea(cnt))
    (x,y),radius=cv.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius=int(radius)
    cv.circle(img, center,radius,(0,255,0),2)
    cv.putText(img,text="#"+str(cnt),org=(int(x),int(y)),fontFace=cv.FONT_HERSHEY_COMPLEX,fontScale=1,color=(255,0,255))
    cnt=cnt+1
    
print(len(countour))
cv.imshow('output', b_img)
cv.waitKey(0)
cv.destroyAllWindows()
