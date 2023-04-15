import cv2 as cv
import numpy as np
#-----------[LOAD IMAGE]-----------------------------------------------------------------------#
# blank img
blank=np.zeros((720,800,3), dtype ='uint8') # y,x,z z là số màu 3: blue , green ,red
img=cv.imread('img/coin2.jpg')


#-----------[DRAWING SHAPE AND TEXT]-----------------------------------------------------------------------#
#1. Paint the img a certain colour
blank[200:300,300:400]=255,0,0 # range of picture 200:300,300:400
#blank[300:400,400:500]=0,255,0 # màu xanh

#2 . Draw a Rectangle
#cv.rectangle(blank,(0,0),(250,800),(0,255,0),thickness=cv.FILLED) #similar to -1
#cv.rectangle(blank,(0,0),(250,800),(0,255,0),thickness=-1)
cv.rectangle(blank,(0,0),(blank.shape[1]//2,blank.shape[0]//2),(0,255,0),thickness=2)

#3. Draw a Cirle
cv.circle(blank,(blank.shape[1]//2,blank.shape[0]//2),40,(0,255,0),thickness=2)

#4. Draw a Line
cv.line(blank,(0,0),(720//2,800//2),(255,255,255))
#5. Write Text
cv.putText(blank,'Hello there',(255,255),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0))
cv.putText(img,"A lot of coin,isn't ??",(255,255),cv.FONT_HERSHEY_COMPLEX,1.0,(0,255,0))
#-----------[SHOW RESULT]-----------------------------------------------------------------------#
#blue and green in blank 
cv.imshow('Rectagle',blank)

cv.imshow('Coin',img)
cv.waitKey(0)
cv.destroyAllWindows()
