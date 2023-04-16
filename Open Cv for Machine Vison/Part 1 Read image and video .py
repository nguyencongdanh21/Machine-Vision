import numpy as np
import cv2 as cv
#----------------------------[Read Image]--------------------------------------------------------#
img = cv.imread('img/coin.jpg')
color = cv.imread('img/coin.jpg',cv.IMREAD_COLOR) # read color picture B G  R
gray = cv. imread('img/coin.jpg',cv.IMREAD_GRAYSCALE) # read gray picture
#----------------------------[Read Video]--------------------------------------------------------#
#capture =cv.VideoCapture('video/Animatronic Eye.mp4')
capture =cv.VideoCapture(0)
gray = cv.cvtColor(capture,cv.COLOR_BGR2GRAY)
# read video frame by frame
while True:
    isTrue,frame= capture.read()
    cv.imshow('video',frame)
    # stop video 20: time of video, d is stop button
    if cv.waitKey(20) &  0xFF == ord('d'): #ord trả về mã Unicode
        cv.destroyAllWindows()
        break
#----------------------------[show Result]-------------------------------------------------------#
# capture =cv.VideoCapture(0)
# -> camera 0 , 1 camera 1 etc...

#coin
cv.imshow('coin',img)
cv.imshow('color',color)
cv.imshow('gray',gray)

#video
capture.release()

cv.waitKey(0)
cv.destroyAllWindows()
