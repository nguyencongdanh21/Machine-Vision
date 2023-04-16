import numpy as np
import cv2 as cv

cap = cv.VideoCapture('video/robocup.mp4')
if not cap.isOpened():
    print('can not open video clip/camera')
    exit()

# Load cascade filter
ball_cascade = cv.CascadeClassifier()
ball_cascade.load('model/cascade.xml')


while True:
    # read frame by frame
    ret, frame = cap.read()
    if not ret:
        print(' can not read video frame. Video ended?')
        break
    # your code
    balls = ball_cascade.detectMultiScale(frame)
    for (x,y,w,h) in balls:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))


    cv.imshow('video', frame)
    # close clip
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()