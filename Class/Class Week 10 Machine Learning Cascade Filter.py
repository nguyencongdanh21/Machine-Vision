import numpy as np
import cv2 as cv

cap = cv.VideoCapture('video/robocup.mp4')
if not cap.isOpened():
    print('Cannot open video clip/camera')
    exit()

#Load casacade filter
ball_cascade = cv.CascadeClassifier()
ball_cascade.load('model/cascade.xml')
while True:
    # read frame by frame
    # đọc từng khung hình
    ret, frame = cap.read()
    if not ret:
        print(' can not read video frame. Video ended?')
        break
    # your code
    # tìm tất cả các đối tượng mà model được training
    balls = ball_cascade.detectMultiScale(frame)
    # lần lượt là 4 giá tri x,y chiều rộng chiều cao
    for (x,y,w,h) in balls:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,0,255))
    cv.imshow('video', frame)
    # close clip
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()