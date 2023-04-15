import numpy as np
import cv2 as cv

capture =cv.VideoCapture(0)
while True:
    isTrue,frame= capture.read()
    cv.imshow('video',frame)
    # stop video 20: time of video, d is stop button
    if cv.waitKey(20) &  0xFF == ord('d'): #ord trả về mã Unicode
        cv.destroyAllWindows()
        break

capture.release()

cv.waitKey(0)
cv.destroyAllWindows()