import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    frame = cv.GaussianBlur(frame, (5,5), 0)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    b_img = cv.inRange(hsv,(85,110,70),(120,255,255))
    # Use morphology to remove noise in the binary image
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
    # b_img = cv.morphologyEx(b_img,cv.MORPH_CLOSE,kernel,iterations=2)
    b_img = cv.morphologyEx(b_img,cv.MORPH_OPEN,kernel,iterations=1)

    contours, hierachy = cv.findContours(b_img,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv.contourArea(cnt)>100:
            x,y,w,h = cv.boundingRect(cnt)
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    # Display the resulting frame
    cv.imshow('frame', frame)
    # cv.imshow('binary', b_img)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()