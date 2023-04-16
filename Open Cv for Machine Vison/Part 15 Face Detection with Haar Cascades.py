import cv2 as cv
import numpy as np
#-----------[LOAD IMAGE]-----------------------------------------------------------------------------------------------------------------------#
img = cv.imread('img/class2.jpg',cv.IMREAD_COLOR)
img = cv.resize(img,(1156,856))
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#gaussian_blur= cv.GaussianBlur(gray,(3,3),1)
#-----------[FACE DETECTION]-----------------------------------------------------------------------------------------------------------------#
haar_cascade= cv.CascadeClassifier('xml file/haar_face.xml')
face_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=2)

for (x,y,w,h) in face_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
#-----------[SHOW RESULT]-----------------------------------------------------------------------------------------------------------------#
print(f'Number of face found = {len(face_rect)}')
cv.imshow('Gray',gray)
cv.imshow('Class',img)
cv.waitKey(0)
cv.destroyAllWindows()