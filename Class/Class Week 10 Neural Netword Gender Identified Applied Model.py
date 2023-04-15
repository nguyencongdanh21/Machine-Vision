import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.neural_network import MLPClassifier

#*****************************[DEPENDENT OF TRAIN PROCESS MODEL]******************************************#
#                                                                                                         #
# DEFIND FACE AREA ---> CROP AREA ----> RESIZE ----> PREPROCESS---->NORMALIZER---->RESHAPE----->PREDICT   #
#                                                                                                         #
#*********************************************************************************************************#


#Select model
model = pickle.load(open('model/Gender Identified.sav','rb'))

#IMG TEST
# Detect Human faces
face_cascade = cv.CascadeClassifier()
faces = face_cascade.load('model/humanFaceCascade.xml')

# Read image
img = cv.imread('img2/Group3.jpg',cv.IMREAD_COLOR)
#img = cv.resize(img,(960, 600))
 #Convert to gray image
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5,5), 0)

faces = face_cascade.detectMultiScale(gray)

for(x,y,w,h) in faces:
    #Crop roi
    roi = gray[y-int(h/5):y+h+int(h/10), x-int(w/10):x+w+int(w/10)]
    #Resize to 45x60
    roi = cv.resize(roi, (45,60))
    #Normalize
    roi = roi.astype(float)
    cv.normalize(roi,roi,0,1.0,cv.NORM_MINMAX)
    #Reshape
    input = np.reshape(roi, (1,2700))
    #Predict
    result = model.predict(input)
    if result == 0:
        text="Female"
    else:
        text="Male"
    cv.rectangle(img, (x,y), (x+w,y+h), (0,0,255),2)
    cv.putText(img,text,(x,y),cv.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)    
cv.imshow('Result',img)
cv.waitKey(0)
cv.destroyAllWindows()
#Vebcam Test

