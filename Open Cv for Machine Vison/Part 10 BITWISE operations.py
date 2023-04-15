import cv2 as cv
import numpy as np
#-----------[LOAD IMAGE]-------------------------------------------------------------------------------#
blank = np.zeros((400,400),dtype='uint8')
#-----------[BITWISE OPERATIONS]-----------------------------------------------------------------------#
rectangle = cv.rectangle(blank.copy(),(30,30),(370,370),255,-1)
circle= cv.circle(blank.copy(),(200,200),200,255,-1)
#1 BITWISE AND
bitwise_and=cv.bitwise_and(circle,rectangle)
#2 BITWISE OR
bitwise_or=cv.bitwise_or(circle,rectangle)
#3 BITWISE XOR
bitwise_xor=cv.bitwise_xor(circle,rectangle)
#4 BITWISE NOT
bitwise_not=cv.bitwise_not(circle)

#-----------[SHOW RESULT]------------------------------------------------------------------------------#
cv.imshow('Rectagle',rectangle)
cv.imshow('Circle',circle)
cv.imshow('BITWISE AND',bitwise_and)
cv.imshow('BITWISE OR',bitwise_or)
cv.imshow('BITWISE XOR',bitwise_xor)
cv.imshow('BITWISE NOT',bitwise_not)
cv.waitKey(0)
cv.destroyAllWindows()