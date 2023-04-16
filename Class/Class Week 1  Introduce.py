import cv2 as cv
import numpy as np

img = cv.imread('img2/halloween.jpg', cv.IMREAD_COLOR)
cv.imshow('hello',img)
cv.waitKey(10000)
cv.destroyAllWindows()


