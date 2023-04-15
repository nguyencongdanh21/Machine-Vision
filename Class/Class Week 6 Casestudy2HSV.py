import cv2 as cv
import numpy as np

low_H = 0; high_H = 180
low_S = 0; high_S = 255
low_V = 0; high_V = 255


def low_h(val):
    global low_H
    low_H = val
    b_img = cv.inRange(hsv, (low_H,low_S,low_V),(high_H,high_S,high_V))
    cv.imshow('Color_Filter', b_img)
def high_h(val):
    global high_H
    high_H = val
    b_img = cv.inRange(hsv, (low_H,low_S,low_V),(high_H,high_S,high_V))
def low_s(val):
    global low_S
    low_S = val
    b_img = cv.inRange(hsv, (low_H,low_S,low_V),(high_H,high_S,high_V))
    cv.imshow('Color_Filter', b_img)
def high_s(val):
    global high_S
    high_S = val
    b_img = cv.inRange(hsv, (low_H,low_S,low_V),(high_H,high_S,high_V))
    cv.imshow('Color_Filter', b_img)
def low_v(val):
    global low_V
    low_V = val
    b_img = cv.inRange(hsv, (low_H,low_S,low_V),(high_H,high_S,high_V))
    cv.imshow('Color_Filter', b_img)
def high_v(val):
    global high_V
    high_V = val
    b_img = cv.inRange(hsv, (low_H,low_S,low_V),(high_H,high_S,high_V))
    cv.imshow('Color_Filter', b_img)


img = cv.imread('img2/wood.jpg', cv.IMREAD_COLOR)
img = cv.GaussianBlur(img, (5,5), 0)

# change color space to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# h,s,v = cv.split(hsv)
cv.namedWindow('Color_Filter')
cv.createTrackbar('low_H','Color_Filter',0, 180, low_h)
cv.createTrackbar('high_H','Color_Filter',0, 180, high_h)
cv.createTrackbar('low_S','Color_Filter',0, 255, low_s)
cv.createTrackbar('high_S','Color_Filter',0, 255, high_s)
cv.createTrackbar('low_V','Color_Filter',0, 255, low_v)
cv.createTrackbar('high_V','Color_Filter',0, 255, high_v)


cv.waitKey(0)
cv.destroyAllWindows()