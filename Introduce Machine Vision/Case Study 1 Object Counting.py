import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
#------------------------------------[LECTURE]----------------------------------------------------------------#
''' 5 STEP FOR OBJECT COUTING 
    1-> LOAD IMAGE
    2-> CONVERT TO GRAY IMAGE
        -> 2.1
    3-> IMAGE BINARIZATION
    4-> CONTOUR DECTION
        -> 4.1 CONTOUR AREA
            -> Minimum Enclosing Circle :
                (x,y),radius = cv.minEnclosingCircle(cnt)
                center = (int(x),int(y))
                radius = int(radius)
                cv.circle(img,center,radius,(0,255,0),2)
        -> 4.2 PUT TEXT OR COMMENT
            -> putText()
                cv.putText(img,text,org,fontFace,fontScale,color[,thickness[,lineType[,bottomLeftOrigin]]])->img

    5-> COUNTING
'''
#------------------------------------[END]--------------------------------------------------------------------#

#------------------------------------[LOAD IMAGE]-------------------------------------------------------------#
img = cv.imread('img3/HoughCircles.jpg',cv.IMREAD_COLOR)
averaging = cv.blur(img,(5,5))
#------------------------------------[CONVERT TO GRAY IMAGE]--------------------------------------------------#
gray = cv.cvtColor(averaging, cv.COLOR_BGR2GRAY)

#------------------------------------[IMAGE BINARIZATION]-----------------------------------------------------#
'''
* cv.threshold (nhập giá trị ngưỡng bằng tay)

-> https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

-> cv.threshold(src, thresh, maxval, type[,dst])->retval, dst
    -> src	 :  gray
    -> dst	 :  [1]
    -> thresh:	tùy chọn
    -> maxval:	255
    -> type	 :  THRESH_BINARY_INV or THRESH_BINARY or THRESH_OTSU

* cv.adaptiveThreshold (chia tấm hình ra làm nhiều tấm nhỏ)
'''
# find threshold by matplotlib
plt.hist(gray.ravel(),256,[0,256]);plt.show()

thresh= 127
maxval= 255
binary_img = cv.threshold(gray,thresh,maxval,cv.THRESH_BINARY_INV)[1]
#------------------------------------[CONTOUR DECTION]--------------------------------------------------------#
''' 
cv.findContours()
cv.findContours(image, mode,method[,contours[,hierarchy[,offset]]])->image, contours, hierarchy

* ContourApproximationModes
    -> CHAIN_APPROX_NONE : lấy hết tất cả các điểm
    -> CHAIN_APPROX_SIMPLE : lấy các điểm đầu mút

* RetrievalModes
    -> RETR_EXTERNAL: trả về đường contour nằm phía ngoài cùng,
    -> RETR_LIST: trả về tất cả các đường contour, không thể hiện hierachy
    -> RETR_TREE: trả về tất cả các đường contour, có xây dựng hierachy
'''
contrours,hierarchy = cv.findContours(binary_img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
#------------------------------------[COUNTING]---------------------------------------------------------------#
n = 1
for cnt in contrours:
    print(cnt.shape)
    (x,y),radius = cv.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    cv.circle(img,center,radius,(0,255,0),2) # cv.circle(img,center,radius,(B,G,R),thickness)
    text= "#"+str(n) # giá trị in ra thay đổi sau mỗi vòng lặp, chuỗi + chuỗi nên phải ép kiểu n
    cv.putText(img, text,center,cv.FONT_HERSHEY_PLAIN,2,(0,0,255),1)
    n+=1  
#------------------------------------[Result]-----------------------------------------------------------------#

print(type(contrours))
print(type(hierarchy))
print('# of object =',len(contrours))


cv.imshow('Original Image',img)
cv.imshow('Gray Image',gray)
cv.imshow('Binary Image',binary_img)
cv.imshow('Averaging',averaging)
cv.waitKey(0)
cv.destroyAllWindows()




