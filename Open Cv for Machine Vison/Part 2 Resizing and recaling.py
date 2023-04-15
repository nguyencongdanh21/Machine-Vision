import cv2 as cv
#-----------[Function]-------------------------------------------------------------------------#

# Tự tạo hàm Rescale 
def rescaleFrame(frame,scale=0.75): # scale 0.75 là 75% của ảnh gốc
    width= int(frame.shape[1]*scale)  # Trục x 1 -> trong python trục y là 0 trục x là 1
    height =int(frame.shape[0]*scale)  # Trục Y 2
    dimensions =(width,height)
    # interpolation là nội suy
    return cv.resize(frame,dimensions, interpolation=cv.INTER_AREA)
# Tự tạo hàm Revolution
def changeRes(width,height):
    # for live video,WebCam  
    capture.set(3,width)
    capture.set(4,height)
#-----------[LOAD IMAGE]-----------------------------------------------------------------------#

img=cv.imread('img/zelda.jpg',cv.IMREAD_COLOR) 
img_resize=rescaleFrame(img)

#-----------[LOAD VIDEO]-----------------------------------------------------------------------#

capture =cv.VideoCapture('video/Animatronic Eye.mp4')

# read video frame by frame
while True:
    isTrue,frame= capture.read()
    frame_resized= rescaleFrame(frame)
    cv.imshow('Video',frame)
    cv.imshow('Video Resize',frame_resized)
    # stop video 20: time of video, d is stop button
    if cv.waitKey(100) &  0xFF == ord('d'): #ord trả về mã Unicode
        cv.destroyAllWindows()
        break

#-----------[SHOW RESULT]-----------------------------------------------------------------------#
cv.imshow('True size',img)
cv.imshow('Resize',img_resize)
cv.waitKey(0)
cv.destroyAllWindows()