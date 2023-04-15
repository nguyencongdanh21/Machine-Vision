# library
import pickle
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier

#*******************************************************************************************************#
# 1.Image size 45x60 -> 1x2700 X_train = 120 x 2700                                                     #
# 2.Number of image:120                                                                                 #
# 3.labels filename[0] -> 'f'->0 : kết quả female -> Y_train =(1x120) 1 kết quả trong 120 phần tử       #
#                      -> 'm'->1 : kết quả male                                                         #
#       1                 2                  3                  4              5              6         #
# 4. LOAD IMAGE ---> PREPEROCESSOR ----> NORMALIZE[0,1] -----> RESHAPE ------> STACK -----> X_TRAIN     #
# 5. LOAD IMAGE ---> PREPEROCESSOR ----> NORMALIZE[0,1] -----> RESHAPE ------> STACK -----> X_Test      #                                                 
#                                                                                                       #
#*******************************************************************************************************#

#Get file list
img_folder = Path('img2/data train/Gender database')

test_folder=img_folder/'test'
train_folder= img_folder/'train'

train_images=list(train_folder.glob('*.png'))
test_images=list(test_folder.glob('*.png'))

    #glob tìm hàm theo wild card .png: nó sẽ tìm tất cả flie png
    #print(train_images)
    #Prepare training data

num_images=len(train_images)#-> để vào vòng for
num_images_test=len(test_images)
# Create Traing Data
X_train = np.empty((0,2700),dtype=float)
Y_train = np.empty(num_images,dtype=np.uint8)

#************************************************************************************************************************************************************************#

for i in range(num_images):
    # Get filepath -> đọc đc ảnh thì phải cvt sang dạng string
    img_path = str(train_images[i])
    #-1. Load images as gray
    img= cv.imread(img_path,cv.IMREAD_COLOR)
    img= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    #-2. Preprocess

    #-3. Normalize
    img= img.astype(float)
    cv.normalize(img,img,0,1.0,cv.NORM_MINMAX) #src = ảnh đầu vào , dst ảnh đầu ra và dùng số thức (giá trị từng pixel/max)

    #-4. Reshape -> vector hàng
    img= np.reshape(img,(1,2700))

    #-5. Stack
    X_train = np.vstack((X_train,img))

    #-6. Create Labels
    if train_images[i].stem[0] == 'f':   #stem trả về tên file
        Y_train[i] = 0    
    #if train_images[i].stem[0] == 'male':
    else:
        Y_train[i] = 1
    #---------------------------------------------------------------------------------------------------------------------------#

#************************************************************************************************************************************************************************#

# Create Test Data
X_test = np.empty((0,2700),dtype=float)
Y_test = np.empty(num_images_test,dtype=np.uint8)

for i in range(num_images_test):
    # Get filepath -> đọc đc ảnh thì phải cvt sang dạng string
    img_path = str(test_images[i])
    #-1. Load images as gray
    img2= cv.imread(img_path,cv.IMREAD_COLOR)
    img2= cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

    #-2. Preprocess

    #-3. Normalize
    img2= img2.astype(float)
    cv.normalize(img2,img2,0,1.0,cv.NORM_MINMAX) #src = ảnh đầu vào , dst ảnh đầu ra và dùng số thức (giá trị từng pixel/max)

    #-4. Reshape -> vector hàng
    img2= np.reshape(img2,(1,2700))

    #-5. Stack
    X_test = np.vstack((X_test,img2))

    #-6. Create Labels
    if test_images[i].stem[0] == 'f':   #stem trả về tên file
        Y_test[i] = 0    
    #if train_images[i].stem[0] == 'male':
    else:
        Y_test[i] = 1
    #---------------------------------------------------------------------------------------------------------------------------#

#************************************************************************************************************************************************************************#

#Creat NN model and train 
mlp = MLPClassifier(hidden_layer_sizes=(20,20),max_iter=300)
model=mlp.fit(X_train,Y_train)

#Testing model
y_predict = model.predict(X_test)
print('Accuracy is: ',metrics.accuracy_score(Y_test,y_predict))

#Display model
ConfusionMatrixDisplay.from_estimator(model,X_train,Y_train)
ConfusionMatrixDisplay.from_estimator(model,X_test,Y_test)
plt.show()

#Save model
filename='Gender Identified.sav'
pickle.dump(model,open(filename,'wb'))

