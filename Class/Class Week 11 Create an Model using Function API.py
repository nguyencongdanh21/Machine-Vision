import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.utils.np_utils import to_categorical
# Prepare data
# load data mnist
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
#print(X_train.shape) #--> not 28,28,1
#print(Y_train) #--> label not correct with output (vector 0 and 1)

# Reshape and normalizer
X_train = X_train.reshape((X_train.shape[0],28,28,1)).astype('float32')/255
X_test = X_test.reshape((X_test.shape[0],28,28,1)).astype('float32')/255

Y_train = to_categorical(Y_train,10)
Y_test = to_categorical(Y_test,10)
print(X_train.shape) 
print(Y_train)

# Create Model using Function API
# Input data size 28x28 
input = tf.keras.Input(shape=(28,28,1))
# Convolutional I size 24x24 
x = layers.Conv2D(10,5,strides=(1,1),padding='valid',activation='relu')(input) # đầu vào input
x = layers.MaxPool2D(pool_size=(2,2),strides=2)(x) # đầu vào là x
# Convolutional II size 8x8 
x = layers.Conv2D(20,5,strides=(1,1),padding='valid',activation='relu')(x)
x = layers.MaxPool2D(pool_size=(2,2),strides=2)(x)
# Dropout Size 4x4 and Flatten Maxtri to Vector
x = layers.Dropout(0.25)(x)
x = layers.Flatten()(x)
# Output and # Neural Network
x = layers.Dense(100,activation='relu')(x)
output = layers.Dense(100,activation = 'softmax')(x)

model = tf.keras.Model(inputs=input,outputs=output)

# Show neural network
model.summary()
# Draw Graph
tf.keras.utils.plot_model(model,'model.png',show_shapes=True)

# Compile model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=["accuracy"])

# Training model
# Data đầu vào 60000
# epochs = số lần học , epochos = 10 ; 10(lần)x64(nhóm nhỏ)
# batch_size = khối lượng học, 64 nhóm nhỏ (48000/64 = 750 image) 
# Total image fro training = 10 x 750 = 7500 imgae -> chưa phải toàn bộ data
# validation_split= 20% ktra bước(12000 image), 80% training (48000 image)
# verbose = 1 hiển thị tất cả các bước training

history = model.fit(X_train,Y_train,epochs=10,batch_size=64,verbose=1,validation_split=0.2)
score = model.evaluate(X_test,Y_test, verbose=2)
print("loss = ",score[0])
print('accuracy = ',score)

# Test
# số 1 ở trước là 1 tấm ảnh
x = tf.ones((1,28,28,1))
model.predict(x)