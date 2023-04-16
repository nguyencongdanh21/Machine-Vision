import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# split data
from sklearn.model_selection import train_test_split
#Load digits datasets số viết tay

from sklearn.datasets import load_digits
dataset=load_digits()
X= dataset['data']
y= dataset['target']
print(X.shape)

#Split data-Trainmodel
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3)
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
#Creart MLP (multi-layer perceptron)
    #max_iter : số vòng lặp tối đa
mlp= MLPClassifier(hidden_layer_sizes= (20,20),max_iter= 1000)

#training model
mlp.fit(X_train,Y_train)

y_predict = mlp.predict(X_test)
print(metrics.accuracy_score(Y_test,y_predict))

# Save model
import pickle
filename= "load_digit.sav"
pickle.dump(mlp, open(filename,"wb"))

ConfusionMatrixDisplay.from_estimator(mlp,X_test,Y_test)
plt.show()
