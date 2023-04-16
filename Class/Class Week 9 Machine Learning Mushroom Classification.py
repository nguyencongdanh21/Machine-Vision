#import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# adaboost library
from sklearn.ensemble import AdaBoostClassifier   
#from sklearn.ensemble import GradientBoostingClassifier   # Thuật toán Boosting
from sklearn import datasets                         # 
from sklearn.model_selection import train_test_split # Automatic split data for train và test
from sklearn import metrics # Caculate accuracy

import pickle

#read cvs to pandas data frame
mushrooms=pd.read_csv("img2/data text/mushrooms.csv")

#Create dummy variabels
mushrooms = pd.get_dummies(mushrooms)

#subset data into dependt and independent variables x,y
LABELS = ['class_e', 'class_p']
FEATURES = [a  for a in mushrooms.columns if a not in LABELS ]
y = mushrooms[LABELS[0]]
x= mushrooms[FEATURES]

# split data into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.3)

#Train data
number_of_estimators=40
#Vẽ biểu đồ
fig = plt.figure(figsize=(10,10))
ax0 =  fig.add_subplot(111)

accuracy =np.empty(40,dtype=float)
for i in np.arange(1,number_of_estimators+1):
    ada = AdaBoostClassifier(n_estimators=i,learning_rate=1)
    #Train the classifier
    model=ada.fit(X_train,Y_train)
    y_pred=model.predict(X_test)
    #Acuracy 0-39(40 lan)
    accuracy[i-1]=metrics.accuracy_score(Y_test,y_pred)
ax0.plot(np.arange(1,number_of_estimators+1),accuracy)
ax0.set_xlabel("# of weak learners")
ax0.set_ylabel('Accuracy')
plt.show()
