import pickle
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

#Open model trained
model = pickle.load(open('model/Iris_classifier.sav','rb'))
# data là  gia trị số liệu mới 
y_pred = model.predict(data)