import numpy as np

#Adaboost Library
from sklearn.ensemble import AdaBoostClassifier      # Thuật toán Boosting
from sklearn import datasets                         # Iris data
from sklearn.model_selection import train_test_split # Automatic split data for train và test
from sklearn import metrics # Caculate accuracy

#Thư viên chuyên để import model
import pickle

#Load dataset
iris = datasets.load_iris()
X = iris.data # feature values
Y = iris.target #labels

# print(type(X))
# print(type(y))
# print(X.shape)
# print(y.shape)
# print(X)
# print(y)

# Split data into training data and test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)

# Create Adaboost classifier
    # n_estimators số lượng bộ phân loại, learing rate tốc độ thay đổi trọng số W
    # Mỗi lần thay đổi sẽ lớn -> quá trình train kết thúc nhanh -> độ chính xác giảm
ada = AdaBoostClassifier(n_estimators=30,learning_rate=1) 

# Train the classifier
model = ada.fit(X_train,Y_train)

# Use classifier
    #accuray_score lấy giá trị bộ test(chuẩn bị từ trước) và bộ phần loại -> so sánh coi giống bao nhiêu %
    # Lần 1 : Accuracy =  0.9111111111111111
    # Lần 2 : Accuracy =  0.9333333333333333
y_pred = model.predict(X_test)
print('Accuracy = ', metrics.accuracy_score(Y_test,y_pred))

# Save model
    #Lấy model đã training xuống vận hành ở các thiết bị ngoại vi khác
filename= "Iris_classifier.sav"
    #dump hàm chung của hàm xuất data ra ngoài
    #model cái ta muốn xuất ra ngoài
    #open mở file muốn sử dụng xuống dạng nhị phân
    #wb : dạng nhị phân (write binary)
pickle.dump(model, open(filename,"wb"))




