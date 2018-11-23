
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("cars.csv")
type(data)
from sklearn.preprocessing import LabelEncoder
processed_data=pd.DataFrame()
lb_make = LabelEncoder()
processed_data["Classification"] = lb_make.fit_transform(data["Classification"])
processed_data["Engine Type"] = lb_make.fit_transform(data["Engine Type"])
processed_data["Height"] = lb_make.fit_transform(data["Height"])
processed_data["Horsepower"] = lb_make.fit_transform(data["Horsepower"])
processed_data["Length"] = lb_make.fit_transform(data["Length"])
processed_data["Hybrid"] = lb_make.fit_transform(data["Hybrid"])
processed_data["Torque"] = lb_make.fit_transform(data["Torque"])
processed_data["Width"] = lb_make.fit_transform(data["Width"])
processed_data["ID"] = lb_make.fit_transform(data["ID"])
features=processed_data.columns
##print(features)
​print(processed_data)
features_raw=processed_data.columns[3]
print(features_raw)
feature_data=processed_data[features_raw]
print(feature_data)
features1=features.drop(features_raw)
feature_data1=processed_data[features1]
print(feature_data1)
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(processed_data, feature_data,train_size=0.990, random_state = 0) 


# training a Naive Bayes classifier
 
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train, y_train) 
gnb_predictions = gnb.predict(X_test) 
# accuracy on X_test 
accuracy = gnb.score(X_test, y_test) 
print (accuracy )
print(y_test, gnb_predictions)

# training a KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier 
knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 
# accuracy on X_test 
accuracy = knn.score(X_test, y_test) 
print (accuracy )
knn_predictions = knn.predict(X_test)  
print(y_test, knn_predictions)

# training a DescisionTreeClassifier 

from sklearn.tree import DecisionTreeClassifier 
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 
accuracy = dtree_model.score(X_test, y_test) 
print (accuracy )
knn_predictions = knn.predict(X_test)  
print(y_test, dtree_predictions)

# training a SVC classifier

from sklearn.svm import SVC 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 
svm_predictions = svm_model_linear.predict(X_test) 
# model accuracy for X_test   
accuracy = svm_model_linear.score(X_test, y_test) 
print (accuracy )
print(y_test, svm_predictions)

​


