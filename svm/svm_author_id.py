#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.svm import SVC
clf = SVC(kernel="rbf", C=10000)
features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]
now = time()
clf.fit(features_train, labels_train)
print("fitting time:{}".format(time()-now))
now = time()
result = clf.predict(features_test)
print("prediction time:{}".format(time()-now))

print("10th:{}".format(result[10]))
print("26th:{}".format(result[26]))
print("50th:{}".format(result[50]))

from sklearn.metrics import accuracy_score
print("accuracy: {}".format(accuracy_score(result, labels_test)))
#########################################################


