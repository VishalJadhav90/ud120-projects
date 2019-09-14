#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
import numpy as np
sys.path.append("../tools/")
from email_preprocess import preprocess

def calculate_accuracy(results, labels_test):
    from sklearn.metrics import accuracy_score
    print("accuracy: {}".format(accuracy_score(results, labels_test)))

def get_results_svm(features_train, features_test, labels_train):
    from sklearn.svm import SVC
    clf = SVC(kernel="rbf", C=10000)
    # features_train = features_train[:len(features_train)/100]
    # labels_train = labels_train[:len(labels_train)/100]
    now = time()
    clf.fit(features_train, labels_train)
    print("fitting time:{}".format(time() - now))
    now = time()
    result = clf.predict(features_test)
    print(type(result))
    print("prediction time:{}".format(time() - now))
    count = 0
    for res in result:
        # print("res:{}".format(res))
        if res == 1:
            count += 1
    print("count:", count)
    return result

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
result = get_results_svm(features_train, features_test, labels_train)
calculate_accuracy(result, labels_test)

#########################################################


