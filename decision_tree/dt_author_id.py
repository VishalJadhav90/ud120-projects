#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

def calculate_accuracy(results, labels_test):
    from sklearn.metrics import accuracy_score
    print(accuracy_score(results, labels_test))

def get_results_decision_tree(features_train, features_test, labels_train):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(min_samples_split=40)
    now = time()
    clf.fit(features_train, labels_train)
    print("time for training: {}".format(time()-now))
    #print("features train: {} len:{}".format(features_train[0], len(features_train[0])))
    results = clf.predict(features_test)
    return results

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
results = get_results_decision_tree(features_train, features_test, labels_train)
calculate_accuracy(results, labels_test)
#########################################################


