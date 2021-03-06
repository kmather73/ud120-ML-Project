#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]


#########################################################
### your code goes here ###

from sklearn import svm
rbf = svm.SVC(kernel='rbf', C=10000)
#rbf = svm.SVC(kernel='linear')
rbf.fit(features_train, labels_train)

#print( rbf.score(features_test, labels_test))

print sum(rbf.predict(features_test))
#########################################################
