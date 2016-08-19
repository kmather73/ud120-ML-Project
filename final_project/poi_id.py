#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot

def plotFeatures(i, j, data, features_list):    
    for point in data:
        feature_i = point[i]
        feature_j = point[j]
    
        matplotlib.pyplot.scatter( feature_i, feature_j )

    matplotlib.pyplot.xlabel(features_list[i])
    matplotlib.pyplot.ylabel(features_list[j])
    matplotlib.pyplot.show()

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Find possible features with enough data
features_list = ['poi','salary','to_messages','deferral_payments','total_payments','loan_advances',\
                 'bonus','restricted_stock_deferred', 'deferred_income','total_stock_value',\
                 'expenses','from_poi_to_this_person', 'exercised_stock_options', 'from_messages',\
                 'long_term_incentive', 'shared_receipt_with_poi']

features_list_freq = []
for feature in features_list:
    countOfMissing = 0.0
    for key in data_dict:
        if data_dict[key][feature] == 'NaN':
            countOfMissing += 1
    
    features_list_freq.append(1.0 - countOfMissing/len(data_dict))
    print "For the feature "+feature+" we have "+str(countOfMissing)+\
          " many NaN out of "+str(len(data_dict))+" possible"

features_with_enough_data=[]
for i in range(len(features_list)):
    if features_list_freq[i] > .55:
        features_with_enough_data.append( features_list[i] )
print len(features_with_enough_data)        
print "\nremaining possible features are:"
print features_with_enough_data

#features_list = features_with_enough_data
features_list = ['poi',  'salary', 'bonus', 'total_stock_value', 'expenses', 'exercised_stock_options'] 
print "\nFeatures selected are:"
print features_list
### Task 2: Remove outliers
data = featureFormat(data_dict, features_list, sort_keys = True)

plotFeatures(1, 2, data, features_list)
plotFeatures(1, 3, data, features_list)
plotFeatures(1, 4, data, features_list)
plotFeatures(1, 5, data, features_list)


### Find possible outliers
toDelete = []
for key in data_dict:
    if data_dict[key]["salary"] != 'NaN' and data_dict[key]["salary"] > 1e7:
        toDelete.append( key )        
        
for key in toDelete:
    del data_dict[key]
print "Outliers are ", toDelete 

data = featureFormat(data_dict, features_list, sort_keys = True)

plotFeatures(1, 2, data, features_list)
plotFeatures(1, 4, data, features_list)
plotFeatures(1, 5, data, features_list)
plotFeatures(1, 3, data, features_list)
plotFeatures(5, 3, data, features_list)
plotFeatures(4, 3, data, features_list)
plotFeatures(3, 4, data, features_list)



### Task 3: Create new feature(s)
import math
for key in data_dict:
    if data_dict[key]["salary"] != 'NaN' and  data_dict[key]["bonus"] != 'NaN':
        data_dict[key]["BSR"] =  data_dict[key]["bonus"] / math.log(data_dict[key]["salary"])
    else:
        data_dict[key]["BSR"] = 'NaN'

    if data_dict[key]['shared_receipt_with_poi'] != 'NaN' and data_dict[key]['from_poi_to_this_person'] != 'NaN':
        data_dict[key]["COM"] = (data_dict[key]['shared_receipt_with_poi']+3*data_dict[key]['from_poi_to_this_person'])/data_dict[key]['to_messages']
    else:
        data_dict[key]["COM"] = 'NaN'

features_list.append( "BSR" )
features_list.append( "COM" )


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
#from sklearn.decomposition import PCA
#from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import  decomposition
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation

def cvFitClassifier(clfName, clf):
    cvRounds = 15
    scores = cross_validation.cross_val_score( clf, features, labels, cv=cvRounds,scoring='accuracy')
    accuracy = sum(scores)/len(scores)

    scores = cross_validation.cross_val_score( clf, features, labels, cv=cvRounds,scoring='precision')
    precision = sum(scores)/len(scores)

    scores = cross_validation.cross_val_score( clf, features, labels, cv=cvRounds,scoring='recall')    
    recall = sum(scores)/len(scores)    
    
    scores = cross_validation.cross_val_score( clf, features, labels, cv=cvRounds,scoring='f1')    
    fScore = sum(scores)/len(scores)
    
    print clfName +" Stats:"
    print "accuracy:", accuracy
    print "precision:", precision
    print "recall:", recall
    print "F1:", fScore
    print
    

pca = decomposition.PCA()
clf = GaussianNB()
cvFitClassifier("Naive Bayes", clf)


clf = tree.DecisionTreeClassifier( min_samples_split=5, min_samples_leaf=4)
cvFitClassifier("Decision Tree", clf)

scaler = MinMaxScaler(feature_range = [0,1])
estimators = [('pca', pca), ('scaler', scaler), ('svm', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False))]
clf = Pipeline(estimators)
cvFitClassifier("SVM", clf)


'''
param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "n_estimators": range(1, 50)
             }
DTC = DecisionTreeClassifier()
ABC =  AdaBoostClassifier(n_estimators=45, base_estimator = DTC)
estimators = [('AB', GridSearchCV(ABC, param_grid=param_grid))]
clf = Pipeline(estimators)
clf.fit(features_train, labels_train)
cvFitClassifier("AdaBoost", clf)
'''

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=20, max_depth=None, min_samples_split=1, random_state=30)
cvFitClassifier("Random Forest", clf)


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=45, max_depth=None, min_samples_split=5, random_state=30)
cvFitClassifier("Random Forest2", clf)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.35, random_state=42)


clf = AdaBoostClassifier(n_estimators=25, random_state=None, learning_rate=0.9)
cvFitClassifier("Final AdaBoost", clf)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)