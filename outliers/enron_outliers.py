#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop( "TOTAL", 0 ) 
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below



for point in data:
    salary = point[0]
    bonus = point[1]
    
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

for k in data_dict:
    if "NaN" != data_dict[k]["salary"] and data_dict[k]["salary"] > 1e6:
        print k  
    elif  data_dict[k]["bonus"] != "NaN" != data_dict[k]["salary"] and data_dict[k]["bonus"] > 0.7*1e7:
        print k
        