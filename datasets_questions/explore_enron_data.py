#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
import sys
sys.path.append("../tools/")
import feature_format
import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

email_count = 0
for k in enron_data:
    if enron_data[k]['email_address'] != 'NaN':
        email_count += 1
print email_count

salary_count = 0
for person in enron_data:
    if enron_data[person]['salary'] != 'NaN':
        salary_count += 1
print salary_count


missing_count = 0
total = 0.0
for person in enron_data:
    if enron_data[person]['poi']:
        missing_count += 1
print missing_count
print len(enron_data)
#data = feature_format.featureFormat()