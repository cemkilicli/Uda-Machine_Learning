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

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

### How many data points in the data set
dataPoints = 0
for i in enron_data:
    dataPoints = dataPoints +1

print "there are", dataPoints, "data points in the dataset"


### How many features in the data set
no_of_features = len(enron_data[enron_data.keys()[0]])

print "there are", no_of_features, "features in the data set"

### How many POI in the data set



