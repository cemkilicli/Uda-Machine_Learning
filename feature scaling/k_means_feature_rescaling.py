#!/usr/bin/python

"""
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it!
data_dict.pop("TOTAL", 0)


### the input features we want to use
### can be any key in the person-level dictionary (salary, director_fees, etc.)
feature_1 = "salary"
feature_2 = "exercised_stock_options"

poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )

from sklearn.preprocessing import MinMaxScaler
salary = []
for users in data_dict:
    val = data_dict[users]["salary"]
    if val =='NaN':
        continue
    salary.append(float(val))

ex_stock = []
for users in data_dict:
    val = data_dict[users]["exercised_stock_options"]
    if val =='NaN':
        continue
    ex_stock.append(float(val))


max_salary = max(salary)
min_salary = min(salary)
max_ex_stock = max(ex_stock)
min_ex_stock = min(ex_stock)

print max_salary,min_salary,max_ex_stock,min_ex_stock

salary_weight = numpy.array([[float(min_salary)],[200000.],[float(max_salary)]])
Scaler = MinMaxScaler()
rescaled_salary_weight = Scaler.fit_transform(salary_weight)
print rescaled_salary_weight

ex_stock_weight = numpy.array([[float(min_ex_stock)],[1000000.],[float(max_ex_stock)]])

rescaled_ex_stock_weight = Scaler.fit_transform(ex_stock_weight)
print rescaled_ex_stock_weight


