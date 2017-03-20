
"""
========================================================================================================================
======================= Classification applications on the handwritten digits data =====================================
========================================================================================================================
In this example, you will see two different applications of Naive Bayesian Algorithm on the
digits dataset.
"""

print(__doc__)

import pylab as pl
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from time import time
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import pylab as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier





########################################################################################################################
##################################### GETTING THE DATA & PREPARATIONS ##################################################
########################################################################################################################

np.random.seed(42)
digits = load_digits()  # the whole dataset with the labels and other information are extracted
data = scale(digits.data)  # the data is scaled with the use of z-score
n_samples, n_features = data.shape  # the no. of samples and no. of features are determined with the help of shape
n_digits = len(np.unique(digits.target))  # the number of labels are determined with the aid of unique formula
labels = digits.target  # get the ground-truth labels into the labels

#print digits.keys()  # this command will provide you the key elements in this dataset
#print digits.DESCR  # to get the descriptive information about this dataset

########################################################################################################################
########################################################################################################################

from sklearn.model_selection import train_test_split  # some documents still include the cross-validation option but it no more exists in version 18.0
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pylab as plt

y = digits.target
X = scale(digits.data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

cemsdigit1 = [[11., 1., 1., 1., 1., 1., 11., 11.], [11., 1., 11., 11., 11., 1., 11., 11.],
              [11., 1., 11., 11., 11., 1., 11., 11.], [11., 1., 1., 1., 1., 1., 11., 11.],
              [11., 11., 11., 11., 11., 1., 11., 11.], [11., 11., 11., 11., 11., 1., 11., 11.],
              [11., 1., 1., 1., 1., 1., 11., 11.], [11., 11., 11., 11., 11., 11., 11., 11.]]

# Below, there is the digit I introduce and named as cemsdigit
cemsdigit = [11., 1., 1., 1., 1., 1., 11., 11., 11., 1., 11., 11., 11., 1., 11., 11.,
              11., 1., 11., 11., 11., 1., 11., 11., 11., 1., 1., 1., 1., 1., 11., 11.,
              11., 11., 11., 11., 11., 1., 11., 11., 11., 11., 11., 11., 11., 1., 11., 11.,
              11., 1., 1., 1., 1., 1., 11., 11., 11., 11., 11., 11., 11., 11., 11., 11.]

'''pl.gray()  # this function disables the colormap and if you disable this line, you will see how it changes.
pl.matshow(cemsdigit1)
pl.show()'''

########################################################################################################################
#########################################   Test Set NB   ##############################################################
########################################################################################################################

print "Test Set NB"
gnb = GaussianNB()
fit = gnb.fit(X_train, y_train)
predicted = fit.predict(X_test)
print confusion_matrix(y_test, predicted)
print "Accuracy is", accuracy_score(y_test, predicted)  # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print "The number of correct predictions is", accuracy_score(y_test, predicted, normalize=False)  # the number of correct predictions
print "Total sample used is", len(predicted)  # number of all of the predictions



########################################################################################################################
#########################################   Full Set NB   ##############################################################
########################################################################################################################

print "Full Set NB"
gnb = GaussianNB()
fit2 = gnb.fit(X, y)
predictedx = fit2.predict(X)
print confusion_matrix(y, predictedx)
print "Accuracy is", accuracy_score(y, predictedx)  # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print "The number of correct predictions is", accuracy_score(y, predictedx, normalize=False)  # the number of correct predictions
print "Total sample used is", len(predictedx)  # number of all of the predictions


########################################################################################################################
#########################################   Test Set DTC   #############################################################
########################################################################################################################

print "Test Set DTC"
clf_dec_train = DecisionTreeClassifier()
fit3 = clf_dec_train.fit(X_train, y_train)
predict_decision_train = fit3.predict(X_test)
print confusion_matrix(y_test, predicted)
print "Accuracy is",accuracy_score(y_test, predicted)  # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print "The number of correct predictions is", accuracy_score(y_test, predicted, normalize=False)  # the number of correct predictions
print "Total sample used is", len(predict_decision_train)  # number of all of the predictions



########################################################################################################################
#########################################   Full Set DTC   #############################################################
########################################################################################################################

print "Full Set DTC"
clf_dec_full = DecisionTreeClassifier()
fit4 = clf_dec_full.fit(X,y)
predict_decision_full = fit4.predict(X)
print confusion_matrix(y, predict_decision_full)
print "Accuracy is",accuracy_score(y, predict_decision_full)
print "The number of correct predictions is",accuracy_score(y, predict_decision_full, normalize=False)
print "Total sample used is", len(predict_decision_full)


########################################################################################################################
#########################################   Test Set KNN   #############################################################
########################################################################################################################

print "Test Set KNN"
clf_knn_train = KNeighborsClassifier(n_neighbors=7)
fit5 = clf_knn_train.fit(X_train, y_train)
predict_knn_train = fit5.predict(X_test)
print confusion_matrix(y_test, predict_knn_train)
print "Accuracy is",accuracy_score(y_test, predict_knn_train)
print "The number of correct predictions is",accuracy_score(y_test, predict_knn_train, normalize=False)
print "Total sample used is", len(predict_knn_train)



########################################################################################################################
#########################################   Full Set KNN   #############################################################
########################################################################################################################

print "Full Set KNN"
clf_knn_full = KNeighborsClassifier(n_neighbors=7)
fit5 = clf_knn_full.fit(X, y)
predict_knn_full = fit5.predict(X)
print confusion_matrix(y, predict_knn_full)
print "Accuracy is",accuracy_score(y, predict_knn_full)
print "The number of correct predictions is",accuracy_score(y, predict_knn_full, normalize=False)
print "Total sample used is", len(predict_knn_full)



########################################################################################################################
########################################################################################################################

mis_pred_nb = []
mis_pred_dec = []
mis_pred_knn = []

i = 0,
for i in range(1797):
    if (y[i]!=predictedx[i]):
        mis_pred_nb.append(predictedx[i])

i = 0,
for i in range(1797):
    if (y[i]!=predict_decision_full[i]):
        mis_pred_nb.append(predict_decision_full[i])

i = 0,
for i in range(1797):
    if (y[i]!=predict_knn_full[i]):
        mis_pred_nb.append(predict_knn_full[i])

mis_pred = mis_pred_nb + mis_pred_dec + mis_pred_knn

from collections import Counter
count = Counter(mis_pred)
print count.most_common(10)


########################################################################################################################
#########################################   Predict Cem's Digits    ####################################################
########################################################################################################################

predicted_cem = fit.predict(cemsdigit)
print "Test Set NB", predicted_cem
predictedx_cem = fit2.predict(cemsdigit)
print "Full Set NB",predictedx_cem
predict_decision_train_cem = fit3.predict(cemsdigit)
print "Test Set DTC",predict_decision_train_cem
predict_decision_full_cem = fit4.predict(cemsdigit)
print "Full Set DTC",predict_decision_full_cem
predict_knn_train_cem = fit5.predict(cemsdigit)
print "Test Set KNN",predict_knn_train_cem
predict_knn_full_cem = fit5.predict(cemsdigit)
print "Full Set KNN",predict_knn_full_cem
