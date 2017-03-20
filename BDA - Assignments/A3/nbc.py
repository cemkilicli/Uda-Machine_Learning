print(__doc__)
import numpy as np

########################################################################################################################
########################################  PREPARING THE IRIS DATA  #####################################################
########################################################################################################################

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names



########################################################################################################################
###################################  NAIVE BAYES CLASSIFIER (NBC)  #####################################################
########################################################################################################################

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

gnb = GaussianNB()
fit = gnb.fit(X, y)
print fit
pred = fit.predict(X)
print y
print pred
print confusion_matrix(pred, y)  #
print fit.score(X, y)  # 96% of accuracy
print accuracy_score(y, pred)  # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print accuracy_score(y, pred, normalize=False)  # the number of correct predictions

print "true label of 121th sample is ", y[120]
print "preidcted label of 121th sample is ", pred[120]

# Find the items/cases that are categorized falsely
i = 0,
for i in range(149):
    if (y[i]!=pred[i]):
        print "The misclassified item:", i
# print X
Zx = [[5, 5, 5, 5],[3, 3, 3, 3]]  # This is the item that I have made up with 4 features
Z = np.array(Zx)  # I have changed it as a numpy array
print Z
print fit.predict_log_proba(Z)
print fit.predict_proba(Z)
print fit.predict(Z)