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
###################################  SUPPORT VECTOR MACHINES (SVM)  ####################################################
########################################################################################################################

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

clf = SVC()
clf.fit(X, y)
predx = clf.predict(X)
print predx
print confusion_matrix(predx, y)  #
print clf.score(X, y)  # 96% of accuracy

print "true label of 121th sample is ", y[120]
print "preidcted label of 121th sample is ", predx[120]

# Find the items/cases that are categorized falsely
i = 0,
for i in range(149):
    if (y[i]!=predx[i]):
        print "The misclassified item:", i

# print X
Zx = [[5, 5, 5, 5],[3, 3, 3, 3]]
Z = np.array(Zx)
print Z
#print clf.predict_log_proba(Z)
#print clf.predict_proba(Z)
print clf.predict(Z)
print clf.decision_function(X)
print clf.get_params()
print confusion_matrix(predx, y)  #
print accuracy_score(y, predx)  # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print accuracy_score(y, predx, normalize=False)  # the number of correct predictions