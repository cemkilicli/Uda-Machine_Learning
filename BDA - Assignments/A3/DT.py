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
########################################### DECISION TREE (DT) #########################################################
########################################################################################################################

from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
DT_pred = clf.predict(X)

print clf.predict(iris.data[:1, :])
print clf.predict_proba(iris.data[:1, :])
print DT_pred
print clf.score(X, y)
# print clf.decision_path(X)


print "true label of 121th sample is ", y[120]
print "preidcted label of 121th sample is ", DT_pred[120]

# Find the items/cases that are categorized falsely
i = 0,
for i in range(149):
    if (y[i]!=DT_pred[i]):
        print "The misclassified item:", i

Zx = [[5, 5, 5, 5],[3, 3, 3, 3]]
Z = np.array(Zx)
print Z
print clf.predict_log_proba(Z)
print clf.predict_proba(Z)
print clf.predict(Z)
print clf.decision_path(Z)
print confusion_matrix(y, DT_pred)  #
#print fit.score(X, y)  # 96% of accuracy
print accuracy_score(y, DT_pred)  # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print accuracy_score(y, DT_pred, normalize=False)  # the number of correct predictions