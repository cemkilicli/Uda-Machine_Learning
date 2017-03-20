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
########################### MULTILAYER PERCEPTRON CLASSIFIER (MLPC)  ###################################################
########################################################################################################################

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='logistic',
                    learning_rate='adaptive', learning_rate_init=0.001,
                     hidden_layer_sizes=(2, 2), random_state=10)

clf.fit(X, y)
MLPC_pred = clf.predict(X)
print MLPC_pred
print clf.score(X, y)

# Find the items/cases that are categorized falsely
i = 0,
for i in range(149):
    if (y[i]!=MLPC_pred[i]):
        print "The misclassified item:", i
print "true label of 121th sample is ", y[120]
print "preidcted label of 121th sample is ", MLPC_pred[120]

Zx = [[5, 5, 5, 5],[3, 3, 3, 3]]
Z = np.array(Zx)
print Z
print clf.predict_log_proba(Z)
print clf.predict_proba(Z)
print clf.predict(Z)
print confusion_matrix(MLPC_pred, y)  #
#print fit.score(X, y)  # 96% of accuracy
print accuracy_score(y, MLPC_pred)  # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print accuracy_score(y, MLPC_pred, normalize=False)  # the number of correct predictions