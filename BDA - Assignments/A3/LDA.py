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
#############################  LINEAR DISCRIMINANT ANALYSIS (LDA)  #####################################################
########################################################################################################################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit(X, y).transform(X)  # fit: Fit the LDA model according to the given training data and parameters.
# transform: Project the data so as to maximize class separation (large separation between projected class means and small variance within each class).
pred = lda.predict(X)  # This function does classification on an array of test vectors X.
print y
print pred
print lda.score(X, y)  # Returns the mean accuracy on the given test data and labels.

print "true label of 121th sample is ", y[120]
print "preidcted label of 121th sample is ",lda.predict(X[120])


# Find the items/cases that are categorized falsely
i = 0,
for i in range(149):
    if (y[i]!=pred[i]):
        print "The misclassified item:", i

Zx = [[5, 5, 5, 5],[3, 3, 3, 3]]  # This is the item that I have made up with 4 features
Z = np.array(Zx)  # I have changed it as a numpy array
print Z
print lda.predict_log_proba(Z)  # This function returns posterior log-probabilities of classification according to each class on an array of test vectors X.
print lda.predict_proba(Z)  # This function returns posterior probabilities of classification according to each class on an array of test vectors X.
print lda.predict(Z)  # This function does classification on an array of test vectors X.
print lda.decision_function(Z)  # This function returns the decision function values related to each class on an array of test vectors X.

print confusion_matrix(pred, y)  #
# print fit.score(X, y)  # 96% of accuracy
print accuracy_score(y, pred)  # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
# print accuracy_score(y, pred, normalize=False)  # the number of correct predictions

colors = ['navy', 'turquoise', 'darkorange']
lw = 2
plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA of IRIS dataset')
plt.show()
