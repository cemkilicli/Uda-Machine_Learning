


print(__doc__)

from sklearn import metrics
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Loading some example data
iris = datasets.load_iris()
# try different combinations of 0, 1, 2, and 3 that corresponds to different features
#X = iris.data[:, [0, 1]] # Accuracy 0.83
X = iris.data[:, [0, 2]] # Accuracy 0.98
#X = iris.data[:, [0, 3]] # Accuracy 0.96
#X = iris.data[:, [1, 2]] # Accuracy 0.966
#X = iris.data[:, [1, 3]] # Accuracy 0.966
#X = iris.data[:, [2, 3]] # Accuracy 0.973

print X
y = iris.target

# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),('svc', clf3)], voting='soft', weights=[2, 1, 2])
# eclf = MLPClassifier(solver='lbfgs', alpha=1e-5, activation='logistic', learning_rate='adaptive', learning_rate_init=0.001, hidden_layer_sizes=(4, 4), random_state=10)

#
clf1.fit(X, y)
pred1 = clf1.predict(X)
print "Decision Tree Classifier"
print confusion_matrix(pred1, y)
print accuracy_score(y, pred1)

#
clf2.fit(X, y)
pred2 = clf2.predict(X)
print "KNN"
print confusion_matrix(pred2, y)
print accuracy_score(y, pred2)

#
clf3.fit(X, y)
pred3 = clf3.predict(X)
print "SVC"
print confusion_matrix(pred3, y)
print accuracy_score(y, pred3)

#
eclf.fit(X, y)
pred4 = eclf.predict(X)
print "Voting Classifier"
print confusion_matrix(pred4, y)
print accuracy_score(y, pred4)

# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        [clf1, clf2, clf3, eclf],
                        ['Decision Tree (depth=4)', 'KNN (k=7)',
                         'Kernel SVM', 'Soft Voting']):
    # print idx # stores the information about the location of the subplot as such: (0,0), (0,1), (1,0), (1,1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    axarr[idx[0], idx[1]].set_title(tt)

print xx.ravel()
plt.show()


########################################################################################################################



