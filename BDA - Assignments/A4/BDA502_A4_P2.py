


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
from sklearn import linear_model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Loading some example data
iris = datasets.load_iris()
X = iris.data[:, [0, 1]]  # try different combinations of 0, 1, 2, and 3 that corresponds to different features
print X
y = iris.target

# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=5)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
clf4 = LinearDiscriminantAnalysis()
clf5 = RandomForestClassifier(min_samples_split = 10,min_samples_leaf = 5 )
clf6 = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf7 = GaussianNB()
clf8 = KMeans(n_clusters=3, random_state=0).fit(X)
#eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3), ('lda', clf4),('Random Forest Classifier', clf5), ('MLPC', clf6), ('NB', clf7), ('Kmeans', clf8)], voting='soft', weights=[1, 1, 1, 1, 1, 1, 1, 1])
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
clf4.fit(X, y)
pred4 = clf4.predict(X)
print "LDA"
print confusion_matrix(pred4, y)
print accuracy_score(y, pred4)

#
clf5.fit(X, y)
pred5 = clf5.predict(X)
print "Random Forest Classifier"
print confusion_matrix(pred5, y)
print accuracy_score(y, pred5)


#
clf6.fit(X, y)
pred6 = clf6.predict(X)
print "MLPC"
print confusion_matrix(pred6, y)
print accuracy_score(y, pred6)

#
clf7.fit(X, y)
pred7 = clf7.predict(X)
print "NB"
print confusion_matrix(pred7, y)
print accuracy_score(y, pred7)

#
clf8.fit(X, y)
pred8 = clf8.predict(X)
print "Kmeans"
print confusion_matrix(pred8, y)
print accuracy_score(y, pred8)

#
eclf.fit(X, y)
pred9 = eclf.predict(X)
print "Voting Classifier"
print confusion_matrix(pred9, y)
print accuracy_score(y, pred9)

# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(product([0, 1, 2], [0, 1, 2]),
                        [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, eclf],
                        ['Decision Tree (depth=4)', 'KNN (k=7)',
                         'Kernel SVM', 'LDA', "Random Forest Classifier", "MLPC", "NB", "Kmeans", 'Soft Voting']):
    # print idx # stores the information about the location of the subplot as such: (0,0), (0,1), (1,0), (1,1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    axarr[idx[0], idx[1]].set_title(tt)

print xx.ravel()
plt.show()


########################################################################################################################