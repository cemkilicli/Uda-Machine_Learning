
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
############################################# K-MEANS CLUSTERING  ######################################################
########################################################################################################################

from sklearn.cluster import KMeans
from sklearn import metrics

kmeans_model = KMeans(n_clusters=3, random_state=10).fit(X)
labels = kmeans_model.labels_
metrics.silhouette_score(X, labels, metric='euclidean')
A = metrics.silhouette_score(X, labels, metric='euclidean')
print A
# print results
centroids = kmeans_model.cluster_centers_
print centroids
######
points = np.array(centroids)
print points
C1_C2 = np.sqrt(sum((points[0, :]-points[1, :])**2))
print "The distance between the centroids C1 and C2:", C1_C2
# You should calculate the distances between the other centroids

print "************** End of Kmeans **************"

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
print "preidcted label of 121th sample is ", lda.predict(X[120])



i = 0,
for i in range(149):
    if (y[i]!=pred[i]):
        print "The misclassified item:", i

Zx = [[5, 5, 5, 5],[3, 3, 3, 3]]   # This is the item that I have made up with 4 features
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
# .....................
# .....................

# print X
Zx = [[5, 5, 5, 5],[3, 3, 3, 3]]   # This is the item that I have made up with 4 features
Z = np.array(Zx)  # I have changed it as a numpy array
print Z
print fit.predict_log_proba(Z)
print fit.predict_proba(Z)
print fit.predict(Z)

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

# Find the items/cases that are categorized falsely
# .....................
# .....................

# print X
Zx = [[5, 5, 5, 5],[3, 3, 3, 3]]
Z = np.array(Zx)
print Z
# print clf.predict_log_proba(Z)
# print clf.predict_proba(Z)
print clf.predict(Z)
print clf.decision_function(X)
print clf.get_params()
print confusion_matrix(predx, y)  #
print accuracy_score(y, predx)  # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print accuracy_score(y, predx, normalize=False)  # the number of correct predictions

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
# .....................
# .....................

Zx = [[5, 5, 5, 5],[3, 3, 3, 3]]
Z = np.array(Zx)
print Z
print clf.predict_log_proba(Z)
print clf.predict_proba(Z)
print clf.predict(Z)
print confusion_matrix(MLPC_pred, y)  #
print fit.score(X, y)  # 96% of accuracy
print accuracy_score(y, MLPC_pred)  # the use of another function for calculating the accuracy (correct_predictions / all_predictions)
print accuracy_score(y, MLPC_pred, normalize=False)  # the number of correct predictions

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

# Find the items/cases that are categorized falsely
# .....................
# .....................

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


########################################################################################################################
#############################  PRINCIPAL COMPONENT ANALYSIS (PCA)  #####################################################
########################################################################################################################

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit(X).transform(X)
print pca.score(X, y)
print pca.score_samples(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.show()

########################################################################################################################
########################################################################################################################


