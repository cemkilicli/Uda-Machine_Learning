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
C1_C3 = np.sqrt(sum((points[0, :]-points[2, :])**2))
print "The distance between the centroids C1 and C2:", C1_C3
C2_C3 = np.sqrt(sum((points[1, :]-points[2, :])**2))
print "The distance between the centroids C1 and C2:", C2_C3

print "true label of 121th sample is ", y[120]
print "preidcted label of 121th sample is ", labels[120]


# Find the items/cases that are categorized falsely
i = 0,
for i in range(149):
    if (y[i]!=labels[i]):
        print "The misclassified item:", i

