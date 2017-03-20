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
#############################  PRINCIPAL COMPONENT ANALYSIS (PCA)  #####################################################
########################################################################################################################

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_pca = pca.fit(X).transform(X)
print pca.score(X, y)
print pca.score_samples(X)

print "true label of 121th sample is ", y[120]
print "preidcted label of 121th sample is ", X_pca[120]

i = 0,
for i in range(149):
    if (y[i]!=X_pca[i]):
        print "The misclassified item:", i


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