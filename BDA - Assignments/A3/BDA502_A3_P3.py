


"""
===========================================================
A demo of K-Means clustering on the handwritten digits data
===========================================================
In this example we compare the various initialization strategies for K-means in terms of runtime and quality of the results.

"""

print(__doc__)

import pylab as pl
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from time import time
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import pylab as plt

np.random.seed(42)
digits = load_digits()  # the whole dataset with the labels and other information are extracted
data = scale(digits.data)  # the data is scaled with the use of z-score
n_samples, n_features = data.shape  # the no. of samples and no. of features are determined with the help of shape
n_digits = len(np.unique(digits.target))  # the number of labels are determined with the aid of unique formula
labels = digits.target  # get the ground-truth labels into the labels

print digits
print digits.target_names
print digits.DESCR
print digits.images

print "The content of the first sample in the dataset:\n", digits.images[0]
print "The given label of the first sample in the dataset:", labels[0]

pl.gray()  # this function disables the colormap and if you disable this line, you will see how it changes.
pl.matshow(digits.images[0])
pl.show()

# Below, there is the digit I introduce and named as tunasdigit
tunasdigit = [[16., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 16., 16., 16., 16.], [16., 15., 14., 13., 12., 11., 10., 9.],
              [0., 1., 2., 3., 4., 5., 6., 7.], [0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 16., 16., 16., 16.], [16., 16., 16., 16., 16., 16., 16., 16.]]

# Below, there is the digit I introduce and named as cemsdigit
cemsdigit = [[8., 8., 8., 1., 2., 3., 4., 4.], [0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 0., 0., 0., 4., 4., 4., 4.], [1., 1., 1., 1., 1., 1., 1., 9.],
              [0., 1., 2., 3., 4., 5., 6., 7.], [0., 0., 0., 0., 0., 0., 0., 0.],
              [0., 1., 2., 3., 4., 5., 6., 7.], [11., 11., 11., 11., 11., 11., 11., 11.]]

pl.gray()  # this function disables the colormap and if you disable this line, you will see how it changes.
pl.matshow(tunasdigit)
pl.show()

########################################################################################################################

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))
print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\t\tAMI\t\tsilhouette')

def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=n_samples)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10), name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10), name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=10), name="PCA-based", data=data)

print(82 * '_')

########################################################################################################################
# The dimension reduction stage by PCA and PCA-reduced data is prepared

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)
model = PCA()
results = model.fit(data)
Z = results.transform(data)

########################################################################################################################

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
Z = Z.reshape(xx.shape)
# print Z
# print len(Z)

plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired, aspect='auto', origin='lower')
plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
print centroids

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=169, linewidths=3, color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

########################################################################################################################
