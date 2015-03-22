"""
homo	homogeneity score
compl	completeness score
v-meas	V measure
ARI	adjusted Rand index
AMI	adjusted mutual information
silhouette	silhouette coefficient
"""
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score


METRIC_NAMES = ['homogeneity', 'completeness', 'v_measure', 'ARI', 'AMI', 'silhouette']

higgs_estimators = {'k_means_higgs_2': KMeans(n_clusters=2),
                    'k_means_higgs_8': KMeans(n_clusters=8)}

bid_estimators = {'k_means_converters_2': KMeans(n_clusters=2),
                  'k_means_converters_3': KMeans(n_clusters=3),
                  'k_means_converters_8': KMeans(n_clusters=8)}


def bench_k_means(estimator, name, data, sample_size):
    features, weights, labels = data
    estimator.fit(features)
    scores = [estimator.inertia_,
              metrics.homogeneity_score(labels, estimator.labels_),
              metrics.completeness_score(labels, estimator.labels_),
              metrics.v_measure_score(labels, estimator.labels_),
              metrics.adjusted_rand_score(labels, estimator.labels_),
              metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
              metrics.silhouette_score(features, estimator.labels_, metric='euclidean', sample_size=sample_size)]
    df = pd.Series(data=scores, index=METRIC_NAMES)
    df.name = name
    df = pd.DataFrame(df)
    df.index.name = 'metric'
    return df


def evaluate_k_means_generic(estimator, data, metric):
    t0 = time()
    features, _, labels = data
    estimator.fit(features)
    elapsed = time() - t0
    return metric(labels, estimator.labels_), elapsed


def evaluate_k_means(data, estimators):
    records = []
    for name, estimator in estimators.items():
        score, elapsed_time = evaluate_k_means_generic(estimator=estimator,
                                                       data=data,
                                                       metric=metrics.v_measure_score)
        records.append([name, score, elapsed_time])
    df = pd.DataFrame.from_records(records, columns=['estimator', 'v-measure', 'time'])
    return df


def estimate_clusters(data):
    features, weights, labels = data
    scores = []
    estimator = KMeans()
    n_clusters = features.shape[1]
    for n in range(1, n_clusters):
        estimator.n_clusters = n
        score = np.mean(cross_val_score(estimator, features, labels, scoring='adjusted_rand_score'))
        scores.append([n, score])
    df = pd.DataFrame.from_records(scores, columns=['clusters', 'score'])
    df['algo'] = 'kmeans'
    return df

"""

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based",
              data=data)
print(79 * '_')

###############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() + 1, reduced_data[:, 0].max() - 1
y_min, y_max = reduced_data[:, 1].min() + 1, reduced_data[:, 1].max() - 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
"""""