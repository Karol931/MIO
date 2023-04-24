import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_moons

SEED = 33
X, y = make_moons(n_samples=1000, noise=0.1, random_state=SEED)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Moons distribution")
plt.show()

# KMeans
kmeans = KMeans(n_clusters=2, random_state=SEED)
kmeans_labels = kmeans.fit_predict(X)

print("KMeans")
print("Silhouette score: " + str(silhouette_score(X, kmeans_labels)))
print("Calinski-Harabasz score: " + str(calinski_harabasz_score(X, kmeans_labels)))
print("Davies-Bouldin score: " + str(davies_bouldin_score(X, kmeans_labels)) + "\n")

# DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

print("DBSCAN")
print("Silhouette score: " + str(silhouette_score(X, dbscan_labels)))
print("Calinski-Harabasz score: " + str(calinski_harabasz_score(X, dbscan_labels)))
print("Davies-Bouldin score: " + str(davies_bouldin_score(X, dbscan_labels)) + "\n")

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=2)
agg_labels = agg.fit_predict(X)

print("Agglomerative Clustering")
print("Silhouette score: " + str(silhouette_score(X, agg_labels)))
print("Calinski-Harabasz score: " + str(calinski_harabasz_score(X, agg_labels)))
print("Davies-Bouldin score: " + str(davies_bouldin_score(X, agg_labels)) + "\n")

# Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, random_state=SEED)
gmm_labels = gmm.fit_predict(X)

print("Gaussian Mixture Model")
print("Silhouette score: " + str(silhouette_score(X, gmm_labels)))
print("Calinski-Harabasz score: " + str(calinski_harabasz_score(X, gmm_labels)))
print("Davies-Bouldin score: " + str(davies_bouldin_score(X, gmm_labels)) + "\n")