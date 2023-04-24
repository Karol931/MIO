import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("planets.csv", skiprows=1)

scaler = StandardScaler()
data = df.values[:,1:]
data = scaler.fit_transform(data)

kmeans_model = KMeans(n_clusters=3)
kemans_clusters = kmeans_model.fit_predict(data)
print(kmeans_model.labels_)
print(kmeans_model.cluster_centers_)

agg_model = AgglomerativeClustering(n_clusters=3)
agg_clusters = agg_model.fit_predict(data)

dbscan_model = DBSCAN(eps=0.3, min_samples=5)
dbscan_clusters = dbscan_model.fit_predict(data)

print('Sihouette Score')
print(f"KMeans: {metrics.silhouette_score(data, kemans_clusters)}")
print(f"AgglomerativeClustering: {metrics.silhouette_score(data, agg_clusters)}")
print(f"DBSCAN: {metrics.silhouette_score(data, dbscan_clusters)}")
print()
print('DB Score')
print(f"KMeans: {metrics.davies_bouldin_score(data, kemans_clusters)}")
print(f"AgglomerativeClustering: {metrics.davies_bouldin_score(data, agg_clusters)}")
print(f"DBSCAN: {metrics.davies_bouldin_score(data, dbscan_clusters)}")

linkage_type = ['ward', 'complete', 'average', 'single']
scorers = [metrics.silhouette_score, metrics.davies_bouldin_score]
for linkage in linkage_type:
    for scorer in scorers:
        score = []
        for i in range(2,8):
            agg_cluster = AgglomerativeClustering(n_clusters=i,linkage=linkage)
            agg_cluster.fit(data)
            labels = agg_cluster.labels_
            score.append(scorer(data,labels))

        plt.plot(range(2,8), score)
        plt.title(str(scorer.__name__) + " " + linkage)
        plt.xlabel("Number of Clusters")
        plt.ylabel("Score")
        plt.figure()
plt.show()
