import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


data = np.loadtxt("press_readers_chicago.csv",delimiter=";",dtype=float)


normal_data = np.zeros((len(data),2))
normal_data[:,0] = (data[:,0] - min(data[:,0]))/(max(data[:,0]) -min(data[:,0]))
normal_data[:,1] = (data[:,1] - min(data[:,1]))/(max(data[:,1]) -min(data[:,1]))


print(normal_data)
print(data)

model = KMeans(n_clusters=4)
model.fit(data)

model_normal = KMeans(n_clusters=4)
model_normal.fit(normal_data)

# print(model.labels_)
# print(model_normal.labels_)

# kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
# kde.fit(x[:, None])

df = pd.DataFrame({
    'x': data[:,0],
    'y': data[:,1],
})

normal_df = pd.DataFrame({
    'x': normal_data[:,0],
    'y': normal_data[:,1],
})

sns.kdeplot(data = df, x = "x", y = "y", shade = True, fill=True)
sns.scatterplot(x=model.cluster_centers_[:,0],y=model.cluster_centers_[:,1],color="red")

plt.figure()
sns.kdeplot(data= normal_df, x = "x", y = "y", shade=True)
sns.scatterplot(x=model_normal.cluster_centers_[:,0],y=model_normal.cluster_centers_[:,1],color="red")
plt.show()