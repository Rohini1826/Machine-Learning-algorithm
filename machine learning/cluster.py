import numpy as np
from sklearn.cluster import KMeans

X=np.array([
    [25,40],
    [27,42],
    [29,45],
    [45,20],
    [47,22],
    [49,18]
])

customers=["A" , "B" , "C" , "D" , "E" , "F"]
init_centroids=np.array([
    [25,40],
    [45,20]
])

kmeans=KMeans(
    n_clusters=2,
    init=init_centroids,
    n_init=1,
    random_state=0
)
kmeans.fit(X)

labels=kmeans.labels_
centroids=kmeans.cluster_centers_
print("Centroids:\n", centroids)
print("Labels:",labels)

if centroids[0] [1] > centroids[1] [1]:
    high_cluster=0
    low_cluster=1
else:
    high_cluster=1
    low_cluster=0

print("\nPredictions:")
for i,name in enumerate (customers):
    if labels[i]==high_cluster:
        print(f"{name} -> High Spender")
    else:
        print(f"{name} -> Low Spender")
