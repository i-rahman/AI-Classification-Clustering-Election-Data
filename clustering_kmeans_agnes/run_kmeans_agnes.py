import kmeans_agnes as models
import numpy as np
import sys
from sklearn import metrics
import time

if(sys.version_info[0] < 3):
    raise Exception("This assignment must be completed using Python 3")

def load_data(path):
    data = np.genfromtxt(path, delimiter=',', dtype=float)
    return data[:,:-1], data[:,-1].astype(int)

def count_clusters(clusters):
	my_set = {}
	for i in range(len(clusters)):
		if(clusters[i] in my_set):
			my_set[clusters[i]] += 1
		else:
			my_set[clusters[i]] = 1
	print(my_set)

X, y = load_data("county_statistics.csv")

#Initialization
# k_means
k = 3
t=50 #max iterations
k_means = models.K_MEANS(k, t)

#AGNES
k = 3
agnes = models.AGNES(k)

#Train
print("K_MEANS")
start = time.perf_counter()
cluster_kmeans = k_means.train(X)
print(time.perf_counter() - start, "seconds spent on training")
print("Cluster has a Silhouette Score of ", metrics.silhouette_score(X, cluster_kmeans, metric='euclidean'))

print("\nAGNES")
start = time.perf_counter()
cluster_agnes = agnes.train(X)
print(time.perf_counter() - start, "seconds spent on training")
print("Cluster has a Silhouette Score of ", metrics.silhouette_score(X, cluster_agnes, metric='euclidean'))
