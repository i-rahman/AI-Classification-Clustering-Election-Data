import numpy as np
import math
import random


class K_MEANS:
    def __init__(self, k, t):
        # k_means state here
        # Feel free to add methods
        # t is max number of iterations
        # k is the number of clusters
        self.k = k
        self.t = t

    def distance(self, centroids, datapoint):
        diffs = (centroids - datapoint) ** 2
        return np.sqrt(diffs.sum(axis=1))

    def train(self, X):
        # training logic here
        # input is array of features (no labels)

        # randomly select 3 centroids from the data points
        centroids = X[np.random.choice(X.shape[0], size=3, replace=False), :]

        # Initate variables
        centriods_changed = True
        self.cluster = []
        iterations = 0

        while centriods_changed == True and iterations < self.t:
            self.cluster.clear()
            # find the distance for each data point from all
            # 3 centroids; choose the index of the min distance;
            # set it as the cluster for that specific data point
            for i in range(len(X)):
                distances = self.distance(centroids, X[i])
                self.cluster.append((distances.argmin()))

            # Check to see if the centroids need to be updated
            changed = False
            for k in range(self.k):
                # collect the data points in cluster k
                cluster_k = [
                    X[i] for i in range(len(self.cluster)) if self.cluster[i] == k
                ]
                # re-calcualte the mean of cluster of cluster k
                new_centroid = np.average(cluster_k, axis=0)

                # if mean of cluster k is different from the exisitng centroid
                # update the centroid value and indicate that centroid has changed

                if (centroids[k] != new_centroid).all():
                    centroids[k] = new_centroid
                    changed = True
            # changed = True is as long as one of the centroid changed
            # update the gloabal centriods_changed variable
            centriods_changed = changed
            iterations += 1

        # return cluster
        return self.cluster


class AGNES:
    # Use single link method(distance between cluster a and b = distance between closest
    # members of clusters a and b
    def __init__(self, k):
        # agnes state here
        # Feel free to add methods
        # k is the number of clusters
        self.k = k

    def distance(self, a, b):
        diffs = (a - b) ** 2
        return np.sqrt(diffs.sum())

    def train(self, X):
        # training logic here
        # input is array of features (no labels)
        self.cluster = []
        for i in range(len(X)):
            self.cluster.append(i)
        cluster_count = len(self.cluster)

        # calc the distance between all the datapoints
        dist = []
        for i in range(len(X)):
            for j in range(0, i):
                dist.append((self.distance(X[i], X[j]), i, j))

        # sort the data-points in descending order
        dist.sort(key=lambda x: x[0], reverse=True)

        # run a while loop - check cluster size != self.k
        # and distance arr still has element
        while cluster_count != self.k and dist:

            # pop the smallest distance data-point 
            curr_item = dist.pop()
            # get the index of the data points
            min_index = (curr_item[1], curr_item[2])

            # get their exiting cluster assigments
            new_cluster = self.cluster[min_index[0]]
            old_cluster = self.cluster[min_index[1]]

            # if the data points are in different clusters
            # then merge their clusters in one, along with
            # all other data points and decrease cluster_count
            if old_cluster != new_cluster:
                for i in range(len(self.cluster)):
                    if self.cluster[i] == old_cluster:
                        self.cluster[i] = new_cluster
                cluster_count -= 1

        # return cluster
        return self.cluster