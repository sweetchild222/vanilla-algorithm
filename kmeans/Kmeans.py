from math import sqrt
import numpy as np
from pprint import pprint
import copy

def euclidean(data, center):

	distance = 0.0

	for i in range(len(data)):
		distance += (data[i] - center[i])**2

	return sqrt(distance)


def clustring(data, centroid):

	k = np.size(centroid, 0)

	while True:
		clusters = [[] for c in range(k)]
		for d in data:
			distance_list = []
			for c in centroid:
				dist = euclidean(d, c)
				distance_list.append(dist)

			center_index = min(range(len(distance_list)), key=distance_list.__getitem__)
			clusters[center_index].append(d)

		new_centroid = [np.array(c).mean(axis=0).tolist() for c in clusters]

		if (np.array(new_centroid) == np.array(centroid)).all():
			return clusters

		centroid = copy.deepcopy(new_centroid)


data = [[2, 5], [4, 5],[3, 6],[2, 8],[4, 7],
		[7, 2],[8, 1],[8, 3],[9, 4],[9, 2],
		[10, 3],[10, 1],[8, 10],[8, 8],[9, 9],
		[9, 8],[10, 8],[11, 9]]

centroid = [
	[3, 3],
	[7, 6],
	[11, 6]]

clusters = clustring(data, centroid)

pprint(clusters)
