# Example of making predictions
from math import sqrt



def euclidean_distance(user, neighbor):
	distance = 0.0
	for i in range(len(user)-1):
		distance += (user[i] - neighbor[i])**2
	return sqrt(distance)



def get_neighbors(user, neighbor_list, k):
	distances = list()
	for neighbor in neighbor_list:
		dist = euclidean_distance(user, neighbor)
		distances.append((neighbor, dist))
	distances.sort(key=lambda tup: tup[1])

	print('neighbors distances : ', distances)

	near_neighbors = list()
	for i in range(k):
		near_neighbors.append(distances[i][0])

	print('near neighbors : ', near_neighbors)

	return near_neighbors


def predict_classification(user, neighbor_list, k):
	neighbors = get_neighbors(user, neighbor_list, k)

	predict_candidate = [row[-1] for row in neighbors]
	print('predict_candidate : ', predict_candidate)
	prediction = max(set(predict_candidate), key=predict_candidate.count)
	return prediction

# Test distance function




k = 3
user = [9, 1, 0]
neighbor_list = [
	[2,8,1],
	[7,2,9],
	[8,1,7],
	[1,9,1],
	[9,2,9],
	[1,8,2]]

prediction = predict_classification(user, neighbor_list, k)

print('Predict %f.' % (prediction))
