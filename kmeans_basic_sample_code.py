import math
import os
import numpy as np
import matplotlib.pyplot as plt

def get_dist(point_a, point_b):
    x1, y1 = point_a
    x2, y2 = point_b
    return math.hypot(x1-x2, y1-y2)

# Calculate the direct distance between the center point and other points
def calc_distance(obs, guess_central_points):
	"""
	:param obs: All observation points
	:param guess_central_points: Center point
	:return:Distance of each point corresponding to the center point
	"""
	distances = []
	for x, y in obs:
		distance = []
		for xc, yc in guess_central_points:
			distance.append(get_dist((x, y), (xc, yc)))
		distances.append(distance)

	return distances


def k_means2(obs, k, guess_central_points = None, dist=np.median):
	"""

	:param obs: Points to be observed
	:param k: Cluster number k
	:param dist: Characterization clustering center function
	:return: guess_central_points Center point
			current_cluster Classification results
	"""
	obs_num = obs.shape[0]
	if k < 1:
		raise ValueError("Asked for %d clusters." % k)
	# Random center point
	if guess_central_points is None:
		guess_central_points = obs[np.random.choice(obs_num, size=k, replace=False)]  # Initialize maximum distance
	# 모든 데이터의 마지막 cluster
	last_cluster = np.zeros((obs_num, ))

	# When it is less than a certain value, clustering is completed
	while True:
		# The key is the following calc_distance to calculate the required distance
		distances = calc_distance(obs, guess_central_points)
		# Gets the index corresponding to the minimum distance
		current_cluster = np.argmin(distances, axis=1)
		print(current_cluster)
		# If the cluster category has not changed, exit directly
		if (last_cluster == current_cluster).all():
			break

		# Calculate new center
		print('cluster num')
		for i in range(k):
			guess_central_points[i] = dist(obs[current_cluster == i], axis=0)

		last_cluster = current_cluster
		print(guess_central_points)
		for i in range(current_cluster.shape[0]):
			if current_cluster[i] == 0:
				plt.scatter(obs[i,0], obs[i,1], s=5, c='r', cmap='viridis')
			elif current_cluster[i] == 1:
				plt.scatter(obs[i,0], obs[i,1], s=5, c='g', cmap='viridis')
			elif current_cluster[i] == 2:
				plt.scatter(obs[i,0], obs[i,1], s=5, c='b', cmap='viridis')
			else:
				plt.scatter(obs[i,0], obs[i,1], s=5, c='m', cmap='viridis')

		plt.scatter(guess_central_points[:,0], guess_central_points[:,1], s=10, c='#33FFCE')
		plt.xlabel('x', fontsize=12)
		plt.ylabel('y', fontsize=12)
		plt.show()
		plt.cla()


	return guess_central_points, current_cluster

test_sample_x = np.random.randint(0, 100, size=100)
test_sample_y = np.random.randint(0, 100, size=100)
test_sample = [test_sample_x, test_sample_y]
test_sample = np.array(test_sample).transpose()

#guess_central_points = None
#for i in range(30):
#	point, cluster = k_means2(test_sample, 4, guess_central_points)
#	guess_central_points = point

# k_means + + calculation center coordinates
def calc_center(boxes):
    box_number = boxes.shape[0]
    # Select the first center point at random
    first_index = np.random.choice(box_number, size=1)
    clusters = boxes[first_index]
    # Calculate the distance from each sample to the center point
    dist_note = np.zeros(box_number)
    dist_note += np.inf
    for i in range(k):
        # If enough cluster centers have been found, exit
        if i+1 == k:
            break
        # Calculates the distance between the current center point and other points
        for j in range(box_number):
            j_dist = calc_distance(boxes[j], clusters[i])
            if j_dist < dist_note[j]:
                dist_note[j] = j_dist
        # Convert to probability
        dist_p = dist_note / dist_note.sum()
        # Use the roulette method to select the next point
        next_index = np.random.choice(box_number, 1, p=dist_p)
        next_center = boxes[next_index]
        clusters = np.vstack([clusters, next_center])
    return clusters