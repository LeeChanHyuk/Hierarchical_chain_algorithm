import math
import os
import numpy as np
import matplotlib.pyplot as plt



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
			distance.append(math.dist((x, y), (xc, yc)))
		distances.append(distance)

	return distances


def k_means(obs, k, guess_central_points = None, dist=np.median):
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
	last_cluster = np.zeros((obs_num, ))

	# When it is less than a certain value, clustering is completed
	while True:
		# The key is the following calc_distance to calculate the required distance
		distances = calc_distance(obs, guess_central_points)
		# Gets the index corresponding to the minimum distance
		current_cluster = np.argmin(distances, axis=1)
		# If the cluster category has not changed, exit directly
		if (last_cluster == current_cluster).all():
			break

		# Calculate new center
		print('cluster num')
		for i in range(k):
			guess_central_points[i] = dist(obs[current_cluster == i], axis=0)
			print(f'cluster_{i} num is {0}'.format(i, len(current_cluster == i)))

		last_cluster = current_cluster

	return guess_central_points, current_cluster

test_sample_x = np.random.randint(0, 100, size=100)
test_sample_y = np.random.randint(0, 100, size=100)
test_sample = [test_sample_x, test_sample_y]
test_sample = np.array(test_sample).transpose()

guess_central_points = None
for i in range(30):
	point, cluster = k_means(test_sample, 4, guess_central_points)
	guess_central_points = point
	plt.scatter(test_sample_x, test_sample_y, s=5, c='#FF5733')
	plt.scatter(point[:,0], point[:,1], s=10, c='#33FFCE')
	plt.show()
