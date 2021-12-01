import matplotlib.pyplot as plt
import cv2
import os
import json
import numpy as np
from kmeans_basic_sample_code import *

face_json_path = './00033/appleFace.json'
left_eye_json_path = './00033/appleLeftEye.json'
right_eye_json_path = './00033/appleRightEye.json'
image_folder_path = './00033/frames'       # Path to training set image
# Number of pictures displayed
show_num = 99
#Open xml document
def read_voc(train_annotation_path, train_image_path, show_num):
	with open(face_json_path) as json_face_file:
		with open(left_eye_json_path) as left_eye_json_file:
			with open(right_eye_json_path) as right_eye_json_file:
				json_face_data = json.load(json_face_file)
				face_height = json_face_data['H']
				face_width = json_face_data['W']
				face_x_coordinate = json_face_data['X']
				face_y_coordinate = json_face_data['Y']
				face_valid = json_face_data['IsValid']
				json_left_eye_data = json.load(left_eye_json_file)
				left_eye_height = json_left_eye_data['H']
				left_eye_width = json_left_eye_data['W']
				left_eye_x_coordinate = json_left_eye_data['X']
				left_eye_y_coordinate = json_left_eye_data['Y']
				left_eye_valid = json_left_eye_data['IsValid']
				json_right_eye_data = json.load(right_eye_json_file)
				right_eye_height = json_right_eye_data['H']
				right_eye_width = json_right_eye_data['W']
				right_eye_x_coordinate = json_right_eye_data['X']
				right_eye_y_coordinate = json_right_eye_data['Y']
				right_eye_valid = json_right_eye_data['IsValid']
				img_wh = []
				bbox_wh = []
				for index, file in enumerate(os.listdir(image_folder_path)):
					if face_valid[index] == 0 or left_eye_valid[index] == 0 or right_eye_valid[index] == 0:
						continue
					# Draw face bounding box
					img = cv2.imread(os.path.join(image_folder_path, file))
					img_wh.append([img.shape[2], img.shape[1]])
					x = int(face_x_coordinate[index])
					y = int(face_y_coordinate[index])
					w = int(face_width[index])
					h = int(face_height[index])
					bbox_wh.append([x/img.shape[2], y/img.shape[1]])
					cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (255, 0, 0), 3)
					# Draw left eye bounding box
					left_x = int(left_eye_x_coordinate[index]) + x
					left_y = int(left_eye_y_coordinate[index]) + y
					left_w = int(left_eye_width[index])
					left_h = int(left_eye_height[index])
					cv2.rectangle(img, (left_x, left_y), (left_x + left_w - 1, left_y + left_h - 1), (0, 255, 0), 3)
					# Draw right eye bounding box
					right_x = int(right_eye_x_coordinate[index]) + x
					right_y = int(right_eye_y_coordinate[index]) + y
					right_w = int(right_eye_width[index])
					right_h = int(right_eye_height[index])
					cv2.rectangle(img, (right_x, right_y), (right_x + right_w - 1, right_y + right_h - 1), (0, 0, 255), 3)
					# Show image
					#cv2.imshow('img',img)
					#cv2.waitKey(100)
	return img_wh, bbox_wh



# 모든 image의 width, height를 저장해 둔 리스트가 img_wh
# 모든 image내 모든 bounding box의 width, height를 저장해 둔 리스트가 bbox_wh
imgs_wh, bboxs_wh = read_voc(face_json_path, image_folder_path, show_num)

def wh_iou(wh1, wh2):
	wh1 = wh1[:, None] # [N, 1, 2]
	wh2 = wh2[None]
	inter_minimum = np.minimum(wh1, wh2)
	inter = np.expand_dims(inter_minimum, axis=2).prod(2)
	return inter / (wh1.prod(2) + wh2.prod(2) - inter)

# 기존의 k-means와 같은데, IoU 파트만 다름.
# k-means clustering, and the evaluation index adopts IOU
def k_means(boxes, k, dist=np.median, use_iou=True, use_pp=False):
    """
    yolo k-means methods
    Args:
        boxes: Need clustering bboxes,bboxes by n*2 contain w，h
        k: Number of clusters(Gather into several categories)
        dist: Method of updating cluster coordinates(The median is used by default, which is slightly better than the mean)
        use_iou: Whether to use IOU As a calculation
    """
    box_number = boxes.shape[0]
    last_nearest = np.zeros((box_number,))
    # k of all bboxes are randomly selected as the center of the cluster
    if not use_pp:
        clusters = boxes[np.random.choice(box_number, k, replace=False)]
    # k_means + + calculates the initial value
    else:
        clusters = calc_center(boxes, k)
    
    # print(clusters)
    while True:
    	# Calculate the distance 1-IOU(bboxes, anchors) from each bboxes to each cluster
        if use_iou:
            distances = 1 - wh_iou(boxes, clusters)
        else:
            distances = calc_distance(boxes, clusters)
        # Calculate the nearest cluster center of each bboxes
        current_nearest = np.argmin(distances, axis=1)
        # If the elements in each cluster are not changing, it indicates that the clustering is completed
        if (last_nearest == current_nearest).all():
            break  # clusters won't change
        for cluster in range(k):
            # Recalculate the cluster center according to the bboxes in each cluster
            clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)

        last_nearest = current_nearest

    return clusters


from tqdm import tqdm
import random
# Calculate the coincidence degree between anchor and real bbox from clustering and genetic algorithm
def anchor_fitness(k: np.ndarray, wh: np.ndarray, thr: float):  # mutation fitness
    """
    Input: k: The results after clustering are in ascending order
         wh: contain bbox in w，h And converted to absolute coordinates
         thr: bbox neutralization k Frame coincidence threshold of clustering
    """
    r = wh[:, None] / k[None]
    x = np.minimum(r, 1. / r).min(2)  # ratio metric
    best = x.max(1)
    f = (best * (best > thr).astype(np.float32)).mean()  # fitness
    bpr = (best > thr).astype(np.float32).mean()  # best possible recall
    return f, bpr


def auto_anchor(img_size, n, thr, gen, img_wh, bbox_wh):
    """
    Input: img_size: Zoom size of picture
          n: Cluster number
          thr: fitness Threshold of
          gen: Iteration times of genetic algorithm
          img_wh: Length and width collection of pictures
          bbox_wh: bbox Long box collection
    """
    # Maximum edge reduction to img_size
    img_wh = np.array(img_wh, dtype=np.float32)
    shapes = (img_size * img_wh / img_wh).max(1, keepdims=True)
    wh0 = np.concatenate([l * s for s, l in zip(shapes, bbox_wh)])  # wh
	

    i = np.expand_dims((wh0 < 3.0), axis=1).any(1).sum()
    if i:
        print(f'WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
    wh = wh0[np.expand_dims((wh0 >= 2.0), axis=1).any(1)]  # Only box es with wh greater than or equal to 2 pixels are reserved
    # k_means cluster computing anchor
    k = k_means(wh, n, use_iou=True, use_pp=False)
    k = k[np.argsort(k.prod(1))]  # sort small to large
    f, bpr = anchor_fitness(k, wh, thr)
    print("kmeans: " + " ".join([f"[{int(i[0])}, {int(i[1])}]" for i in k]))
    print(f"fitness: {f:.5f}, best possible recall: {bpr:.5f}")
        
    # YOLOV5 improved genetic algorithm
    npr = np.random
    f, sh, mp, s = anchor_fitness(k, wh, thr)[0], k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc=f'Evolving anchors with Genetic Algorithm:')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg, bpr = anchor_fitness(kg, wh, thr)
        if fg > f:
            f, k = fg, kg.copy()
        pbar.desc = f'Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'

    # Sort by area
    k = k[np.argsort(k.prod(1))]  # sort small to large
    print("genetic: " + " ".join([f"[{int(i[0])}, {int(i[1])}]" for i in k]))
    print(f"fitness: {f:.5f}, best possible recall: {bpr:.5f}")

auto_anchor(img_size=416, n=9, thr=0.25, gen=1000, img_wh=imgs_wh, bbox_wh=bboxs_wh)