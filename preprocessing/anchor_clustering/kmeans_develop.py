import matplotlib.pyplot as plt
import cv2
import os
import json
import numpy as np
from kmeans_basic_sample_code import *
from tqdm import tqdm

# Number of pictures displayed
show_num = 99
#Open xml document
image_base_path = 'E:/Human_information_data/GazeCapture'


def read_voc(image_folder_path, face_annotation_path, left_eye_annotation_path, right_eye_annotation_path, img_wh,
             bbox_wh, bbox_dict, mode='face'):
    with open(face_annotation_path) as json_face_file:
        with open(left_eye_annotation_path) as left_eye_json_file:
            with open(right_eye_annotation_path) as right_eye_json_file:
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
                for index, file in enumerate(os.listdir(image_folder_path)):
                    if face_valid[index] == 0 or left_eye_valid[index] == 0 or right_eye_valid[index] == 0:
                        continue

                    img = cv2.imread(os.path.join(image_folder_path, file))

                    x = int(face_x_coordinate[index])
                    y = int(face_y_coordinate[index])
                    w = int(face_width[index])
                    h = int(face_height[index])

                    left_x = int(left_eye_x_coordinate[index]) + x
                    left_y = int(left_eye_y_coordinate[index]) + y
                    left_w = int(left_eye_width[index])
                    left_h = int(left_eye_height[index])

                    right_x = int(right_eye_x_coordinate[index]) + x
                    right_y = int(right_eye_y_coordinate[index]) + y
                    right_w = int(right_eye_width[index])
                    right_h = int(right_eye_height[index])
                    if mode == 'face':
                        if bbox_dict.get(w) != h:
                            bbox_dict[w] = h
                            bbox_wh.append([w/img.shape[1], h/img.shape[0]])
                            img_wh.append([img.shape[1], img.shape[0]])
                    elif mode == 'eye':
                        if bbox_dict.get(left_w) == left_h :
                            bbox_dict[left_w] = left_h
                            bbox_dict[right_w] = right_h
                            bbox_wh.append([left_w/img.shape[1], left_h/img.shape[0]])
                            bbox_wh.append([right_w/img.shape[1], right_h/img.shape[0]])
                            img_wh.append([img.shape[1], img.shape[0]])

    return img_wh, bbox_wh


imgs_wh, bboxs_wh, bboxs_dict= [], [], {}
for tar_folder in tqdm(os.listdir(image_base_path)):
    for num_folder in os.listdir(os.path.join(image_base_path, tar_folder)):
        image_folder_path = os.path.join(image_base_path, tar_folder,num_folder, 'frames')
        face_annotation_path = os.path.join(image_base_path,tar_folder, num_folder, 'appleFace.json')
        left_eye_annotation_path = os.path.join(image_base_path,tar_folder, num_folder, 'appleLeftEye.json')
        right_eye_annotation_path = os.path.join(image_base_path,tar_folder, num_folder, 'appleRightEye.json')
        imgs_wh, bboxs_wh = read_voc(image_folder_path, face_annotation_path, left_eye_annotation_path, right_eye_annotation_path, imgs_wh, bboxs_wh, bboxs_dict, 'face')
imgs_wh, bboxs_wh = np.array(imgs_wh), np.array(bboxs_wh)
def wh_iou(wh1, wh2):
	wh1 = wh1[:, None] # [N, 1, 2] # Bounding box
	wh2 = wh2[None] # Cluster [1, Cluster]
    # ???, ???????????? cluster??? ???????????? ??? ????????? ????????? cluster??? ???????????????, ?????? bounding box??? cluster ?????? ????????? intersection?????? ????????????
    # 
	inter_minimum = np.minimum(wh1, wh2) # ?????? ?????? (1513,1)??? (1, 9)??? ???????????????, ?????? ?????? ?????? ???????????? ???????????? ??? ??????????????? ????????????
	inter = inter_minimum.prod(2)
	return inter / (wh1.prod(2) + wh2.prod(2) - inter)

def find_cluster_index(cluster, bboxes):
    for i in range(len(cluster)):
        width = cluster[i, 0]
        height = cluster[i, 1]
        for j in range(len(bboxes)):
            bbox_width = bboxes[j, 0]
            bbox_height = bboxes[j, 1]
            if bbox_width == width and bbox_height == height:
                print('{}th cluster is matched with {}th bbox'.format(i, j))
                break

# ????????? k-means??? ?????????, IoU ????????? ??????.
# k-means clustering, and the evaluation index adopts IOU
def k_means(boxes, k, dist=np.median, use_iou=True, use_pp=False):
    """
    yolo k-means methods
    Args:
        boxes: Need clustering bboxes,bboxes by n*2 contain w???h
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
    find_cluster_index(clusters, boxes)
    
    # print(clusters)
    while True:
    	# Calculate the distance 1-IOU(bboxes, anchors) from each bboxes to each cluster
        if use_iou:
            distances = 1 - wh_iou(boxes, clusters)
        else:
            distances = calc_distance(boxes, clusters)
        distance_array = np.array(distances)
        # Calculate the nearest cluster center of each bboxes
        current_nearest = np.argmin(distances, axis=1)
        count = 0
        stop = False
        while 1:
            for i in range(len(current_nearest)):
                if current_nearest[i] == count:
                    print('{0}th cluster is {1}'.format(i, current_nearest[i]))
                    count += 1
            if count == 9:
                stop = True
            if stop:
                break
        # If the elements in each cluster are not changing, it indicates that the clustering is completed
        if (last_nearest == current_nearest).all():
            break  # clusters won't change
        for cluster in range(k):
            # Recalculate the cluster center according to the bboxes in each cluster
            clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)

        last_nearest = current_nearest

    return clusters


import random
# Calculate the coincidence degree between anchor and real bbox from clustering and genetic algorithm
def anchor_fitness(k: np.ndarray, wh: np.ndarray, thr: float):  # mutation fitness
    """
    Input: k: The results after clustering are in ascending order
         wh: contain bbox in w???h And converted to absolute coordinates
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
    shapes = (img_size * img_wh / img_wh).max(1, keepdims=True)
    resized_bboxes = bbox_wh * shapes
    wh0 = np.concatenate([l * s for s, l in zip(shapes, bbox_wh)])  # wh??? ?????? ???????????? bounding box??? ???????????? ????????????.
	

    i = np.expand_dims((wh0 < 3.0), axis=1).any(1) # 3.0 ????????? ????????? bool ????????? ??????. any(1)??? ??? ???
    i = i.sum()
    if i:
        print(f'WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
    #wh = img_wh[(np.expand_dims(wh0, axis=1) >= 2.0).any(1)]  # Only box es with wh greater than or equal to 2 pixels are reserved
    # k_means cluster computing anchor
    k = k_means(resized_bboxes, n, use_iou=False, use_pp=False)
    k = k[np.argsort(k.prod(1))]  # sort small to large
    f, bpr = anchor_fitness(k, resized_bboxes, thr)
    print("kmeans: " + " ".join([f"[{int(i[0])}, {int(i[1])}]" for i in k]))
    print(f"fitness: {f:.5f}, best possible recall: {bpr:.5f}")
        
    # YOLOV5 improved genetic algorithm
    npr = np.random
    f, sh, mp, s = anchor_fitness(k, resized_bboxes, thr)[0], k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc=f'Evolving anchors with Genetic Algorithm:')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg, bpr = anchor_fitness(kg, resized_bboxes, thr)
        if fg > f:
            f, k = fg, kg.copy()
        pbar.desc = f'Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'

    # Sort by area
    k = k[np.argsort(k.prod(1))]  # sort small to large
    print("genetic: " + " ".join([f"[{int(i[0])}, {int(i[1])}]" for i in k]))
    print(f"fitness: {f:.5f}, best possible recall: {bpr:.5f}")

auto_anchor(img_size=416, n=9, thr=0.25, gen=1000, img_wh=imgs_wh, bbox_wh=bboxs_wh)