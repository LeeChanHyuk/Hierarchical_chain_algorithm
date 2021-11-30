import json
import os
import cv2

face_json_path = './00033/appleFace.json'
left_eye_json_path = './00033/appleLeftEye.json'
right_eye_json_path = './00033/appleRightEye.json'
image_folder_path = './00033/frames'

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
			for index, file in enumerate(os.listdir(image_folder_path)):
				if face_valid[index] == 0 or left_eye_valid[index] == 0 or right_eye_valid[index] == 0:
					continue
				# Draw face bounding box
				img = cv2.imread(os.path.join(image_folder_path, file))
				x = int(face_x_coordinate[index])
				y = int(face_y_coordinate[index])
				w = int(face_width[index])
				h = int(face_height[index])
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
				cv2.imshow('img',img)
				cv2.waitKey(100)