import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def datasetPre(dim=96):

	dataset_dir = 'flower_photos/'  # Adjust this path to your dataset location

	# Initialize lists to store images and labels
	images = []
	labels = []

	# Iterate through each class directory
	class_names = sorted(os.listdir(dataset_dir))
	for class_label, class_name in enumerate(class_names):
		class_path = os.path.join(dataset_dir, class_name)

		# Iterate through images in the class directory
		for filename in os.listdir(class_path):
			image_path = os.path.join(class_path, filename)

			# Load the image, convert to array, turn to grayscale, and append to the lists
			image = load_img(image_path, target_size=(dim, dim))  # You can adjust the target size
			image_array = img_to_array(image)
			images.append(image_array)
			labels.append(class_label)

	# Convert lists to numpy arrays
	images = np.array(images) / 127.5 -1
	labels = np.array(labels) 

	# Split the dataset into training and testing sets
	train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.15, random_state=545)

	# Print the shape of the resulting datasets
	print("Train images shape:", train_images.shape)
	print("Train labels shape:", train_labels.shape)
	print("Test images shape:", test_images.shape)
	print("Test labels shape:", test_labels.shape)
	return train_images,train_labels,test_images,test_labels
