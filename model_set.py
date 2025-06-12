import tensorflow as tf
import math

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
from tensorflow_model_optimization.quantization.keras import quantize_model
from tensorflow_model_optimization.quantization.keras import QuantizeConfig
import tensorflow_model_optimization as tfmot
from keras import backend as K
from tensorflow.keras.optimizers import Adam

def modelSet(train_images, train_labels, test_images, test_labels, model_name='mobilenet_v1_0.25_96.h5'):
	# Load the H5 model file
	model = load_model(model_name)
	
	# Initialize an empty list to store the indices
	indices = []

	#Iterate through the layers and collect indices of ReLU layers
	for idx, layer in enumerate(model.layers):
		if isinstance(layer, tf.keras.layers.ReLU):
			indices.append(idx)
		if isinstance(layer, tf.keras.layers.ZeroPadding2D):
			indices.append(idx)
		if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
			indices.append(idx)
		if isinstance(layer, tf.keras.layers.Activation):
			indices.append(idx)
	print("Indices of layers:", indices)

	indices = [idx + 1 for idx in indices]

	print(indices)
	model.compile(optimizer='adam',
		                loss='sparse_categorical_crossentropy',
		                metrics=['accuracy'])

	model.evaluate(test_images,test_labels)

	#model.summary()
	optimizer = Adam(learning_rate=0.00001)
	quantized_model = quantize_model(model)

	quantized_model.compile(optimizer=optimizer,
		                loss='sparse_categorical_crossentropy',
		                metrics=['accuracy'])

	datagen = tf.keras.preprocessing.image.ImageDataGenerator(
	    featurewise_center=True,
	    featurewise_std_normalization=False
	)

	train_generator = datagen.flow(train_images, train_labels, batch_size=64)

	quantized_model.fit(train_generator, epochs=18, steps_per_epoch=len(train_images)//64, verbose=2)

	results = quantized_model.evaluate(test_images, test_labels)
	#quantized_model.summary()

	return indices, quantized_model
