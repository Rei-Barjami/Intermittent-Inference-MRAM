modelM = quantized_model

def accuracyCalc(modelM):
	model_with_noise2 = tf.keras.Sequential()
	for j, layer in enumerate(modelM.layers):
	    model_with_noise2.add(layer)
	    if j in indices:
	      model_with_noise2.add(BinaryBitFlipNoiseLayer(bit_flip_probability=1e-3, seed=d))

	model_with_noise2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	print("QL4")
	print(model_with_noise2.evaluate(test_images, test_labels)[1])


	model_with_noise2 = tf.keras.Sequential()
	for j, layer in enumerate(modelM.layers):
	    model_with_noise2.add(layer)
	    if j in indices:
	      model_with_noise2.add(BinaryBitFlipNoiseLayer(bit_flip_probability=1e-4, seed=d))

	model_with_noise2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	print("QL3")
	print(model_with_noise2.evaluate(test_images, test_labels)[1])

	model_with_noise2 = tf.keras.Sequential()
	for j, layer in enumerate(modelM.layers):
	    model_with_noise2.add(layer)
	    if j in indices:
	      model_with_noise2.add(BinaryBitFlipNoiseLayer(bit_flip_probability=1e-5, seed=d))
	model_with_noise2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	print("QL2")
	print(model_with_noise2.evaluate(test_images, test_labels)[1])

	
	model_with_noise2 = tf.keras.Sequential()
	for j, layer in enumerate(modelM.layers):
	    model_with_noise2.add(layer)
	    if j in indices1:
	      model_with_noise2.add(BinaryBitFlipNoiseLayer(bit_flip_probability=1e-6, seed=d))

	model_with_noise2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	print("QL1")
	print(model_with_noise2.evaluate(test_images, test_labels)[1])
	
