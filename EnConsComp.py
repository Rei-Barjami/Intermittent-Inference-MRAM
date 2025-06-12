from keras import backend as K

def compute_cons(mcuTotalEnergy):
	modelM = quantized_model
	dims = {}
	total=0
	for j, layer in enumerate(modelM.layers):
	      if(j in indices1):
		output_tensor = layer.output
		output_shape = K.int_shape(output_tensor)

		# Replace None with 1 in the dimensions
		fixed_output_shape = [dim if dim is not None else 1 for dim in output_shape]

		total_elements = K.prod(fixed_output_shape)
		dims[j] = total_elements
		total=total+total_elements
	print(total)
	print(dims)

	tensor_size = tf.reduce_prod(dims[4])

	print(type(tensor_size.numpy()))
	enConsumptionQ0b=mcuTotalEnergy
	enConsumptionQ1b=mcuTotalEnergy
	enConsumptionQ2b=mcuTotalEnergy
	enConsumptionQ3b=mcuTotalEnergy
	enConsumptionQ4b=mcuTotalEnergy

	print(enConsumptionQ0b)
	print(enConsumptionQ1b)
	print(enConsumptionQ2b)
	print(enConsumptionQ3b)
	print(enConsumptionQ4b)
