import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


def flip_float_bits(input_float, bit_position):
    float_bits = tf.bitcast(input_float, tf.uint32)
    # Create a bitmask with a 1 at the specified bit_position
    bitmask = 1 << bit_position
    # Use bitwise XOR to flip the bit
    modified_float_bits = tf.bitwise.bitwise_xor(float_bits, bitmask)
    # Convert the modified bits back to a float
    modified_float = tf.bitcast(modified_float_bits, tf.float32)
    return modified_float

class BinaryBitFlipNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, bit_flip_probability=1e-3, seed=5, **kwargs):

        super(BinaryBitFlipNoiseLayer, self).__init__(**kwargs)
        tf.random.set_seed(seed)
        self.bit_flip_probability = bit_flip_probability

    def call(self, inputs, training=None):
            # Generate random noise
            noise = tf.random.uniform(tf.shape(inputs), dtype=tf.float32)
            # Determine which bits to flip
            flip_bits = noise < self.bit_flip_probability

            # Apply bit-flip noise to each element in the tensor
            noisy_output = tf.identity(inputs)
            for bit_position in range(32):
                flip_condition = flip_bits
                noisy_output = tf.where(flip_condition, flip_float_bits(noisy_output, bit_position), noisy_output)
            return noisy_output

    def get_config(self):
        config = super(BinaryBitFlipNoiseLayer, self).get_config()
        config['bit_flip_probability'] = self.bit_flip_probability
        return config
