from EnConsComp import *
from accuracyCalc import *
from dataset_p import *
from errorInj import *
from model_set import *

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import math
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow_model_optimization.quantization.keras import quantize_model
from tensorflow_model_optimization.quantization.keras import QuantizeConfig
import tensorflow_model_optimization as tfmot
from keras import backend as K
from saveTheTrace import *

from tensorflow.python.client import device_lib
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np


train_images,train_labels,test_images,test_labels = datasetPre(dim=128)
indices, quantized_model= modelSet(train_images,train_labels,test_images,test_labels,'fdmobilenet_0.25_128.h5')
compute_cons(2.8176e-05)
accuracyCalc(quantized_model)
