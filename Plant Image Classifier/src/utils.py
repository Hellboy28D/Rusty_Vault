# Set seeds for reproducibility
import random
random.seed(0)

import numpy as np
np.random.seed(0) 
import tensorflow as tf
tf.random.set_seed(0)


# Importing the dependencies
import os
import json
from zipfile import ZipFile
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models