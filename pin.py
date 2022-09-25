import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

PATH = os.path.join('C:/Users/6eom9eun/Desktop/pin')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
train_normal_dir = os.path.join(train_dir, 'normal')
train_defect_dir = os.path.join(train_dir, 'defect')
validation_normal_dir = os.path.join(validation_dir, 'normal')
validation_defect_dir = os.path.join(validation_dir, 'defect')

num_normal_tr = len(os.listdir(train_normal_dir))
num_defect_tr = len(os.listdir(train_defect_dir))

num_normal_val = len(os.listdir(validation_normal_dir))
num_defect_val = len(os.listdir(validation_defect_dir))

total_train = num_normal_tr + num_defect_tr
total_val = num_normal_val + num_defect_val

print('total training normal images:', num_normal_tr)
print('total training defect images:', num_defect_tr)
print('total validation normal images:', num_normal_val)
print('total validaation defect images:', num_defect_val)
print("-------------------------------------------------")
print("Total training images:", total_train)
print("Total validation images:", total_val)