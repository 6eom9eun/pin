import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from matplotlib import pyplot as plt

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


batch_size = 40
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

train_image_generator = ImageDataGenerator(rescale = 1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)
image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size, directory=train_dir, shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size, directory=validation_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='binary')

sample_training_images, _ = next(train_data_gen)

#def plotImages(images_arr):
    #fig, axes = plt.subplots(1, 5, figsize=(20,20))
    #axes = axes.flatten()
    #for img, ax in zip(images_arr, axes):
        #ax.imshow(img)
        #ax.axis('off')
    #plt.tight_layout()
    #plt.show()

#plotImages(sample_training_images[:5])

model = Sequential([
    Conv2D(16, 3, padding = 'same', activation = 'relu', input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding = 'same', activation = 'relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding = 'same', activation = 'relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
    ])

model.compile(optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'])

model.summary()

hist = model.fit(
    train_data_gen,
    steps_per_epoch=batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=batch_size,
)

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
