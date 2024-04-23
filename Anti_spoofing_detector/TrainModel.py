import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

parentDir = os.getcwd()

fake_path = r"D:\Computer Vision\Murtaza_\Anti Spoofing_Liveliness Detector\Testing_Scripts\Dataset\Fake"
real_path = r"D:\Computer Vision\Murtaza_\Anti Spoofing_Liveliness Detector\Testing_Scripts\Dataset\Real"

# Define ImageDataGenerator for data augmentation and normalization

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

def load_images_and_labels(image_paths, labels):
    images = []
    label_list = []

    for image_path, label in zip(image_paths, labels):
        img = cv2.imread(image_path)
        img=cv2.resize(img,(150, 150))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
        label_list.append(label)
        
    return np.array(images), np.array(label_list)

fake_images, fake_labels = load_images_and_labels(
    [os.path.join(fake_path, f) for f in os.listdir(fake_path) if f.endswith(('.png', '.jpg', '.jpeg'))], [1] * len(os.listdir(fake_path)))

real_images, real_labels = load_images_and_labels(
    [os.path.join(real_path, f) for f in os.listdir(real_path) if f.endswith(('.png', '.jpg', '.jpeg'))], [0] * len(os.listdir(real_path)))

all_images = np.concatenate((real_images, fake_images), axis=0)
all_labels = np.concatenate((real_labels, fake_labels), axis=0)

#---------------------- Shuffle the data ------------------
indices = np.arange(all_images.shape[0])
np.random.shuffle(indices)
all_images = all_images[indices]
all_labels = all_labels[indices]

#-------------------- Split the data into train, validation, and test sets --------------------
train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

#--------------------- Create data generators ---------------------------------
train_generator = datagen.flow(train_images, train_labels, batch_size=16)
val_generator = datagen.flow(val_images, val_labels, batch_size=16)

image_height, image_width = all_images.shape[1], all_images.shape[2]

#---------------------------- Define the model ----------------------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(image_height, image_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

#-------------------------------- Train the model -------------------------------------------
history = model.fit(train_generator, steps_per_epoch=len(train_images) // 16, epochs=10, validation_data=val_generator, validation_steps=len(val_images) // 16)
model.save("anti_spoof_model.h5")