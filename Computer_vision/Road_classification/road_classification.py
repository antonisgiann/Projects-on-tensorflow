# %%
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import plot_history, data_extractor
from helper import road_classification_model


PROJECT_NAME = __file__.split("/")[-1][:-3]
DATA_PATH = os.path.join(f"../datasets/{PROJECT_NAME}/Images")
BATCH_SIZE = 32
IMG_SHAPE = (224,224,3)

# Extract the file
data_extractor(PROJECT_NAME)

# Load the data
metadata = pd.read_csv(os.path.join(DATA_PATH, os.pardir, "metadata.csv"))
metadata = metadata.assign(
    images = lambda x: x["filename"].map(
        lambda y: cv2.resize(cv2.imread(os.path.join(DATA_PATH, "Images", y)), (224,224)),
    )
)
# %%
# Plot some images
class_names = ["clean", "dirty"]
plt.figure(figsize=(10,7))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(metadata["images"][i])
    plt.axis("off")
    plt.title(class_names[metadata["label"][i]])
plt.show()

# Create the training and validation datasets
train, valid = train_test_split(metadata, test_size=0.1, random_state=47)

train_ds = tf.data.Dataset.from_tensor_slices(
    (np.array(train["images"].tolist()), train["label"])
).shuffle(1000).batch(BATCH_SIZE)

valid_ds = tf.data.Dataset.from_tensor_slices(
    (np.array(valid["images"].tolist()), valid["label"])
).shuffle(1000).batch(BATCH_SIZE)

# Data augmentation
data_augm = tf.keras.models.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomZoom(0.1,0.1)
])

# Plot an augmented image
plt.figure(figsize=(10,7))
for img, labels in train_ds.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(data_augm(img[1]).numpy().astype("uint8"))
        plt.title(class_names[labels[1].numpy()])
        plt.axis("off")
        plt.show

# Create a new augmented dataset 
train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32), y))
train_ds2 = train_ds.map(lambda x, y: (data_augm(x), y))
train_ds3 = train_ds.map(lambda x, y: (data_augm(x), y))
train_ds4 = train_ds.map(lambda x, y: (data_augm(x), y))
merged_ds = (train_ds
             .concatenate(train_ds2)
             .concatenate(train_ds3)
             .concatenate(train_ds4))
##############
### MODELS ###
##############
base_mobilenet = tf.keras.applications.mobilenet_v2.MobileNetV2(
    weights="imagenet", include_top=False, input_shape=IMG_SHAPE
    )
# %% Without data augmentation
base_mobilenet.trainable = False
model =  road_classification_model(IMG_SHAPE, base_mobilenet)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=tf.keras.metrics.BinaryAccuracy()
)
initial_epochs = 40
history_initial = model.fit(train_ds, epochs=initial_epochs, validation_data=valid_ds)

# Fine tuning, freeze 70% of the baseline model 
fine_tune_epochs = 40

for l in model.layers[3].layers[int(0.7*len(model.layers[3].layers)):]:
    l.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=tf.keras.metrics.BinaryAccuracy())

history_fine = model.fit(train_ds,
                        epochs = initial_epochs + fine_tune_epochs,
                        initial_epoch=history_initial.epoch[-1],
                        validation_data=valid_ds)

# Plots
acc = history_initial.history["binary_accuracy"] + history_fine.history["binary_accuracy"]
loss = history_initial.history["loss"] + history_fine.history["loss"]
val_acc = history_initial.history["val_binary_accuracy"] + history_fine.history["val_binary_accuracy"]
val_loss = history_initial.history["val_loss"] + history_fine.history["val_loss"]

plot_history((loss, val_loss, acc, val_acc), initial_epochs, "Without data augmentation")

# %% With data augmentation
model_augm = road_classification_model(IMG_SHAPE, base_mobilenet, augmentation=data_augm)
model_augm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                   loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                   metrics=tf.keras.metrics.BinaryAccuracy())

initial_epochs = 40
fine_tune_epochs = 40
history_initial_augm = model_augm.fit(train_ds, epochs=initial_epochs, validation_data=valid_ds)

for l in model_augm.layers[3].layers[int(0.7*len(model_augm.layers[3].layers)):]:
    l.trainable = True

model_augm.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=tf.keras.metrics.BinaryAccuracy())

history_fine_augm = model_augm.fit(train_ds,
                        epochs = initial_epochs + fine_tune_epochs,
                        initial_epoch=history_initial_augm.epoch[-1],
                        validation_data=valid_ds)

# Plots
acc = history_initial_augm.history["binary_accuracy"] + history_fine_augm.history["binary_accuracy"]
loss = history_initial_augm.history["loss"] + history_fine_augm.history["loss"]
val_acc = history_initial_augm.history["val_binary_accuracy"] + history_fine_augm.history["val_binary_accuracy"]
val_loss = history_initial_augm.history["val_loss"] + history_fine_augm.history["val_loss"]

plot_history((loss, val_loss, acc, val_acc), initial_epochs, "With data augmentation")

# %%
# %% Augmented dataset
base_mobilenet.trainable = False
model_merged =  road_classification_model(IMG_SHAPE, base_mobilenet)

model_merged.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=tf.keras.metrics.BinaryAccuracy()
)
initial_epochs = 40
fine_tune_epochs = 40
history_initial_merged = model_merged.fit(merged_ds, epochs=initial_epochs, validation_data=valid_ds)

# Fine tuning, freeze 70% of the baseline model_merged 

for l in model_merged.layers[3].layers[int(0.7*len(model_merged.layers[3].layers)):]:
    l.trainable = True

model_merged.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=tf.keras.metrics.BinaryAccuracy())

history_fine_merged = model_merged.fit(merged_ds,
                        epochs = initial_epochs + fine_tune_epochs,
                        initial_epoch=history_initial_merged.epoch[-1],
                        validation_data=valid_ds)

# Plots
acc = history_initial_merged.history["binary_accuracy"] + history_fine_merged.history["binary_accuracy"]
loss = history_initial_merged.history["loss"] + history_fine_merged.history["loss"]
val_acc = history_initial_merged.history["val_binary_accuracy"] + history_fine_merged.history["val_binary_accuracy"]
val_loss = history_initial_merged.history["val_loss"] + history_fine_merged.history["val_loss"]

plot_history((loss, val_loss, acc, val_acc), initial_epochs, "On the augmented dataset")
# %% Predictions on the Validation set
# Retrieve a batch of images from the test set
image_batch, label_batch = valid_ds.as_numpy_iterator().next()
predictions = model_merged.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predictions[i]])
    plt.axis("off")
# %%
