# %%
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import gc
from utils import data_extractor, my_pollution_resnet, plot_history, pollution_model_transfer
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

PROJECT_NAME = __file__.split("/")[-1][:-3]
DATA_PATH = os.path.join(os.getcwd(), "datasets", PROJECT_NAME)
BATCH_SIZE = 32
IMG_SHAPE = (224,224,3)

data_extractor(PROJECT_NAME)

train_metadata = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
test_metadata = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))
class_names = dict(zip(train_metadata["class"].astype(int), train_metadata["name"]))

if not os.path.exists(os.path.join(DATA_PATH, "train_dataset")):
    train_metadata = train_metadata.assign(
        image = lambda x: x["image_path"]
            .map(lambda img_path: 
                cv2.resize(
                        cv2.imread(os.path.join(DATA_PATH, "images/images", img_path)),
                        (IMG_SHAPE[0], IMG_SHAPE[1]))
                )
    )
    train_ds = tf.data.Dataset.from_tensor_slices(
        (np.array(train_metadata["image"].tolist()).astype(np.float32), train_metadata["class"].astype(int)))
    train_ds.save(os.path.join(DATA_PATH, "train_dataset"))
    
    test_metadata = test_metadata.assign(
        image = lambda x: x["image_path"]
            .map(lambda img_path: 
                cv2.resize(
                        cv2.imread(os.path.join(DATA_PATH, "images/images", img_path)),
                        (IMG_SHAPE[0], IMG_SHAPE[1]))
                )
    )
    test_ds = tf.data.Dataset.from_tensor_slices(
        (np.array(test_metadata["image"].tolist()).astype(np.float32), test_metadata["class"].astype(int)))
    test_ds.save(os.path.join(DATA_PATH, "test_dataset"))
else:
    train_ds = tf.data.Dataset.load(os.path.join(DATA_PATH, "train_dataset"))
    test_ds = tf.data.Dataset.load(os.path.join(DATA_PATH, "test_dataset"))

# Create a testset
num_of_batches = tf.data.experimental.cardinality(test_ds).numpy()
valid_ds = test_ds.take(int(0.8 * num_of_batches))
test_ds = test_ds.skip(int(0.8 * num_of_batches))

train_ds = train_ds.batch(BATCH_SIZE).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
train_ds = train_ds.map(lambda img, label: (tf.cast(img, tf.float32), label))
test_ds = test_ds.map(lambda img, label: (tf.cast(img, tf.float32), label))

# %%
#############
###MODELS####
#############
# %% Transfer learning on mobilenet_v2
model = pollution_model_transfer(IMG_SHAPE)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=["accuracy"])

history = model.fit(train_ds, epochs=15, validation_data=test_ds)
# Fine tune 
for l in model.layers[3].layers[int(0.7*len(model.layers[3].layers)):]:
    l.trainable=True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
               metrics=["accuracy"])

history2 = model.fit(train_ds, 
                      epochs=35,
                      validation_data=test_ds, 
                      initial_epoch=history.epoch[-1])
# Plots
plot_history(
    (history.history["loss"] + history2.history["loss"],
        history.history["val_loss"] + history2.history["val_loss"],
        history.history["accuracy"] + history2.history["accuracy"],
        history.history["accuracy"] + history2.history["val_accuracy"]),
    history.epoch[-1]
)
# %%
