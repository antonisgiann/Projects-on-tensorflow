# %%
import tensorflow as tf
import sys
sys.path.append("..")
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from utils import plot_history, data_extractor
from helper import brain_tumor_model
import opendatasets as od

PROJECT_URL = "https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c"
PROJECT_NAME = PROJECT_URL.split("/")[-1]
DATA_PATH = os.path.join("../datasets", PROJECT_NAME)
BATCH_SIZE = 96
IMG_SHAPE = (224,224,3)

od.download(PROJECT_URL, f"../datasets")

# Extract the data in the dataset folder in the cwd
# data_extractor(PROJECT_NAME)
# %%
num_classes = len(os.listdir(DATA_PATH))

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH, 
    subset="training",
    image_size=IMG_SHAPE[:-1],
    validation_split=0.2,
    seed=47,
    batch_size=BATCH_SIZE)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH, 
    subset="validation",
    image_size=IMG_SHAPE[:-1],
    validation_split=0.2,
    seed=47,
    batch_size=BATCH_SIZE)

train_ds = train_ds.map(lambda img, label: (img, tf.one_hot(label, depth=44)))
valid_ds = valid_ds.map(lambda img, label: (img, tf.one_hot(label, depth=44)))

model = brain_tumor_model(IMG_SHAPE, num_classes)
model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

history = model.fit(train_ds, epochs=20, validation_data=valid_ds)

plot_history((history.history["loss"],
              history.history["val_loss"],
              history.history["accuracy"],
              history.history["val_accuracy"]))

# %% fine tuning
for layer in model.layers[2].layers[int(len(model.layers[2].layers)*0.7):]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

history2 = model.fit(train_ds, epochs=40, initial_epoch=history.epoch[-1], validation_data=valid_ds)

plot_history((history.history["loss"] + history2.history["loss"],
              history.history["val_loss"] + history2.history["val_loss"],
              history.history["accuracy"] + history2.history["accuracy"],
              history.history["val_accuracy"] + history2.history["val_accuracy"]),
              20)
# %%
