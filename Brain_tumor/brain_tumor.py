# %%
import tensorflow as tf
import sys
sys.path.append("..")
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from utils import plot_history
from helper import brain_tumor_model
import opendatasets as od

PROJECT_URL = "https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c"
PROJECT_NAME = PROJECT_URL.split("/")[-1]
DATA_PATH = os.path.join("../datasets", PROJECT_NAME)
BATCH_SIZE = 96
IMG_SHAPE = (224,224,3)

od.download(PROJECT_URL, "../datasets")

files = []
classes = []
for c in os.listdir(DATA_PATH):
    for f in os.listdir(os.path.join(DATA_PATH, c)):
        classes.append(c)
        files.append(os.path.join(DATA_PATH, c, f))

metadata = pd.DataFrame({"filepath":files, "tumor":classes})
# %%
num_classes = len(os.listdir(DATA_PATH))

X_train, X_valid = train_test_split(metadata, test_size=0.2, random_state=47, stratify=metadata["tumor"])
X_valid, X_test = train_test_split(X_valid, test_size=0.5, random_state=47, stratify=X_valid["tumor"])

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True).flow_from_dataframe(
    X_train, x_col="filepath", y_col="tumor", target_size=(224,224), color_mode="rgb",
    class_mode="categorical", batch_size=BATCH_SIZE, seed=47
)
valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True).flow_from_dataframe(
    X_valid, x_col="filepath", y_col="tumor", target_size=(224,224), color_mode="rgb",
    class_mode="categorical", batch_size=BATCH_SIZE, seed=47
)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True).flow_from_dataframe(
    X_test, x_col="filepath", y_col="tumor", target_size=(224,224), color_mode="rgb",
    class_mode="categorical", batch_size=BATCH_SIZE, seed=47
)

model = brain_tumor_model(IMG_SHAPE, num_classes)
model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

history = model.fit(train_gen, epochs=20, validation_data=valid_gen)

plot_history((history.history["loss"],
              history.history["val_loss"],
              history.history["accuracy"],
              history.history["val_accuracy"]))

# %% fine tuning
for layer in model.layers[4].layers[int(len(model.layers[4].layers)*0.7):]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

history2 = model.fit(train_gen, epochs=40, initial_epoch=history.epoch[-1], validation_data=valid_gen)

plot_history((history.history["loss"] + history2.history["loss"],
              history.history["val_loss"] + history2.history["val_loss"],
              history.history["accuracy"] + history2.history["accuracy"],
              history.history["val_accuracy"] + history2.history["val_accuracy"]),
              20)
# %%
