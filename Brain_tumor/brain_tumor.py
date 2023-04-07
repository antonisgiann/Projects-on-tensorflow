# %%
import tensorflow as tf
import sys
sys.path.append("..")
import pandas as pd
import os
from utils import plot_history
from helper import brain_tumor_model, get_data_generators
import opendatasets as od

PROJECT_URL = "https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c"
PROJECT_NAME = PROJECT_URL.split("/")[-1]
DATA_PATH = os.path.join("../datasets", PROJECT_NAME)
BATCH_SIZE = 64
IMG_SHAPE = (456,456,3)

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

train_gen, valid_gen, test_gen = get_data_generators(metadata, IMG_SHAPE[:-1], BATCH_SIZE)

model = brain_tumor_model(IMG_SHAPE, num_classes)
model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

history = model.fit(train_gen, epochs=15, validation_data=valid_gen)

plot_history((history.history["loss"],
              history.history["val_loss"],
              history.history["accuracy"],
              history.history["val_accuracy"]))

# %% fine tuning
for layer in model.layers[2].layers[int(len(model.layers[2].layers)*0.7):]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.00015),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

history2 = model.fit(train_gen, epochs=30, initial_epoch=history.epoch[-1], validation_data=valid_gen)

plot_history((history.history["loss"] + history2.history["loss"],
              history.history["val_loss"] + history2.history["val_loss"],
              history.history["accuracy"] + history2.history["accuracy"],
              history.history["val_accuracy"] + history2.history["val_accuracy"]),
              20)
# %%
