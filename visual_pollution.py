# %%
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import gc
from utils import data_extractor, PollutionModel

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

train_ds = train_ds.batch(BATCH_SIZE).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
train_ds = train_ds.map(lambda img, label: (tf.cast(img, tf.float32), label))
test_ds = test_ds.map(lambda img, label: (tf.cast(img, tf.float32), label))

#############
###MODELS####
#############
model = PollutionModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        preds = model(images, training=True)
        loss = loss_object(labels, preds)
    grads = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    train_loss(loss)
    train_accuracy(labels,preds)

@tf.function
def test_step(images, labels):
    preds = model(images, training=False)
    loss = loss_object(labels, preds)

    test_loss(loss)
    test_accuracy(labels, preds)

# Training loop
epochs = 10
for i in range(epochs):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for img , label in train_ds:
        train_step(img, label)
        test_step(img, label)

    print(
            f"Epoch {i+1}, " +
            f"Loss {train_loss.result()}, " +
            f"Accuracy {train_accuracy.result()}, " +
            f"Test Loss {test_loss.result()}, " +
            f"Test Accuracy {test_accuracy.result()}"
        )
# %%x
