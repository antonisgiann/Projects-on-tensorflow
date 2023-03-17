# %%
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import gc
from utils import data_extractor

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
        (np.array(train_metadata["image"].tolist()), train_metadata["class"].astype(int)))
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
        (np.array(test_metadata["image"].tolist()), test_metadata["class"].astype(int)))
    test_ds.save(os.path.join(DATA_PATH, "test_dataset"))
else:
    train_ds = tf.data.Dataset.load(os.path.join(DATA_PATH, "train_dataset"))
    test_ds = tf.data.Dataset.load(os.path.join(DATA_PATH, "test_dataset"))

train_ds = train_ds.batch(BATCH_SIZE).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).cache().prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# %%
