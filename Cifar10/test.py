import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

from helper import ModelWrapper, EarlyStopLearningRateCallback

# Download the data and split test set to validation and test set
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_valid, X_test, y_valid, y_test = train_test_split(X_test, 
                                                    y_test, 
                                                    test_size=0.5, 
                                                    stratify=y_test, 
                                                    random_state=47)

BATCH_SIZE = 512
IMG_SHAPE = X_train.shape[1:]

# Testing custom training loop
test_model1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=IMG_SHAPE),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10)
])

wrap = ModelWrapper(test_model1)

history1 = wrap.fit(X_train, y_train, epochs=15, batch_size=BATCH_SIZE, 
                   validation_data=(X_valid, y_valid))

test_model2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=IMG_SHAPE),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10)
])

test_model2.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=tf.keras.metrics.SparseCategoricalAccuracy())

history2 = test_model2.fit(X_train, y_train, epochs=15, batch_size=BATCH_SIZE, validation_data=(X_valid, y_valid))

plt.plot(history1["val_loss"], label="val_loss_custom")
plt.plot(history2.history["val_loss"], label="val_loss")
plt.legend()
plt.show()
plt.plot(history1["val_accuracy"], label="val_accuracy_custom")
plt.plot(history2.history["val_sparse_categorical_accuracy"], label="val_accuracy")
plt.legend()
plt.show()
plt.plot(history1["accuracy"], label="accuracy_custom")
plt.plot(history2.history["sparse_categorical_accuracy"], label="accuracy")
plt.legend()
plt.show()