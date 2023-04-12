# %%
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

from helper import object_map
from helper import ModelWrapper, EarlyStopLearningRateCallback
from helper import simple_dense_model, simple_conv_model, conv_model
from utils import plot_history

# Download the data and split test set to validation and test set
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_valid, X_test, y_valid, y_test = train_test_split(X_test, 
                                                    y_test, 
                                                    test_size=0.5, 
                                                    stratify=y_test, 
                                                    random_state=47)

BATCH_SIZE = 512
IMG_SHAPE = X_train.shape[1:]

# Show a small sample of the dataset
plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.title(object_map[y_train[i][0]])
    plt.imshow(X_train[i])
    plt.axis("off")
plt.show()

###############
####MODELS#####
###############
### Simple Dense model ###
# %% Get the simple dense model
model_den = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    simple_dense_model(shape=IMG_SHAPE)
])

# Compile and train the model
model_den.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

history_dense = model_den.fit(X_train, 
                    y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=20, 
                    validation_data=(X_valid,y_valid),
                    callbacks=[EarlyStopLearningRateCallback()]
                    )

# Plot training
plot_history(
    (history_dense.history["loss"],
     history_dense.history["val_loss"],
     history_dense.history["accuracy"],
     history_dense.history["val_accuracy"])
)
# %% Simple convolutional model
model_simple_conv = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    simple_conv_model(shape=IMG_SHAPE)
])

# Compile and train the model
model_simple_conv.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

history_simple_conv = model_simple_conv.fit(X_train, 
                    y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=20, 
                    validation_data=(X_valid,y_valid),
                    callbacks=[EarlyStopLearningRateCallback()]
                    )

# Plot training
plot_history(
    (history_simple_conv.history["loss"],
     history_simple_conv.history["val_loss"],
     history_simple_conv.history["accuracy"],
     history_simple_conv.history["val_accuracy"])
)
# %% Optimized convolutional model
model_opt_conv = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    conv_model(shape=IMG_SHAPE)
])

# Compile and train the model
model_opt_conv.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

history_opt_conv = model_opt_conv.fit(X_train, 
                    y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=20, 
                    validation_data=(X_valid,y_valid),
                    callbacks=[EarlyStopLearningRateCallback()]
                    )

# Plot training
plot_history(
    (history_opt_conv.history["loss"],
     history_opt_conv.history["val_loss"],
     history_opt_conv.history["accuracy"],
     history_opt_conv.history["val_accuracy"])
)
# %%
