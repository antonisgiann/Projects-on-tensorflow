# %%
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

from helper import object_map
from helper import ModelWrapper, EarlyStopLearningRateCallback
from helper import simple_dense_model, simple_conv_model, conv_model, residual_model, bottleneck_model
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
wrap_simple_den = ModelWrapper(model_den, tf.keras.optimizers.Adamax(learning_rate=0.001))

history_dense = wrap_simple_den.fit(X_train, 
                    y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=20, 
                    validation_data=(X_valid,y_valid),
                    )

# Plot training
plot_history(
    (history_dense["loss"],
     history_dense["val_loss"],
     history_dense["accuracy"],
     history_dense["val_accuracy"])
)
# %% Simple convolutional model
model_simple_conv = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    simple_conv_model(shape=IMG_SHAPE)
])

# Compile and train the model
wrap_simple_conv = ModelWrapper(model_simple_conv, tf.keras.optimizers.Adamax(learning_rate=0.001))

history_simple_conv = wrap_simple_conv.fit(X_train, 
                    y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=20, 
                    validation_data=(X_valid,y_valid),
                    )

# Plot training
plot_history(
    (history_simple_conv["loss"],
     history_simple_conv["val_loss"],
     history_simple_conv["accuracy"],
     history_simple_conv["val_accuracy"])
)
# %% Optimized convolutional model
model_opt_conv = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.RandomFlip("horizontal"),
    conv_model(shape=IMG_SHAPE)
])

# Compile and train the model
wrap_opt_conv = ModelWrapper(model_opt_conv, tf.keras.optimizers.Adamax(learning_rate=0.001))

history_opt_conv = wrap_opt_conv.fit(X_train, 
                    y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=40, 
                    validation_data=(X_valid,y_valid),
                    )

# Plot training
plot_history(
    (history_opt_conv["loss"],
     history_opt_conv["val_loss"],
     history_opt_conv["accuracy"],
     history_opt_conv["val_accuracy"])
)
# %% Residual convolutional model
model_resnet = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.RandomFlip("horizontal"),
    residual_model(shape=IMG_SHAPE)
])

# Compile and train the model
wrap_resnet = ModelWrapper(model_resnet, tf.keras.optimizers.Adamax(learning_rate=0.001))

history_resnet = wrap_resnet.fit(X_train, 
                    y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=25, 
                    validation_data=(X_valid,y_valid),
                    )

# Plot training
plot_history(
    (history_resnet["loss"],
     history_resnet["val_loss"],
     history_resnet["accuracy"],
     history_resnet["val_accuracy"])
)
# %% Bottleneck model
model_bottleneck = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    bottleneck_model(shape=IMG_SHAPE)
])

wrap_bottleneck = ModelWrapper(model_bottleneck, tf.keras.optimizers.Adamax(learning_rate=0.001))

history_bottleneck = wrap_bottleneck.fit(X_train, 
                    y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=20, 
                    validation_data=(X_valid,y_valid),
                    )

# Plot training
plot_history(
    (history_bottleneck["loss"],
     history_bottleneck["val_loss"],
     history_bottleneck["accuracy"],
     history_bottleneck["val_accuracy"])
)
# %%
