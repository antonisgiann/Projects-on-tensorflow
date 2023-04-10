# %%
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from helper import object_map, ModelWrapper, simple_dense_model, MyCallBack
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
# Get the simple dense model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    simple_dense_model(shape=IMG_SHAPE)
])

# Compile and train the model
model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

history = model.fit(X_train, 
                    y_train, 
                    batch_size=BATCH_SIZE, 
                    epochs=20, 
                    validation_data=(X_valid,y_valid),
                    callbacks=[MyCallBack()]
                    )

# Plot training
plot_history(
    (history.history["loss"],
     history.history["val_loss"],
     history.history["accuracy"],
     history.history["val_accuracy"])
)
# %%
