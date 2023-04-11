import tensorflow as tf
import time
import numpy as np

object_map = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck"
}

class ModelWrapper():
    def __init__(self, model):
        self.model = model

    def compile(self, optimizer, loss_fn):
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @tf.function
    def fit(self, x_train, y_train, epochs, batch_size, validation_data):
        """
        Defines a custom training loop
        inputs:
            x_train: numpy array with the training dataset
            y_train: numpy array with the labels
            epochs: int number of epochs
            batch_size: int batch size
            validation_data: tuple in format (x_validation, y_validation)
        """
        # work in progress
        train_accuracy_calc = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
        valid_accuracy_calc = tf.keras.metrics.SparseCategoricalAccuracy(name="valid_accuracy")
        epoch_train_loss = []
        epoch_train_accuracy = []
        epoch_valid_loss = []
        epoch_valid_accuracy = []
        for i in range(epochs):
            train_loss, train_accuracy, valid_loss, valid_accuracy = [], [], [], []

            # work in progress
            # Batch training
            for j in range(x_train.shape[0]//batch_size):
                x_train_b = x_train[j: j + batch_size]
                y_train_b = y_train[j: j + batch_size]
                loss, preds = self.train_step(x_train_b, y_train_b)
                train_loss.append(loss)
                train_accuracy.append(train_accuracy_calc(y_train_b, preds).numpy())
            
            # Validation
            for j in range(validation_data[0].shape[0]//batch_size):
                x_valid_b = validation_data[0][j: j + batch_size]
                y_valid_b = validation_data[0][j: j + batch_size]
                valid_preds = self.model(x_valid_b, y_valid_b)
                t_loss = self.loss_fn(y_valid_b, valid_preds)
                valid_loss.append(t_loss)
                valid_accuracy.append(valid_accuracy_calc(y_valid_b, valid_preds).numpy())
            
            # Save metrics
            epoch_train_loss.append(np.mean(train_loss))
            epoch_train_accuracy.append(np.mean(train_accuracy))
            epoch_valid_loss.append(np.mean(valid_loss))
            epoch_valid_accuracy.append(np.mean(valid_accuracy))


    @tf.function
    def train_step(self, x_train, y_train):
        with tf.GradientTape() as tape:
            preds = self.model(x_train, training=True)
            loss = self.loss_fn(y_train, preds)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.mode.trainable_variables))

        return loss, preds

    def get_model(self):
        return self.model
    

class EarlyStopLearningRateCallback(tf.keras.callbacks.Callback):
    def __init__(self, lr_patience=3, stop_patience=5):
        super(EarlyStopLearningRateCallback, self).__init__()
        self.lr_patience = lr_patience
        self.stop_patience = stop_patience
        self.best_weights = None
        self.best_acc = 0
        self.lr_wait = 0
        self.stop_wait = 0

    def on_train_begin(self, logs=None):
        self.start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """
        This function runs at the end of each epoch and
        checks if the accuracy of the network improves.
        if not and a number of epochs has passed, decrease learning rate by 50%
        if not and a number of epochs has passed, stop training
        """
        current_acc = logs.get("val_accuracy")
        if current_acc > self.best_acc:
            self.best_weights = self.model.get_weights()
            self.best_acc = current_acc
            self.lr_wait = 0
            self.stop_wait = 0
        else:
            self.lr_wait += 1
            self.stop_wait += 1
            if self.lr_wait == self.lr_patience:
                current_lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
                new_lr = current_lr * 0.5
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                print("\nThe learning rate has changed, old value was {0} and the new value is {1}".format(current_lr, new_lr))
                self.lr_wait = 0
            if self.stop_wait == self.stop_patience:
                self.model.stop_training = True
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs):
        train_end = time.time() - self.start
        hours = int(train_end//3600)
        minutes = int((train_end - hours*3600)//60)
        seconds = round(train_end - hours*3600 - minutes*60, 0)
        print(f"Total training time -> {hours} hours, {minutes:2.0f} minutes, {seconds:2.2f} seconds")
        self.model.set_weights(self.best_weights)


def simple_dense_model(shape: tuple):
        """
        Defines a simple dense neural network
        inputs:
            shape: tuple, the shape of the input of the network
        returns:
            tf.keras.Model
        """
        inputs = tf.keras.Input(shape=shape)
        x = tf.keras.layers.Flatten(input_shape=shape)(inputs)
        x = tf.keras.layers.Dense(500, 
                                  activation="relu", 
                                  kernel_regularizer=tf.keras.regularizers.l2(),
                                  bias_regularizer=tf.keras.regularizers.l2())(x)
        x = tf.keras.layers.Dense(500, 
                                  activation="relu", 
                                  kernel_regularizer=tf.keras.regularizers.l2(),
                                  bias_regularizer=tf.keras.regularizers.l2())(x)
        x = tf.keras.layers.Dense(500, 
                                  activation="relu", 
                                  kernel_regularizer=tf.keras.regularizers.l2(),
                                  bias_regularizer=tf.keras.regularizers.l2())(x)
        x = tf.keras.layers.Dense(500, 
                                  activation="relu", 
                                  kernel_regularizer=tf.keras.regularizers.l2(),
                                  bias_regularizer=tf.keras.regularizers.l2())(x)
        x = tf.keras.layers.Dense(500, 
                                  activation="relu", 
                                  kernel_regularizer=tf.keras.regularizers.l2(),
                                  bias_regularizer=tf.keras.regularizers.l2())(x)
        outputs = tf.keras.layers.Dense(10)(x)

        return tf.keras.Model(inputs, outputs)  

def simple_conv_model(shape: tuple):
    """
    Defines a simple convolutional neural network
    inputs:
        shape: tuple, the shape of the input of the network
    returns:
        tf.keras.Model
    """  
    inputs = tf.keras.Input(shape=shape)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation="relu")(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(100, activation="relu")(x)
    outputs = tf.keras.layers.Dense(10)(x)

    return tf.keras.Model(inputs, outputs)

def conv_model(shape: tuple):
    """
    Defines a convolutional neural network with optimizations
    """
    if tf.keras.backend.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    inputs = tf.keras.Input(shape=shape)
    x = tf.keras.layers.Conv2D(filters=32, 
                               kernel_size=(5,5),
                               kernel_regularizer=tf.keras.regularizers.l2(0.011))(inputs)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(filters=64, 
                               kernel_size=(3,3), 
                               padding="same",
                               kernel_regularizer=tf.keras.regularizers.l2(0.011))(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(filters=128, 
                               kernel_size=(3,3), 
                               padding="same",
                               kernel_regularizer=tf.keras.regularizers.l2(0.011))(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(filters=256, 
                               kernel_size=(3,3),
                               kernel_regularizer=tf.keras.regularizers.l2(0.011))(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(100, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.45)(x)
    outputs = tf.keras.layers.Dense(10)(x)

    return tf.keras.Model(inputs, outputs)
  