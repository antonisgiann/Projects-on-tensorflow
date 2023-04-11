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

    def compile(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss_fn = loss
        self.train_accuracy_calc = metrics()
        self.valid_accuracy_calc = metrics()

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
        epoch_train_loss = []
        epoch_train_accuracy = []
        epoch_valid_loss = []
        epoch_valid_accuracy = []
        for i in range(epochs):
            # Batch training
            valid_loss_list = []
            valid_accuracy_list = []
            epoch_start = time.time()
            for j in range(x_train.shape[0]//batch_size):
                x_train_b = x_train[j: j + batch_size]
                y_train_b = y_train[j: j + batch_size]
                train_loss, train_preds = self.train_step(x_train_b, y_train_b)
            epoch_time = self.time_in_human_format(time.time() - epoch_start)
            # Validation
            for j in range(validation_data[0].shape[0]//batch_size):
                x_valid_b = validation_data[0][j: j + batch_size]
                y_valid_b = validation_data[1][j: j + batch_size]
                valid_preds = self.model(x_valid_b)
                valid_loss_list.append(self.loss_fn(y_valid_b, valid_preds).numpy())
                valid_accuracy_list.append(self.valid_accuracy_calc(y_valid_b, valid_preds).numpy())
            
            train_accuracy = self.train_accuracy_calc(y_train_b, train_preds).numpy()
            valid_loss = np.mean(valid_loss_list)
            valid_accuracy = np.mean(valid_accuracy_list)
            
            print(f"Epoch number {i}, training time: {epoch_time} -->  loss: {train_loss.numpy():.4f}, accuracy: {train_accuracy:.4f}, val_loss: {valid_loss:.4f}, val_accuracy: {valid_accuracy:.4f}")
            # Save metrics
            epoch_train_loss.append(train_loss.numpy())
            epoch_train_accuracy.append(train_accuracy)
            epoch_valid_loss.append(valid_loss)
            epoch_valid_accuracy.append(valid_accuracy)

        return {
            "loss": epoch_train_loss,
            "accuracy": epoch_train_accuracy,
            "val_loss": epoch_valid_loss,
            "val_accuracy": epoch_valid_accuracy
        }
    
    def train_step(self, x_train, y_train):
        with tf.GradientTape() as tape:
            preds = self.model(x_train, training=True)
            loss = self.loss_fn(y_train, preds)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss, preds
    
    def time_in_human_format(self, t):
        hours = int(t//3600)
        minutes = int((t - hours * 3600)//60)
        seconds = round(t - hours * 3600 - minutes * 60, 2)
        if hours > 0:
            return f"{hours} hours, {minutes} minutes, {seconds}s"
        elif minutes > 0:
            return f"{minutes} minutes, {seconds}s"
        else:
            return f"{seconds}s"

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
  