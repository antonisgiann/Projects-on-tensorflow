import tensorflow as tf
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

    @tf.function
    def fit(self, x_train, y_train, epochs, batch_size, validation_data, loss, optimizer):
        for i in range(epochs):
            pass
            # work in progress

    def get_model(self):
        return self.model
    

class MyCallBack(tf.keras.callbacks.Callback):
    def __init__(self, lr_patience=3, stop_patience=5):
        super(MyCallBack, self).__init__()
        self.lr_patience = lr_patience
        self.stop_patience = stop_patience
        self.best_weights = None
        self.best_acc = 0
        self.lr_wait = 0
        self.stop_wait = 0

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
  