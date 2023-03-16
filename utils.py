import tensorflow as tf
import matplotlib.pyplot as plt

def plot_history(history, fine_tune_epoch, title=""):
    """
    Plots the history
    Args:
        history: tuple in format (loss, validation loss, accuracy, validation accuracy)
    """
    loss, val_loss, acc, val_acc = history
    plt.figure(figsize=(14,10))
    fig, ax = plt.subplots(1,2)
    fig.suptitle(title)
    ax[0].plot(loss, label="Loss")
    ax[0].plot(val_loss, label="Val_loss")
    ax[0].plot([fine_tune_epoch, fine_tune_epoch], [0, max(loss+val_loss)], label="Fine tune point")
    ax[0].legend(loc="upper right")
    ax[1].plot(acc, label="Accuracy")
    ax[1].plot(val_acc, label="Val_Accuracy")
    ax[1].plot([fine_tune_epoch, fine_tune_epoch], [0, max(acc+val_acc)], label="Fine tune point")
    ax[1].legend(loc="lower right")
    plt.show()


def make_model(shape, base_model, augmentation=None):
    """
    Returned a model using as feature extractor the base_model
    Args:
        shape: tuple, the input shape of the model
        base_model: the base model
        augmentation: data augmentation step or steps wrapped in a tf.keras.models.Sequential
    Returns:
        tf.keras.Model
    """
    base = tf.keras.models.clone_model(base_model)
    base.set_weights(base_model.get_weights())

    inputs = tf.keras.Input(shape=shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    if augmentation:
        x = augmentation(x)
    x = base(x, training=False)
    x = tf.keras.layers.Conv2D(256, (3,3), padding="same")(x)
    x = tf.keras.layers.BatchNormalization(axis=1)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(512, (3,3))(x)
    x = tf.keras.layers.BatchNormalization(axis=1)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D((2,2))(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(400, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1)(x)

    return  tf.keras.Model(inputs, outputs)