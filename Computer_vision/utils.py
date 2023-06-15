import tensorflow as tf
import tensorflow.keras.layers as tfl
import matplotlib.pyplot as plt
import os
import zipfile


DONWLOADS_FOLDER = "/home/olaf/Downloads"


def data_extractor(file_name):
    # Extract the file
    if not os.path.exists(os.path.join("../datasets")):
        os.makedirs(os.path.join(os.getcwd(), "../datasets"))
    if not os.path.exists(os.path.join(os.path.join(f"../datasets/{file_name}"))):
        with zipfile.ZipFile(f"{DONWLOADS_FOLDER}/{file_name}.zip", "r") as zp:
            zp.extractall(os.path.join(f"../datasets/{file_name}"))


def plot_history(history, fine_tune_epoch=None, title=""):
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
    if fine_tune_epoch:
        ax[0].plot([fine_tune_epoch, fine_tune_epoch], [min(loss+val_loss)-0.1, max(loss+val_loss)], label="Fine tune point")
    ax[0].legend(loc="upper right")
    ax[1].plot(acc, label="Accuracy")
    ax[1].plot(val_acc, label="Val_Accuracy")
    if fine_tune_epoch:
        ax[1].plot([fine_tune_epoch, fine_tune_epoch], [min(acc+val_acc)-0.1, max(acc+val_acc)], label="Fine tune point")
    ax[1].legend(loc="lower right")
    plt.show()


def identity_block(input_tensor, filters, kernel_size=(3,3), bn_axis=3):
    """
    Residual identity convolutional block with three convolutions
    inputs:
        input_tensor: tensor in shape (batch, x, y, channel)
        filters: list or tuple with two elements
        kernel_size: kernel size of the second convolution
    return:
        tensor same dimensions as input_tensor
    """
    shape = input_tensor.shape
    f1, f2 = filters
    x = tfl.Conv2D(f1, (1,1))(input_tensor)
    x = tfl.BatchNormalization(axis=bn_axis)(x)
    x = tfl.Activation("relu")(x)

    x = tfl.Conv2D(f2, kernel_size, padding="same")(x)
    x = tfl.BatchNormalization(axis=bn_axis)(x)
    x = tfl.Activation("relu")(x)

    x = tfl.Conv2D(shape[-1], (1,1))(x)
    x = tfl.BatchNormalization(axis=bn_axis)(x)

    x = tfl.Add()([input_tensor, x])
    return tfl.Activation("relu")(x)


def conv_block(input_tensor, filters, kernel=(3,3), bn_axis=3):
    """
    Residual convolutional block with three convolutions
    inputs:
        input_tensor: tensor in shape (batch, x, y, channel)
        filters: list or tuple with three elements
        kernel_size: kernel size of the second convolution
    return:
        tensor dimensions defined by the third filter size
    """
    x = tfl.Conv2D(filters[0], (1,1), data_format="channels_last")(input_tensor)
    x = tfl.BatchNormalization(axis=bn_axis)(x)
    x = tfl.Activation("relu")(x)

    x = tfl.Conv2D(filters[1],
                           kernel, 
                           data_format="channels_last", 
                           padding="same")(x)
    x = tfl.BatchNormalization(axis=bn_axis)(x)
    x = tfl.Activation("relu")(x)

    x = tfl.Conv2D(filters[2], (1,1), data_format="channels_last")(x)
    x = tfl.BatchNormalization(axis=bn_axis)(x)

    short = tfl.Conv2D(filters[2], (1,1), data_format="channels_last")(input_tensor)
    short = tfl.BatchNormalization(axis=bn_axis)(short)
        
    x = tfl.Add()([x, short])
    return tfl.Activation("relu")(x)

def inverted_residual_block(inputs, expansion_factor, stride, filters):
    # Expansion phase
    in_channels = inputs.shape[-1]
    x = tf.keras.layers.Conv2D(expansion_factor * in_channels, kernel_size=1, strides=1, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Depthwise convolution
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Linear projection
    x = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Skip connection
    if stride == 1 and in_channels == filters:
        x = tf.keras.layers.Add()([x, inputs])

    return x

