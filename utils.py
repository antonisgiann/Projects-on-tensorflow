import tensorflow as tf
import tensorflow.keras.layers as tfl
import matplotlib.pyplot as plt
import os
import zipfile


DONWLOADS_FOLDER = "/home/olaf/Downloads"


def data_extractor(file_name):
    # Extract the file
    if not os.path.exists(os.path.join(os.getcwd(), "datasets")):
        os.makedirs(os.path.join(os.getcwd(), "datasets"))
    if not os.path.exists(os.path.join(os.path.join(os.getcwd(), f"datasets/{file_name}"))):
        with zipfile.ZipFile(f"{DONWLOADS_FOLDER}/{file_name}.zip", "r") as zp:
            zp.extractall(os.path.join(os.getcwd(), f"datasets/{file_name}"))


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


def road_classification_model(shape, base_model, augmentation=None):
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
    x = tfl.Conv2D(256, (3,3), padding="same")(x)
    x = tfl.BatchNormalization(axis=1)(x)
    x = tfl.Activation("relu")(x)
    x = tfl.Conv2D(512, (3,3))(x)
    x = tfl.BatchNormalization(axis=1)(x)
    x = tfl.Activation("relu")(x)
    x = tfl.MaxPooling2D((2,2))(x)
    x = tfl.GlobalAveragePooling2D()(x)
    x = tfl.Dense(400, activation="relu")(x)
    x = tfl.Dropout(0.2)(x)
    outputs = tfl.Dense(1)(x)

    return  tf.keras.Model(inputs, outputs)


def pollution_model_transfer(shape):

    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights="imagenet", include_top=False, input_shape=shape)
    base_model.trainable = False
    inputs = tf.keras.Input(shape=shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x , training=False)
    x = conv_block(x, [256, 256, 512])
    x = identity_block(x, [256, 256])
    x = identity_block(x, [512, 512])
    x = tfl.GlobalAveragePooling2D()(x)
    x = tfl.Dense(256)(x)
    x = tfl.Dropout(0.4)(x)
    x = tfl.Dense(128)(x)
    x = tfl.Dropout(0.4)(x)
    outputs = tfl.Dense(11)(x)

    return tf.keras.models.Model(inputs, outputs)


def my_pollution_resnet(shape):

    inputs = tf.keras.Input(shape=shape)
    x = tfl.Conv2D(64, (7,7))(inputs)
    x = tfl.BatchNormalization(axis=3)(x)
    x = tfl.Activation("relu")(x)
    x = tfl.MaxPooling2D((2,2))(x)

    x = conv_block(x, [128,128, 256])
    x = identity_block(x, [128,128,256])
    x = identity_block(x, [128,128,256])

    x = tfl.GlobalAveragePooling2D()(x)
    outputs = tfl.Dense(11)(x)

    return tf.keras.Model(inputs, outputs)


def identity_block(input_tensor, filters, kernel_size=(3,3)):
    """
    Residual identity convolutional block with three convolutions
    inputs:
        input_tensor: tensor in shape (batch, x, y, channel)
        filters: list or tuple with three elements
        kernel_size: kernel size of the second convolution
    return:
        tensor same dimensions as input_tensor
    """
    shape = input_tensor.shape
    f1, f2 = filters
    x = tfl.Conv2D(f1, (1,1))(input_tensor)
    x = tfl.BatchNormalization(axis=3)(x)
    x = tfl.Activation("relu")(x)

    x = tfl.Conv2D(f2, kernel_size, padding="same")(x)
    x = tfl.BatchNormalization(axis=3)(x)
    x = tfl.Activation("relu")(x)

    x = tfl.Conv2D(shape[-1], (1,1))(x)
    x = tfl.BatchNormalization(axis=3)(x)

    x = tfl.Add()([input_tensor, x])
    return tfl.Activation("relu")(x)


def conv_block(input_tensor, filters, kernel=(3,3)):
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
    x = tfl.BatchNormalization(axis=3)(x)
    x = tfl.Activation("relu")(x)

    x = tfl.Conv2D(filters[1],
                           kernel, 
                           data_format="channels_last", 
                           padding="same")(x)
    x = tfl.BatchNormalization(axis=3)(x)
    x = tfl.Activation("relu")(x)

    x = tfl.Conv2D(filters[2], (1,1), data_format="channels_last")(x)
    x = tfl.BatchNormalization(axis=3)(x)

    short = tfl.Conv2D(filters[2], (1,1), data_format="channels_last")(input_tensor)
    short = tfl.BatchNormalization(axis=3)(short)
        
    x = tfl.Add()([x, short])
    return tfl.Activation("relu")(x)

