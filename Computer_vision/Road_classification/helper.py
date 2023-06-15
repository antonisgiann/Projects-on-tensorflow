import tensorflow as tf
import tensorflow.keras.layers as tfl


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