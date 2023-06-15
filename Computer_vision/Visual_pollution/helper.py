import sys
sys.path.append("..")
import tensorflow as tf
import tensorflow.keras.layers as tfl
from utils import conv_block, identity_block


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


