import tensorflow as tf

def brain_tumor_model(shape, num_classes):
    """
    Define the model
    shape: tuple in format (width, height, channels)
    num_classes: int, number of classes in the dataset
    """
    data_augm = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=47),
    ])
    
    # base_model = tf.keras.applications.efficientnet.EfficientNetB3(weights="imagenet",
    #                                                       include_top=False,
    #                                                       input_shape=shape,
    #                                                       pooling="max")
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights="imagenet",
                                                          include_top=False,
                                                          input_shape=shape,
                                                          pooling="max")
    
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=shape)
    x = data_augm(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Dense(300,
                              kernel_regularizer=tf.keras.regularizers.l2(0.09), 
                              activity_regularizer=tf.keras.regularizers.l2(),
                              bias_regularizer=tf.keras.regularizers.l2(), 
                              activation= 'relu')(x)
    x = tf.keras.layers.Dropout(0.45)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)

    return tf.keras.Model(inputs, outputs)