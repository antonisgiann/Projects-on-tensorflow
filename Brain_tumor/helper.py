import tensorflow as tf
from sklearn.model_selection import train_test_split


def brain_tumor_model(shape: tuple, num_classes: int):
    """
    Define the model
    inputs:
        shape: tuple in format (width, height, channels)
        num_classes: int, number of classes in the dataset
    returns:
        tf.keras.Model
    """
    data_augm = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=47),
    ])
    
    base_model = tf.keras.applications.efficientnet.EfficientNetB5(weights="imagenet",
                                                          include_top=False,
                                                          input_shape=shape,)
    # base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(weights="imagenet",
    #                                                       include_top=False,
    #                                                       input_shape=shape,)
    
    base_model.trainable = False
    
    inputs = tf.keras.Input(shape=shape)
    x = data_augm(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Dense(300,
                              kernel_regularizer=tf.keras.regularizers.l2(0.09), 
                              activity_regularizer=tf.keras.regularizers.l2(0.012),
                              bias_regularizer=tf.keras.regularizers.l2(0.012), 
                              activation= 'relu')(x)
    x = tf.keras.layers.Dropout(0.45)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)

    return tf.keras.Model(inputs, outputs)


def get_data_generators(df,
                        shape: tuple, 
                        batch_size: int = 96,
                        test_size: float = 0.2):
    """
    Returns three image data generators out of a dataframe, 
    one for train, one for valid and one for test
    inputs:
        df: pandas.DataFrame
        shape: tuple , image dimensions
        batch_size: int, batch size of the generators
    returns:
        train_gen: image data generator
        valid_gen: image data generator
        test_gen: image data generator
    """
    X_train, X_valid = train_test_split(df, test_size=test_size, random_state=47, stratify=df["tumor"])
    X_valid, X_test = train_test_split(X_valid, test_size=0.5, random_state=47, stratify=X_valid["tumor"])

    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True).flow_from_dataframe(
        X_train, x_col="filepath", y_col="tumor", target_size=shape, color_mode="rgb",
        class_mode="categorical", batch_size=batch_size, seed=47
    )
    valid_gen = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_dataframe(
        X_valid, x_col="filepath", y_col="tumor", target_size=shape, color_mode="rgb",
        class_mode="categorical", batch_size=batch_size, seed=47
    )
    test_gen = tf.keras.preprocessing.image.ImageDataGenerator().flow_from_dataframe(
        X_test, x_col="filepath", y_col="tumor", target_size=shape, color_mode="rgb",
        class_mode="categorical", batch_size=batch_size, seed=47
    )

    return train_gen, valid_gen, test_gen