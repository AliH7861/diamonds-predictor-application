"""Model builders for clarity classification."""


TUNED_ANN_PARAMETERS = {
    "batch_size": 128,
    "class_weight_power": 0.5,
    "activation": "relu",
    "dropout": 0.1654555859304803,
    "l2_strength": 0.00035530339266550635,
    "batchnorm": True,
    "learning_rate": 0.0018248766726067505,
    "units": [96, 128, 128, 256, 384, 512],
}


def build_ann(input_dim: int, seed: int = 42):
    """Build the ANN architecture used in the maintained classification notebook."""
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(5, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_ordinal_ann(input_dim: int, seed: int = 42):
    """Build an ANN that predicts the four ordered clarity boundaries."""
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(192, activation="gelu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.08),
            tf.keras.layers.Dense(128, activation="gelu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.08),
            tf.keras.layers.Dense(64, activation="gelu"),
            tf.keras.layers.Dense(4, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
    )
    return model


def build_tuned_ann(input_dim: int, seed: int = 42):
    """Rebuild the tuned six-layer ANN preserved by the Y17 experiment."""
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    regularizer = tf.keras.regularizers.l2(TUNED_ANN_PARAMETERS["l2_strength"])
    layers = [tf.keras.layers.Input(shape=(input_dim,))]

    for units in TUNED_ANN_PARAMETERS["units"]:
        layers.append(
            tf.keras.layers.Dense(
                units,
                activation=TUNED_ANN_PARAMETERS["activation"],
                kernel_regularizer=regularizer,
            )
        )
        if TUNED_ANN_PARAMETERS["batchnorm"]:
            layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Dropout(TUNED_ANN_PARAMETERS["dropout"]))

    layers.append(tf.keras.layers.Dense(5, activation="softmax"))
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(TUNED_ANN_PARAMETERS["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
