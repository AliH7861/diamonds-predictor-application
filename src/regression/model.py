"""Model builders for diamond price regression."""


def build_ann(input_dim: int, seed: int = 42):
    """Build the course-required price ANN with a single nonnegative output."""
    import tensorflow as tf

    # Clear any existing TensorFlow sessions and set the random seed for reproducibility, then define a sequential neural network model with input, hidden, and output layers, compile it with the Adam optimizer and mean squared error loss, and return the compiled model.
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    # Define a Sequential Neural Network Model with Input, Hidden, and Output Layers, Using ReLU Activation for Hidden Layers, Batch Normalization, Dropout for Regularization, and a Single Output Neuron for Price Prediction
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(1),
        ]
    )
    
    # Compile the Model with the Adam Optimizer, Mean Squared Error Loss, and Mean Absolute Error Metric for Evaluation
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse", metrics=["mae"])
    return model
