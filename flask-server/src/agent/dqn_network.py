import tensorflow as tf


class DQNNetwork(tf.keras.Model):
    """
    A Deep Q-Network (DQN) model using TensorFlow Keras.

    This model is a fully connected neural network used for reinforcement learning,
    specifically in the context of Q-learning where the goal is to learn the value of
    actions in given states of the environment.

    Attributes:
        input_layer (tf.keras.layers.InputLayer): The input layer of the network.
        dense1 (tf.keras.layers.Dense): The first dense layer.
        dense2 (tf.keras.layers.Dense): The second dense layer.
        output_layer (tf.keras.layers.Dense): The output layer that predicts Q-values for each action.

    Args:
        state_size (int): The size of the state space of the environment.
        action_size (int): The size of the action space of the environment.
    """

    def __init__(self, state_size: int, action_size: int):
        super(DQNNetwork, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(
            input_shape=(state_size,))

        self.normalization_layer = tf.keras.layers.BatchNormalization()

        self.dense1 = tf.keras.layers.Dense(
            128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout1 = tf.keras.layers.Dropout(0.3)

        # self.dense3 = tf.keras.layers.Dense(
        #     64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.007))
        # self.dropout3 = tf.keras.layers.Dropout(0.3)

        self.dense4 = tf.keras.layers.Dense(
            256, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.005))
        self.dropout4 = tf.keras.layers.Dropout(0.3)

        self.dense5 = tf.keras.layers.Dense(
            128, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.003))
        self.dropout5 = tf.keras.layers.Dropout(0.3)

        self.dense6 = tf.keras.layers.Dense(
            64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.003))
        self.dropout6 = tf.keras.layers.Dropout(0.3)

        self.dense7 = tf.keras.layers.Dense(
            32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))

        self.output_layer = tf.keras.layers.Dense(
            action_size, activation='linear')

    def call(self, inputs):
        """
        Forward pass through the network.

        This method defines the computation from inputs to outputs (Q-values).

        Args:
            inputs (Tensor): The input state tensor, typically representing the state of the environment.

        Returns:
            Tensor: The output tensor containing Q-values for each action.
        """
        x = self.input_layer(inputs)

        x = self.normalization_layer(x)

        x = self.dense1(x)
        x = self.dropout1(x)

        # x = self.dense3(x)
        # x = self.dropout3(x)

        x = self.dense4(x)
        x = self.dropout4(x)

        x = self.dense5(x)
        x = self.dropout5(x)

        x = self.dense6(x)
        x = self.dropout6(x)

        x = self.dense7(x)

        return self.output_layer(x)
