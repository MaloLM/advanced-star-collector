import logging
import os
import random
import tensorflow as tf
from ..utils.game_states import DOWN_LEFT, DOWN_RIGHT, UP, RIGHT, DOWN, LEFT, UP_LEFT, UP_RIGHT
from ..settings import ACTION_POSSIBILITIES, BATCH_SIZE, BUFFER_MAX_LEN, DISCOUNT_FACTOR, \
    LEARNING_RATE, MODELS_PATH, STATE_SIZE

from ..utils.replay_buffer import ReplayBuffer
from .dqn_network import DQNNetwork
from ..utils.common import flatten_list

app_logger = logging.getLogger('app_logger')


class DQNAgent:

    def __init__(self, state_size: int = STATE_SIZE, action_size: int = ACTION_POSSIBILITIES):
        self.models_saving_path = MODELS_PATH
        self.modelname = None
        self.action_size = action_size
        self.model = DQNNetwork(state_size, action_size)
        self.buffer = ReplayBuffer(buffer_size=BUFFER_MAX_LEN)
        self.optimizer = tf.keras.optimizers.legacy.Adam(
            learning_rate=LEARNING_RATE)
        self.batch_size = BATCH_SIZE
        self.gamma = DISCOUNT_FACTOR

        # logging metrics
        self.current_loss = 0
        self.current_grad_norm = 0
        self.current_reward = 0

    def get_model_path(self, modelname):

        if not os.path.exists(self.models_saving_path):
            raise FileNotFoundError(
                f"The specified path '{self.models_saving_path}' does not exist.")

        full_path = os.path.join(self.models_saving_path, modelname)

        return full_path

    def is_model_saved(self, modelname):
        model_path = self.get_model_path(modelname)
        if os.path.exists(model_path):
            return True
        else:
            return False

    def choose_random_action(self):
        actions = [UP, UP_RIGHT, RIGHT, DOWN_RIGHT,
                   DOWN, DOWN_LEFT, LEFT, UP_LEFT]
        return random.choice(actions)

    def choose_action_with_model(self, state, modelname):
        if not hasattr(self, '_loaded_model'):
            model_path = self.get_model_path(modelname)
            if os.path.exists(model_path):
                self._loaded_model = tf.keras.models.load_model(model_path)
            else:
                raise FileNotFoundError('Model not found')

        flattened_state = flatten_list(state)

        state_tensor = tf.convert_to_tensor(
            [flattened_state], dtype=tf.float32)
        q_values = self._loaded_model(state_tensor)[0]
        action = int(tf.argmax(q_values).numpy())
        return action

    def choose_action_for_training(self, state: list, epsilon: float) -> int:

        flattened_state = flatten_list(state)
        print(flattened_state)

        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state_tensor = tf.convert_to_tensor(
                [flattened_state], dtype=tf.float32)
            q_values = self.model(state_tensor)[0]

            action = int(tf.argmax(q_values).numpy())

            return action

    def update_policy(self):
        if len(self.buffer) >= self.batch_size:
            minibatch = self.buffer.sample(self.batch_size)

            for (prev_state, action, reward, next_state, done, total_game_reward) in minibatch:
                flattened_prev_state = flatten_list(prev_state)
                flattened_next_state = flatten_list(next_state)

                if not done:
                    next_state_tensor = tf.convert_to_tensor(
                        [flattened_next_state], dtype=tf.float32)

                    target = reward + self.gamma * \
                        tf.reduce_max(self.model(next_state_tensor)[0])
                else:
                    target = total_game_reward

                with tf.GradientTape() as tape:
                    state_tensor = tf.convert_to_tensor(
                        [flattened_prev_state], dtype=tf.float32)
                    q_values = self.model(state_tensor)[0]

                    # Ensure both tensors are at least 1D
                    target_tensor = tf.expand_dims(target, axis=0)
                    q_value_tensor = tf.expand_dims(q_values[action], axis=0)

                    # Mean square error computation
                    loss = tf.keras.losses.MSE(target_tensor, q_value_tensor)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                grad_norm = tf.linalg.global_norm(gradients)
                self.optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables))

            self.current_loss = loss
            self.current_grad_norm = grad_norm
            self.current_reward = reward

    def save_model(self, modelname):
        model_path = self.get_model_path(modelname)

        try:
            self.model.save(model_path)
            app_logger.info(f'Model saved at {model_path}.')
            return True
        except Exception as error:
            app_logger.error(f'Saving model failed: {error}')
            return False
