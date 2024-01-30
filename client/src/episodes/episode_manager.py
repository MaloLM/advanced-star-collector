import logging
import os
# from data_management.tensorflow_logging import TensorFlowLogger
from utils.timer import Timer
from typing import Any, Callable
from episode import Episode
from utils.logging import setup_loggers
from utils.game_states import ON_EXIT_DOOR, OUT_OF_BOUNDS, RANDOM, TESTING, TRAINING
from config.settings import MODELS_PATH, NB_OF_EPISODES, TENSORFLOW_LOG_PATH

setup_loggers()
app_logger = logging.getLogger('app_logger')


class EpisodeManager:
    def __init__(self, nb_eps: int = NB_OF_EPISODES):
        self.mode = TRAINING
        self.callback = None
        self.nb_episodes = nb_eps
        self.current_running_ep_idx = -1
        self.current_episode: Episode = None
        self.agent = DQNAgent()

        # --- logging
        self.timer = Timer()
        # self.tensorflow_logger = TensorFlowLogger()
        self.cummulative_exit_doors = 0
        self.cummulative_out_of_bouds = 0

    def set_mode(self, mode):
        if mode in [TRAINING, TESTING, RANDOM]:
            self.mode = mode
        else:
            raise ValueError(
                "Mode must be 'TRAINING' or 'TESTING' or 'RANDOM'")

    def set_callback(self, callback: Callable) -> None:
        self.interface_update_callback: Callable = callback

    def log_episode_to_tensorboard(self) -> None:
        dqn: DQNAgent = self.agent

        metrics = {
            "Cumulative OOBounds/ExitDoor": (self.cummulative_out_of_bouds + 1) / (self.cummulative_exit_doors + 1),
            "Episode number of step": self.current_episode.step_index,
            "Loss": dqn.current_loss,
            "Gradient norm": dqn.current_grad_norm,
            "Total episode reward": self.current_episode.total_reward,
        }

        self.tensorflow_logger.log(self.current_running_ep_idx, metrics)

    def train_model(self, model_name: str) -> None:

        app_logger.info('TRAINING: Start of a training')
        self.set_mode(TRAINING)
        # self.tensorflow_logger.set_tensorflow_logger(self.mode)
        self.timer.start()

        for idx in range(1, self.nb_episodes + 1):
            self.current_running_ep_idx = idx
            self.current_episode = Episode(
                idx, self.agent, self.interface_update_callback)
            self.current_episode.process_game(mode=TRAINING)
            self.cummulative_exit_doors += 1 if self.current_episode.game_state.current_state == ON_EXIT_DOOR else 0
            self.cummulative_out_of_bouds += 1 if self.current_episode.game_state.current_state == OUT_OF_BOUNDS else 0

            self.log_episode_to_tensorboard()

        self.timer.end()
        self.agent.save_model(model_name)
        app_logger.info(
            f'TRAINING: End of the training, duration: {self.timer.get_formatted_duration}')

    def run_model(self, model_name: str, models_path: str = MODELS_PATH) -> None:
        app_logger.info('TESTING: Start of inference')
        self.set_mode(TESTING)
        # self.tensorflow_logger.set_tensorflow_logger(self.mode)
        self.timer.start()

        self.agent.model_path = os.path.join(models_path, model_name)

        for idx in range(1, self.nb_episodes + 1):
            self.current_running_ep_idx = idx
            self.current_episode = Episode(
                idx, self.agent, self.interface_update_callback)
            self.current_episode.process_game(mode=TESTING)
            self.cummulative_exit_doors += 1 if self.current_episode.game_state.current_state == ON_EXIT_DOOR else 0
            self.cummulative_out_of_bouds += 1 if self.current_episode.game_state.current_state == OUT_OF_BOUNDS else 0

            self.log_episode_to_tensorboard()

        self.timer.end()
        app_logger.info(
            f'TESTING: End of inference, duration: {self.timer.get_formatted_duration}')

    def run_random(self) -> None:
        app_logger.info('RANDOM: Start of random play')
        self.set_mode(RANDOM)
        # self.tensorflow_logger.set_tensorflow_logger(self.mode)
        self.timer.start()

        for idx in range(1, self.nb_episodes + 1):
            self.current_running_ep_idx = idx
            self.current_episode = Episode(
                idx, self.agent, self.interface_update_callback)
            self.current_episode.process_game(mode=RANDOM)
            self.cummulative_exit_doors += 1 if self.current_episode.game_state.current_state == ON_EXIT_DOOR else 0
            self.cummulative_out_of_bouds += 1 if self.current_episode.game_state.current_state == OUT_OF_BOUNDS else 0

            self.log_episode_to_tensorboard()

        self.timer.end()
        app_logger.info(
            f'RANDOM: End of random play, duration: {self.timer.get_formatted_duration}')

    def get_current_state_to_display(self) -> dict[str, dict]:
        current_episode: Episode = self.current_episode
        props = current_episode.get_current_state()
        return props

    def get_episode_info(self) -> dict[str, Any]:

        info = {
            "Mode": self.mode,
            "Duration": self.timer.get_formatted_duration(),
            "Episode": f'{self.current_running_ep_idx + 1}/{self.nb_episodes}',
            "Epsilon":  round(self.agent.epsilon, 3)
        }

        episode_details = self.current_episode.get_info()

        for index in episode_details:
            info[index] = episode_details[index]

        return info
