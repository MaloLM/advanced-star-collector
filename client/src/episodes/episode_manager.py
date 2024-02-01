import logging
from time import sleep
from api.requests import end_training
from utils.timer import Timer
from typing import Any, Callable, Optional
from .episode import Episode
from logger.logging import setup_loggers
from utils.game_states import ON_EXIT_DOOR, OUT_OF_BOUNDS, RANDOM, TESTING, TRAINING
from settings import EPSILON, EPSILON_DECAY, MIN_EPSILON, NB_OF_EPISODES


setup_loggers()
app_logger = logging.getLogger('app_logger')


class EpisodeManager:
    def __init__(self, nb_eps: int = NB_OF_EPISODES):
        self.mode = TRAINING
        self.callback: Optional[Callable] = None
        self.nb_episodes = nb_eps
        self.current_running_ep_idx = -1
        self.current_episode: Optional[Episode] = None
        # personnal epsilon
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.min_epsilon = MIN_EPSILON
        # --- logging
        self.timer = Timer()
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

    def train_model(self) -> None:

        app_logger.info('TRAINING: Start of a training')
        self.set_mode(TRAINING)
        self.timer.start()

        for idx in range(1, self.nb_episodes + 1):
            self.current_running_ep_idx = idx
            self.current_episode = Episode(
                idx, self.interface_update_callback, self.epsilon)
            self.decay_exploration_rate()
            self.current_episode.process_game(mode=TRAINING)
            self.cummulative_exit_doors += 1 if self.current_episode.game_state.current_state == ON_EXIT_DOOR else 0
            self.cummulative_out_of_bouds += 1 if self.current_episode.game_state.current_state == OUT_OF_BOUNDS else 0
            sleep(1.2)
        self.timer.end()
        app_logger.info(
            f'TRAINING: End of the training, duration: {self.timer.get_formatted_duration()}')

    def run_model(self) -> None:
        app_logger.info('TESTING: Start of inference')
        self.set_mode(TESTING)
        self.timer.start()

        for idx in range(1, self.nb_episodes + 1):
            self.current_running_ep_idx = idx
            self.current_episode = Episode(
                idx, self.agent, self.interface_update_callback)
            self.current_episode.process_game(mode=TESTING)
            self.cummulative_exit_doors += 1 if self.current_episode.game_state.current_state == ON_EXIT_DOOR else 0
            self.cummulative_out_of_bouds += 1 if self.current_episode.game_state.current_state == OUT_OF_BOUNDS else 0

        self.timer.end()
        app_logger.info(
            f'TESTING: End of inference, duration: {self.timer.get_formatted_duration()}')

    def run_random(self) -> None:
        app_logger.info('RANDOM: Start of random play')
        self.set_mode(RANDOM)
        self.timer.start()

        for idx in range(1, self.nb_episodes + 1):
            self.current_running_ep_idx = idx
            self.current_episode = Episode(
                idx, self.interface_update_callback, self.epsilon)
            self.current_episode.process_game(mode=RANDOM)
            self.cummulative_exit_doors += 1 if self.current_episode.game_state.current_state == ON_EXIT_DOOR else 0
            self.cummulative_out_of_bouds += 1 if self.current_episode.game_state.current_state == OUT_OF_BOUNDS else 0

        self.timer.end()
        app_logger.info(
            f'RANDOM: End of random play, duration: {self.timer.get_formatted_duration()}')

    def decay_exploration_rate(self):
        """
        Update the exploration rate (epsilon).

        This gradually reduces the rate of random action selection to favor exploitation over exploration.
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def get_current_state_to_display(self) -> dict[str, dict]:
        current_episode: Episode = self.current_episode
        props = current_episode.get_current_state()
        return props

    def get_episode_info(self) -> dict[str, Any]:

        info = {
            "Mode": self.mode,
            "Duration": self.timer.get_formatted_duration(),
            "Episode": f'{self.current_running_ep_idx}/{self.nb_episodes}',
            "Epsilon":  round(self.epsilon, 3)
        }

        episode_details = self.current_episode.get_info()

        for index in episode_details:
            info[index] = episode_details[index]

        return info
