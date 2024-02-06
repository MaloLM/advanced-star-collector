import logging
from time import sleep
from api.requests import get_queue_size
from utils.common import epsilon_decay
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
        self.episode_timeout = 1
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
                idx, self.interface_update_callback, self.epsilon, self.mode)
            self.decay_exploration_rate()
            self.update_episode_timeout()
            self.current_episode.process_game()
            sleep(self.episode_timeout)
            self.update_state_counters()
        self.timer.end()
        app_logger.info(
            f'TRAINING: End of the training, duration: {self.timer.get_formatted_duration()}')

    def run_model(self, modelname) -> None:
        app_logger.info('TESTING: Start of inference')
        self.set_mode(TESTING)
        self.episode_timeout = 0
        self.timer.start()

        for idx in range(1, self.nb_episodes + 1):
            self.current_running_ep_idx = idx
            self.current_episode = Episode(
                idx, self.interface_update_callback, self.epsilon, self.mode)
            self.current_episode.modelname = modelname
            self.current_episode.process_game()
            self.update_state_counters()

        self.timer.end()
        app_logger.info(
            f'TESTING: End of inference, duration: {self.timer.get_formatted_duration()}')

    def run_random(self) -> None:
        app_logger.info('RANDOM: Start of random play')
        self.set_mode(RANDOM)
        self.episode_timeout = 0
        self.timer.start()

        for idx in range(1, self.nb_episodes + 1):
            self.current_running_ep_idx = idx
            self.current_episode = Episode(
                idx, self.interface_update_callback, self.epsilon, self.mode)
            self.current_episode.process_game()
            self.update_state_counters()

        self.timer.end()
        app_logger.info(
            f'RANDOM: End of random play, duration: {self.timer.get_formatted_duration()}')

    def update_episode_timeout(self):
        queue_size = get_queue_size()
        self.episode_timeout = queue_size / 6

    def decay_exploration_rate(self):
        """
        Update the exploration rate (epsilon).

        This gradually reduces the rate of random action selection to favor exploitation over exploration.
        """
        self.epsilon = epsilon_decay(
            self.current_running_ep_idx, self.nb_episodes)

    def update_state_counters(self):
        self.cummulative_exit_doors += 1 if self.current_episode.game_state.current_state == ON_EXIT_DOOR else 0
        self.cummulative_out_of_bouds += 1 if self.current_episode.game_state.current_state == OUT_OF_BOUNDS else 0

    def get_current_state_to_display(self) -> dict[str, dict]:
        current_episode: Episode = self.current_episode
        props = current_episode.get_current_state()
        return props

    def get_episode_info(self) -> dict[str, Any]:

        info = {
            "Mode": self.mode,
            "Duration": self.timer.get_formatted_duration(),
            "Episode": f'{self.current_running_ep_idx}/{self.nb_episodes}',
            "Epsilon":  round(self.epsilon, 3),
            "Timeout": f'{round(self.episode_timeout, 2)}s'
        }

        episode_details = self.current_episode.get_info()

        for index in episode_details:
            info[index] = episode_details[index]

        return info
