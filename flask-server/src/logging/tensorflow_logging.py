import os
import tensorflow as tf
from datetime import datetime
from utils.settings import TENSORFLOW_LOG_PATH


class TensorFlowLogger():

    def __init__(self, mode: str = 'UNSET') -> None:
        self.mode = mode

    def set_tensorflow_logger(self, mode: str):
        self.mode = mode
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir_name = f"{current_time}_{self.mode}"

        if not os.path.exists(TENSORFLOW_LOG_PATH):
            os.makedirs(TENSORFLOW_LOG_PATH)

        self.logs_path = os.path.join(TENSORFLOW_LOG_PATH, log_dir_name)

        self.summary_writer = tf.summary.create_file_writer(self.logs_path)

    def log(self, step_count, metrics: dict) -> None:
        with self.summary_writer.as_default():
            for key, value in metrics.items():
                tf.summary.scalar(
                    key, value, step=step_count)
            self.summary_writer.flush()

# def log_episode_to_tensorboard(self) -> None:
#     metrics = {
#         "Cumulative OOBounds/ExitDoor": (self.cummulative_out_of_bouds + 1) / (self.cummulative_exit_doors + 1),
#         "Episode number of step": self.current_episode.step_index,
#         "Loss": dqn.current_loss,
#         "Gradient norm": dqn.current_grad_norm,
#         "Total episode reward": self.current_episode.total_reward,
#     }
#     self.tensorflow_logger.log(self.current_running_ep_idx, metrics)
