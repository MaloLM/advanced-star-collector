import os
import tensorflow as tf
from datetime import datetime
from ..settings import TENSORFLOW_LOG_PATH


class TensorFlowLogger():

    def __init__(self, mode: str = 'UNSET') -> None:
        self.mode = mode
        self.step_count = 0

    def set_tensorflow_logger(self, mode: str):
        self.mode = mode
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir_name = f"{current_time}_{self.mode}"

        if not os.path.exists(TENSORFLOW_LOG_PATH):
            os.makedirs(TENSORFLOW_LOG_PATH)

        self.logs_path = os.path.join(TENSORFLOW_LOG_PATH, log_dir_name)

        self.summary_writer = tf.summary.create_file_writer(self.logs_path)

    def log(self, metrics: dict) -> None:
        with self.summary_writer.as_default():
            for key, value in metrics.items():
                tf.summary.scalar(
                    key, value, step=self.step_count)
            self.summary_writer.flush()
