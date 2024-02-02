import queue
import sys
from threading import Thread
from .agent_manager import DQNAgentManager
from ..utils.game_states import TRAINING
from ..logger.tensorflow_logging import TensorFlowLogger


class ModelUpdaterThread:
    def __init__(self, agent_manager: DQNAgentManager):
        self.agent_manager = agent_manager
        self.thread = None
        self.is_running = False
        self.training_finished = False
        self.tf_logger = TensorFlowLogger()
        self.tf_logger.set_tensorflow_logger(TRAINING)

    def start(self):
        def run():
            while True:
                try:
                    data = self.agent_manager.update_queue.get(timeout=1)
                    self.agent_manager.update_agent(data)
                    self.tf_log()
                except queue.Empty:
                    if self.training_finished:
                        self.is_running = False
                        break

            self.agent_manager.agent.save_model()

        self.thread = Thread(target=run, daemon=True)
        self.thread.start()
        self.is_running = True

    def stop(self):
        self.training_finished = True

    def check_and_restart_thread(self):
        if not self.is_running:
            self.start()

    def tf_log(self):
        metrics = {
            "Queue size": self.agent_manager.update_queue.qsize(),
            "Buffer size": len(self.agent_manager.agent.buffer),
            "Gradient norm": self.agent_manager.agent.current_grad_norm,
            "Loss": self.agent_manager.agent.current_loss,
            "Fail/success proportion inside experience pool": self.agent_manager.nb_failed_ep_count / self.agent_manager.nb_suceeded_ep_count
        }
        self.tf_logger.log(metrics)
        self.tf_logger.step_count += 1
