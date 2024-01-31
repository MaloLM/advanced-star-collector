import queue
from .dqn_agent import DQNAgent


class DQNAgentManager:
    def __init__(self):
        self.agent = DQNAgent()
        self.update_queue = queue.Queue()

    def reset_agent(self, modelname: str):
        self.agent = DQNAgent()
        self.update_queue = queue.Queue()
        self.agent.modelname = modelname

    def update_agent(self, experiences):
        for experience in experiences:
            self.agent.buffer.add(experience)
        self.agent.update_policy()
        self.agent.decay_exploration_rate()
