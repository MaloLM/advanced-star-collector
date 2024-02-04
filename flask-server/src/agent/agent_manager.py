import queue
import sys
from .dqn_agent import DQNAgent


class DQNAgentManager:
    def __init__(self):
        self.agent = DQNAgent()
        self.update_queue = queue.Queue()

        # fail/success episodes proportion control
        self.target_prop = 0.75
        self.nb_failed_ep_count = 0
        self.nb_suceeded_ep_count = 1  # to avoid 0 division

    def reset_agent(self, modelname: str):
        self.agent = DQNAgent()
        self.update_queue = queue.Queue()
        self.agent.modelname = modelname
        self.nb_failed_ep_count = 0
        self.nb_suceeded_ep_count = 1

    def update_agent(self, experiences):
        for experience in experiences:
            self.agent.buffer.add(experience)
        self.agent.update_policy()

    def update_experience_replay(self, experiences: list, episode_failed: bool, force_update: bool = False):
        if episode_failed:
            if self.nb_failed_ep_count/self.nb_suceeded_ep_count <= self.target_prop or force_update:
                self.nb_failed_ep_count += 1
                self.update_queue.put(experiences)
        else:  # success
            if self.nb_failed_ep_count/self.nb_suceeded_ep_count >= self.target_prop or force_update:
                self.nb_suceeded_ep_count += 1
                self.update_queue.put(experiences)
