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
        print("INSIDE UPDATE", file=sys.stdout)
        for experience in experiences:
            self.agent.buffer.add(experience)
        print("BEFORE UPDATE POLICY \n", file=sys.stdout)
        self.agent.update_policy()

    def update_experience_replay(self, experiences: list, episode_failed: bool):
        if episode_failed == True:
            if self.nb_failed_ep_count/self.nb_suceeded_ep_count <= self.target_prop:
                self.nb_failed_ep_count += 1
                self.update_queue.put(experiences)
        else:  # success
            if self.nb_failed_ep_count/self.nb_suceeded_ep_count >= self.target_prop:
                self.nb_suceeded_ep_count += 1
                self.update_queue.put(experiences)
