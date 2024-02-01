import math
from utils.game_states import ON_EXIT_DOOR, ONTO_SURFACE, OUT_OF_BOUNDS, STAR_COLLECTED


MAX_STEP = 100  # until reward is null
ATTENUATION_FACTOR = 0.8


@staticmethod
def get_step_reward(step_number: int, agent_situation: int, nb_collected: int, nb_collectibles: int, current_total_reward: float) -> float:

    reward = 0

    collect_factor = nb_collected / \
        nb_collectibles + 0.01  # hardcore compensation to avoid collect_factor to be 0

    if agent_situation == OUT_OF_BOUNDS:
        reward = - current_total_reward

    if agent_situation == ONTO_SURFACE:
        # Very small reward when progressing in the surface without finding anything
        reward += 0.05

    if agent_situation == STAR_COLLECTED:
        reward += 20
        reward = attenuate_reward(
            step_number, reward)

    if agent_situation == ON_EXIT_DOOR:

        reward = exit_reward(collect_factor)

        reward = attenuate_reward(
            step_number, reward)

    return reward


@staticmethod
def attenuate_reward(step: int, reward: float):
    return math.exp(-step * 0.05) * reward


@staticmethod
def exit_reward(star_proportion: float):
    x = star_proportion + 0.01
    return (1/((1-x)**2))-1
