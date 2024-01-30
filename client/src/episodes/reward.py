
import math
from utils.game_states import ON_EXIT_DOOR, ONTO_SURFACE, OUT_OF_BOUNDS, STAR_COLLECTED


MAX_STEP = 100  # until reward is null
ATTENUATION_FACTOR = 0.8


@staticmethod
def get_step_reward(step_number: int, agent_situation: int, nb_collected: int, nb_collectibles: int) -> float:
    """
        Calculate the reward for a given step based on the agent's situation and collectible items.

        Args:
            step_number (int): The current step number in the game.
            agent_situation (int): The current situation of the agent (e.g., on surface, collected star, etc.).
            nb_collected (int): The number of collectibles gathered by the agent so far.
            nb_collectibles (int): The total number of collectibles available.

        Returns:
            float: The calculated reward for the given step.

        The reward is calculated based on the agent's situation. Penalties are given for being out of bounds,
        small rewards for progressing without finding anything, and larger rewards for collecting stars and
        reaching the exit door. The reward is further adjusted based on the number of collected items to 
        encourage collection efficiency.
        """
    reward = 0

    collect_factor = nb_collected / \
        nb_collectibles + \
        0.01  # hardcore compensation to avoid collect_factor to be 0

    if agent_situation == OUT_OF_BOUNDS:
        reward = -50 * (1/collect_factor)

    if agent_situation == ONTO_SURFACE:
        # Very small reward when progressing in the surface without finding anything
        reward += 0.05

    if agent_situation == STAR_COLLECTED:
        reward += 20
        reward = attenuate_reward(
            step_number, reward)

    if agent_situation == ON_EXIT_DOOR:

        reward = exit_reward(nb_collected)

        reward = attenuate_reward(
            step_number, reward)

    return reward


@staticmethod
def attenuate_reward(step: int, reward: float, max_steps: int = MAX_STEP, attenuation_factor: float = ATTENUATION_FACTOR):
    """
        Attenuate the reward based on the step number to encourage quicker completion.

        Args:
            step (int): The current step number in the game.
            reward (float): The current calculated reward before attenuation.
            max_steps (int, optional): The maximum number of steps for full attenuation. Defaults to 100.
            attenuation_factor (float, optional): The factor by which the reward is attenuated. Defaults to 0.8.

        Returns:
            float: The attenuated reward.

        This function reduces the reward as the number of steps increases to encourage the agent to complete
        the objectives more quickly. The attenuation is calculated as a fraction of the max_steps,
        multiplied by the attenuation_factor.
        """
    attenuation = (step / max_steps) * attenuation_factor
    return reward * (1 - attenuation)


@staticmethod
def exit_reward(star_proportion: float):
    """
    Calculate the reward for exiting based on the proportion of stars/collectible collected.

    This function computes the exit reward as an inversely proportional function
    of the star/collectible collection proportion. The reward increases sharply as the star /collectible
    proportion approaches 1, following a function with an asymptote at 
    star_proportion = 1.

    Args:
        star_proportion (float): The proportion of stars/collectible collected, 
                                 a value between 0 and 1.

    Returns:
        float: The calculated exit reward based on the star proportion.

    Note:
        The function uses the formula 1/((1 - (star_proportion - 0.01))**2) to 
        calculate the exit reward. It's important to note that the function 
        will have an asymptotic behavior as star_proportion approaches 1.
    """
    x = star_proportion - 0.01

    return (1/((1-x)**2))
