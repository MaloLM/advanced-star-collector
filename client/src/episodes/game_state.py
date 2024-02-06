from settings import MAX_STEP_PER_EP
from utils.common import normalize_group, one_hot_encode
from world.world import World
from utils.game_states import ONTO_SURFACE


class GameState:
    """
    UPDATE ME
    Represents the current state of the game environment.

    This class keeps track of various game state variables such as the number of collectibles,
    the status of the exit door, the agent's orientation, and the next state representation.

    Attributes:
        num_collectibles (int): The total number of collectibles in the world.
        nb_collected (int): The number of collectibles that have been collected.
        exit_door_found (int): Indicates whether the exit door has been found (1) or not (0).
        orientation (int): The current orientation of the agent in degrees.
        next_state (list): The representation of the next state, encoded as a one-hot vector.
    """

    def __init__(self, world: World):
        self.world: World = world
        self.current_state: int = ONTO_SURFACE
        self.nb_collected: int = 0
        self.step_index = 0
        self.max_step_count: int = MAX_STEP_PER_EP
        self.num_collectibles: int = len(self.world.collectibles)
        self.next_states: list = [0] * len(self.world.agent.heads)

    def evaluate_next_states(self) -> list:
        agent_next_status = []

        for idx, _ in enumerate(self.world.agent.heads):
            status = self.world.evaluate_next_position_status(idx)
            agent_next_status.append(status)

        return agent_next_status

    def update_current_state(self, state_index: int) -> None:
        if 0 <= state_index <= 3:
            self.current_state = state_index

    def update_collectibles_status(self, nb_collected: int) -> None:
        """
        Update the number of collected collectibles.

        Args:
            nb_collected (int): The updated number of collected collectibles.
        """
        if 0 <= nb_collected <= self.num_collectibles:
            self.nb_collected = nb_collected

    def get_state(self):
        default_distance = self.world.surface.shape.radius * 2

        normd_collection_progress = (self.nb_collected / self.num_collectibles
                                     if self.num_collectibles > 0 else 0)
        num_remaining_stars = self.num_collectibles - self.nb_collected

        head_detection, head_distance_to_collectible = self.world.get_agent_direction_sensing()

        normd_next_states = [one_hot_encode(
            value, 4) for value in self.next_states]

        normd_head_detection = [one_hot_encode(
            value, 3) for value in head_detection]

        normd_head_distances_to_collectible = normalize_group(head_distance_to_collectible, 0,
                                                              default_distance)

        prepared_state = [
            # one-hot encoded current agent position
            one_hot_encode(self.current_state, 4),
            normd_collection_progress,  # normalized collection progress
            num_remaining_stars,  # number of remaining stars
            self.world.agent.door_found,  # is door found ?
            self.step_index / self.max_step_count,  # normalized step count
            normd_next_states,  # next state (short vision)
            normd_head_detection,  # directionnal sensing
            normd_head_distances_to_collectible  # normalized direction sensing distances
        ]

        return prepared_state
