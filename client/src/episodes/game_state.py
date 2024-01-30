from rl_module.world.world import World
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
        self.num_collectibles: int = len(self.world.collectibles)
        self.next_states: list = [0] * len(self.world.agent.heads)

    def reset(self, world: World) -> None:
        """
        Reset the game state to the initial state based on the given world.

        Args:
            world (World): An instance of the World class to reset the game state according to.
        """
        self.world = world
        self.num_collectibles = len(self.world.collectibles)
        self.nb_collected = 0

        current_state = world.evaluate_current_positions_status()
        self.update_current_state(current_state)

        self.next_states = self.evaluate_next_states()

    def evaluate_next_states(self) -> list:
        agent_next_status = []

        for idx, _ in enumerate(self.world.agent.heads):
            status = self.world.evaluate_next_position_status(idx)
            agent_next_status.append(status)

        return agent_next_status

    def update_current_state(self, state_index: int) -> None:
        if 0 <= state_index <= len(self.world.agent.heads) - 1:
            self.current_state = state_index

    def update_collectibles_status(self, nb_collected: int) -> None:
        """
        Update the number of collected collectibles.

        Args:
            nb_collected (int): The updated number of collected collectibles.
        """
        if 0 <= nb_collected < self.num_collectibles:
            self.nb_collected = nb_collected

    def get_state(self) -> tuple[list, list]:
        """
        Return the current state of the game.

        Combines various state components such as collection progress, exit door status,
        orientation, and next state representation into a single state vector.

        Returns:
            list: The current state of the game as a normalized list.
        """
        normalized_collection_progress = self.nb_collected / \
            self.num_collectibles if self.num_collectibles > 0 else 0

        return [normalized_collection_progress, self.world.agent.door_found, self.current_state], \
            self.next_states, self.world.agent.head_detection, self.world.agent.head_distance_to_a_collectible
