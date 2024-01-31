import logging
from typing import Callable
from api.requests import get_action, update_model
from utils.replay_buffer import ReplayBuffer
from utils.timer import Timer
from world.world import World

from .game_state import GameState
from .reward import get_step_reward
from settings import MAX_STEP_PER_EP
from logger.data_recorder import create_gif

from utils.common import normalize_group, one_hot_encode
from utils.game_states import OUT_OF_BOUNDS, ON_EXIT_DOOR, TESTING, TRAINING

app_logger = logging.getLogger('app_logger')


class Episode:
    """
    Equivalent to a game
    """

    def __init__(self, ep_number: int, interface_update_callback: Callable):
        self.ep_number: int = ep_number
        self.world = World()
        self.buffer = ReplayBuffer()
        self.game_state = GameState(self.world)
        self.step_index = 0
        self.total_reward = 0
        self.steps_reward = []
        self.max_step_count: int = MAX_STEP_PER_EP
        # ----- metrics
        self.timer = Timer()
        self.total_reward_history = []
        self.total_collected_history = []
        # ---- callback
        self.interface_update_callback = interface_update_callback

    def __del__(self):
        app_logger.info(
            f'episode: {self.ep_number}, duration: {self.timer.get_formatted_duration()}')
        # create_gif()

    def process_game(self, mode: str = TRAINING) -> None:
        self.timer.start()
        state = self.reset(self.world)
        done = self.is_game_over()
        self.step_index += 1
        self.interface_update_callback()
        self.log_ml_metrics()

        while not done:

            if self.step_index >= 200:
                done = True

            state_to_choose_an_action = self.prepare_state_for_model(state)

            action = get_action(state_to_choose_an_action, mode)

            new_state, reward, done = self.step(action)
            self.step_index += 1

            if mode == TRAINING:

                next_state = self.prepare_state_for_model(new_state)

                self.save_to_buffer(
                    state_to_choose_an_action, action, reward, next_state, done)

            state = new_state

            self.interface_update_callback()
            self.log_ml_metrics()

        if mode == TRAINING:
            update_model(self.buffer)

        self.total_reward_history.append(self.total_reward)
        self.total_collected_history.append(
            self.game_state.nb_collected)

        self.timer.end()

    def prepare_state_for_model(self, state: list):
        distance_when_nothing = self.world.surface.shape.radius * 4
        next_s = one_hot_encode(state[0][2], 4)  # len 4
        star_prop = state[0][0]  # len 1
        door_found = state[0][1]  # len 1
        step_count = self.step_index / self.max_step_count  # len 1
        vision = state[1]  # len 8
        sensing_state = state[2]  # len 8
        sensing_distances = normalize_group(
            state[3], 0, distance_when_nothing)  # len 8

        prapared_state = [next_s, star_prop, door_found, step_count,
                          vision, sensing_state, sensing_distances,]
        return prapared_state

    def save_to_buffer(self, state_to_choose_an_action, action, reward, next_state, done):
        self.buffer.add(
            (state_to_choose_an_action, action, reward, next_state, done), round(self.total_reward, 3))

    def update_game_state_after_action(self, action: int) -> tuple[list, list]:
        """
        Update the game state after performing an action.

        Args:
            action (int): The rotation action performed by the agent.

        Returns:
            list: The new state of the game after the action.
        """
        self.world.move_agent(action)

        agent_current_state = self.world.evaluate_current_positions_status()
        self.game_state.update_current_state(agent_current_state)

        self.game_state.next_states = self.game_state.evaluate_next_states()

        nb_collected = self.game_state.num_collectibles - \
            len(self.world.collectibles)

        self.game_state.update_collectibles_status(nb_collected)
        self.game_state.nb_collected = nb_collected

        return self.game_state.get_state()

    def is_game_over(self) -> bool:
        """
        Determine if the game is over based on the current state.

        Returns:
            bool: True if the game is over, False otherwise.
        """
        is_out_of_bounds = self.game_state.current_state == OUT_OF_BOUNDS
        is_exit_door_found = self.game_state.current_state == ON_EXIT_DOOR

        return is_out_of_bounds or is_exit_door_found

    def step(self, action: int) -> tuple[tuple[list, list], float, bool]:
        """
        Perform a game step given an action.

        Args:
            action (int): The action to be performed in this step.

        Returns:
            tuple: A tuple containing the new state, step reward, and a flag indicating if the game is over.
        """
        new_state = self.update_game_state_after_action(action)

        self.world.agent.head_detection, self.world.agent.head_distance_to_a_collectible = self.world.get_agent_direction_sensing()

        step_reward = get_step_reward(self.step_index, self.game_state.current_state,
                                      self.game_state.nb_collected, self.game_state.num_collectibles)

        self.total_reward += step_reward
        self.steps_reward.append(step_reward)

        done = self.is_game_over()

        return new_state, step_reward, done

    def reset(self, new_world: World) -> tuple[list, list]:
        """
        Reset the game environment for a new episode.

        Args:
            new_world (World): The new world to be used for the next episode.

        Returns:
            list: The initial state of the game after resetting.
        """
        self.steps_reward = []
        self.total_reward = 0
        self.world = new_world
        self.world.reset()
        self.game_state.reset(self.world)

        self.world.agent.head_detection, self.world.agent.head_distance_to_a_collectible = self.world.get_agent_direction_sensing()

        return self.game_state.get_state()

    def get_current_state(self) -> dict[str, dict]:
        props = {
            "agent": {},
            "exit-door": {},
            "collectibles": {},
        }

        agent = self.world.agent
        exit_door = self.world.exit_door

        props["surface"] = {}
        props["surface"]["x"] = self.world.surface.x_pos
        props["surface"]["y"] = self.world.surface.y_pos
        props["surface"]["radius"] = self.world.surface.shape.radius
        props["surface"]["color"] = self.world.surface.shape.color

        props["agent"]["radius"] = agent.shape.radius
        props["agent"]["color"] = agent.shape.color
        props["agent"]["x"] = agent.x_pos
        props["agent"]["y"] = agent.y_pos

        props["agent"]["heads"] = {}
        for idx, head in enumerate(agent.heads):
            props["agent"]["heads"][idx] = {}
            props["agent"]["heads"][idx]["color"] = head.sensing_color
            props["agent"]["heads"][idx]["angle"] = head.angle
            props["agent"]["heads"][idx]["radius"] = head.radius
            props["agent"]["heads"][idx]["distance_to_center"] = head.distance_to_center
            x, y = head.get_seen_position(
                (self.world.agent.x_pos, self.world.agent.x_pos))
            props["agent"]["heads"][idx]["x"] = x
            props["agent"]["heads"][idx]["y"] = y
            props["agent"]["heads"][idx]["is_within_surface"] = head.is_within_surface

            props["agent"]["heads"][idx]["intersection_with_circle_x"] = head.intersection_with_circle_pos[0]
            props["agent"]["heads"][idx]["intersection_with_circle_y"] = head.intersection_with_circle_pos[1]

        props["exit-door"]["radius"] = exit_door.shape.radius
        props["exit-door"]["color"] = exit_door.color
        props["exit-door"]["x"] = exit_door.x_pos
        props["exit-door"]["y"] = exit_door.y_pos

        for idx, collectible in enumerate(self.world.collectibles):
            props["collectibles"][idx] = {}
            props["collectibles"][idx]["radius"] = collectible.shape.radius
            props["collectibles"][idx]["color"] = collectible.color
            props["collectibles"][idx]["x"] = collectible.x_pos
            props["collectibles"][idx]["y"] = collectible.y_pos

        return props  # to draw

    def log_ml_metrics(self) -> None:
        app_logger.info(
            f'episode: {self.ep_number}, \
                current agent situation: {self.game_state.current_state}, \
                current vision: {self.game_state.evaluate_next_states()}, \
                current collection {self.game_state.nb_collected}/{self.game_state.num_collectibles} \
                ended ? {self.game_state.current_state in [OUT_OF_BOUNDS, ON_EXIT_DOOR]}'
        )

    def get_info(self) -> dict[str, int]:
        info = {
            "Frame": self.step_index,
        }
        return info
