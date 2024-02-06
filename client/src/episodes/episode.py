import logging
from typing import Callable
from utils.replay_buffer import ReplayBuffer
from utils.timer import Timer
from world.world import World
from .game_state import GameState
from .reward import get_step_reward
from api.requests import get_action, update_model
from logger.data_recorder import create_gif
from utils.game_states import OUT_OF_BOUNDS, ON_EXIT_DOOR, RANDOM, TESTING, TRAINING, UNSET

app_logger = logging.getLogger('app_logger')


class Episode:
    """
    Equivalent to a game
    """

    def __init__(self, ep_number: int, interface_update_callback: Callable, epsilon: float = None, mode: str = UNSET):
        self.ep_number: int = ep_number
        self.world = World()
        self.buffer = ReplayBuffer()
        self.game_state = GameState(self.world)
        self.step_index = 0
        self.total_reward = 0
        self.ep_epsilon = epsilon
        self.steps_reward = []

        self.mode = mode
        self.modelname = None
        # ----- metrics
        self.timer = Timer()
        # ---- callback
        self.interface_update_callback = interface_update_callback

    def __del__(self):
        app_logger.info(
            f'episode: {self.ep_number}, duration: {self.timer.get_formatted_duration()}')
        if self.mode == TESTING:
            create_gif()

    def update_step_count(self):
        self.step_index += 1
        self.game_state.step_index = self.step_index

    def process_game(self) -> None:

        self.timer.start()
        state = self.game_state.get_state()
        done = self.is_game_over()
        self.interface_update_callback()
        self.log_ml_metrics()

        while not done:

            action = get_action(state,
                                self.mode, self.ep_epsilon, self.modelname)

            new_state, reward, done = self.step(action)

            if self.mode == TRAINING:
                self.save_to_buffer(
                    state, action, reward, new_state, done)

            state = new_state

            self.interface_update_callback()
            self.log_ml_metrics()

            if self.step_index >= 600:
                done = True

        if self.mode == TRAINING:
            update_model(self.buffer)

        self.timer.end()

    def save_to_buffer(self, state_to_choose_an_action, action, reward, next_state, done):
        self.buffer.add(
            (state_to_choose_an_action, action, reward, next_state, done), round(self.total_reward, 3))

    def move_and_update(self, action: int) -> tuple[list, list]:

        self.world.move_agent(action)

        agent_current_state = self.world.evaluate_current_positions_status()

        self.game_state.update_current_state(agent_current_state)

        self.world.handle_collisions(agent_current_state, action)

        self.game_state.next_states = self.game_state.evaluate_next_states()

        nb_collected = self.game_state.num_collectibles - \
            len(self.world.collectibles)

        self.game_state.update_collectibles_status(nb_collected)

        return self.game_state.get_state()

    def is_game_over(self) -> bool:
        is_out_of_bounds = self.game_state.current_state == OUT_OF_BOUNDS
        is_exit_door_found = self.game_state.current_state == ON_EXIT_DOOR

        return is_out_of_bounds or is_exit_door_found

    def is_game_over(self) -> bool:

        return self.game_state.current_state in [
            ON_EXIT_DOOR, OUT_OF_BOUNDS]

    def step(self, action: int) -> tuple[tuple[list, list], float, bool]:

        new_state = self.move_and_update(action)

        step_reward = get_step_reward(self.step_index, self.game_state.current_state,
                                      self.game_state.nb_collected, self.game_state.num_collectibles, self.total_reward)

        self.total_reward += step_reward
        self.steps_reward.append(step_reward)

        self.update_step_count()

        done = self.is_game_over()

        return new_state, step_reward, done

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
