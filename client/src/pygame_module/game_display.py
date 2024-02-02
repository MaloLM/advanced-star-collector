import os
import math
import pygame
import datetime
from utils.colors import WHITE, BLACK
from utils.game_states import RANDOM, TESTING
from settings import FRAMES_PATH, GAME_TITLE, WINDOW_HEIGHT, WINDOW_WIDTH


class GameDisplay:

    def __init__(self, width: int = WINDOW_WIDTH, height: int = WINDOW_HEIGHT, background_color: tuple = BLACK):
        self.width = width
        self.height = height
        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(GAME_TITLE)
        self.background_color = background_color

    def update(self, game_state, episode_info, mode):
        self.reset_content()
        self.draw_game_state(game_state)
        self.display_info(episode_info)
        pygame.display.flip()
        if mode == TESTING:  
            self.save_pygame_frame()

    def reset_content(self):
        self.window.fill(self.background_color)

    def draw_circle(self, x, y, radius, color):
        pygame.draw.circle(self.window, color, (x, y), radius)

    def draw_line(self, start_pos, end_pos, color, width=1):
        pygame.draw.line(self.window, color, start_pos, end_pos, width)

    def draw_game_state(self, game_state):
        agent = game_state["agent"]
        agent_heads = agent["heads"]
        exit_door = game_state["exit-door"]
        collectibles = game_state["collectibles"]
        surface = game_state["surface"]

        # draw game surface
        self.draw_circle(surface["x"], surface["y"],
                         surface["radius"], surface["color"])

        # draw agent
        self.draw_circle(agent["x"], agent["y"],
                         agent["radius"], agent["color"])

        # draw agent heads
        for head_idx in agent_heads:
            head_x, head_y = self.calculate_coordinates(
                agent_heads[head_idx]["distance_to_center"], agent["x"],
                agent["y"], agent_heads[head_idx]["angle"]
            )

            intersec_x = agent_heads[head_idx]["intersection_with_circle_x"]
            intersec_y = agent_heads[head_idx]["intersection_with_circle_y"]

            self.draw_line((agent["x"], agent["y"]),
                           (head_x, head_y), agent["color"], 4)

            if agent_heads[head_idx]["is_within_surface"] == True:
                self.draw_line((head_x, head_y),
                               (intersec_x, intersec_y), agent_heads[head_idx]["color"], 1)

            self.draw_circle(
                head_x, head_y, agent_heads[head_idx]["radius"], agent_heads[head_idx]["color"])

        # draw exit door
        self.draw_circle(exit_door["x"], exit_door["y"],
                         exit_door["radius"], exit_door["color"])

        # draw collectibles
        for collectible in collectibles:
            self.draw_circle(collectibles[collectible]["x"], collectibles[collectible]["y"],
                             collectibles[collectible]["radius"], collectibles[collectible]["color"])

    def save_pygame_frame(self, frames_path=FRAMES_PATH):
        if not os.path.exists(frames_path):
            os.makedirs(frames_path)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        filename = f"frame_{timestamp}.png"
        filepath = os.path.join(frames_path, filename)

        pygame.image.save(self.window, filepath)

    def display_info(self, episode_info):
        font = pygame.font.SysFont(None, 24)
        start_y = 10
        line_height = 30

        for key, value in episode_info.items():
            info_text = f"{key}: {value}"
            text = font.render(info_text, True, WHITE)
            self.window.blit(text, (10, start_y))
            start_y += line_height  # Move to the next line

    def calculate_coordinates(self, distance, center_x, center_y, angle):
        # Convert angle to radians
        angle_rad = math.radians(angle)

        # Calculate the x and y offsets using cosine and sine
        offset_x = distance * math.cos(angle_rad)
        offset_y = distance * math.sin(angle_rad)

        # Calculate the final coordinates and return them as integers
        final_x = int(center_x + offset_x)
        final_y = int(center_y + offset_y)

        return final_x, final_y
