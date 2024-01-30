import pygame
from api.requests import hello_world
from utils.common import generate_datetime_string
# from pygame_module.game_display import GameDisplay
# from episodes.episode_manager import EpisodeManager
import multiprocessing


def main():
    print("main")
    num_cores = multiprocessing.cpu_count()
    num_used_cores = 1
    print(f"Number of cores: {num_cores}")
    try:
        res = hello_world()
        print(res)
    except:
        print("error")
    print(f"Number of used cores: {num_used_cores}")


def main_old():
    pygame.init()

    game_display = GameDisplay()
    episode_manager = EpisodeManager(nb_eps=3000)

    def callback():
        state = episode_manager.get_current_state_to_display()
        info = episode_manager.get_episode_info()
        game_display.update(state, info)

    episode_manager.set_callback(callback)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # episode_manager.run_random()

        model_name = generate_datetime_string() + "_model"

        episode_manager.train_model(model_name)

        # episode_manager.run_model(model_name)

        running = False

    pygame.quit()


if __name__ == "__main__":
    main()
