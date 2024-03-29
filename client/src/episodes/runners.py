import sys
import pygame
import multiprocessing
from pygame_module.game_display import GameDisplay
from episodes.episode_manager import EpisodeManager
from api.requests import end_training, save_model, start_training
from utils.common import distribute_episodes, generate_datetime_string


def run_multicore_training(num_used_cores: int, num_episodes: int):
    modelname = generate_datetime_string() + "_model"

    start_training(modelname)

    total_num_cores = multiprocessing.cpu_count()
    if num_used_cores <= total_num_cores:
        episode_distribution = distribute_episodes(
            num_episodes, num_used_cores)
        processes = []

        try:
            for process_index in range(num_used_cores):
                num_eps = episode_distribution[process_index]
                p = multiprocessing.Process(
                    target=run_training_client, args=(process_index, num_eps))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        except KeyboardInterrupt:
            end_training()
            save_model(modelname)
            sys.exit()

        end_training()

    return modelname


def run_training_client(process_index, num_eps):
    pygame.init()
    game_display = GameDisplay()
    episode_manager = EpisodeManager(nb_eps=num_eps)

    def callback():
        state = episode_manager.get_current_state_to_display()
        info = episode_manager.get_episode_info()
        mode = episode_manager.mode
        game_display.update(state, info, mode)

    episode_manager.set_callback(callback)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        episode_manager.train_model()

        running = False

    pygame.quit()


def run_random(num_eps):
    pygame.init()
    game_display = GameDisplay()
    episode_manager = EpisodeManager(nb_eps=num_eps)

    def callback():
        state = episode_manager.get_current_state_to_display()
        info = episode_manager.get_episode_info()
        mode = episode_manager.mode
        game_display.update(state, info, mode)

    episode_manager.set_callback(callback)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        episode_manager.run_random()

        running = False

    pygame.quit()


def run_trained_model(modelname, num_eps):
    pygame.init()
    game_display = GameDisplay()
    episode_manager = EpisodeManager(nb_eps=num_eps)

    def callback():
        state = episode_manager.get_current_state_to_display()
        info = episode_manager.get_episode_info()
        mode = episode_manager.mode
        game_display.update(state, info, mode)

    episode_manager.set_callback(callback)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        episode_manager.run_model(modelname)

        running = False

    pygame.quit()
