import pygame
from utils.common import distribute_episodes, generate_datetime_string
from pygame_module.game_display import GameDisplay
from episodes.episode_manager import EpisodeManager
import multiprocessing
import pygame


def run_episode(process_index, num_eps):
    pygame.init()
    game_display = GameDisplay()
    episode_manager = EpisodeManager(nb_eps=num_eps)

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

        episode_manager.run_random()
        # Add more logic as needed
        running = False

    pygame.quit()


def main():
    total_num_cores = multiprocessing.cpu_count()
    num_used_cores = 2
    num_episode = 20

    if num_used_cores <= total_num_cores:
        episode_distribution = distribute_episodes(num_episode, num_used_cores)
        processes = []

        for process_index in range(num_used_cores):
            num_eps = episode_distribution[process_index]
            p = multiprocessing.Process(
                target=run_episode, args=(process_index, num_eps))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


if __name__ == '__main__':
    main()

    # try:
    #     res = hello_world()
    # except:
    #     print("error")
