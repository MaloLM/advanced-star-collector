import pygame
import multiprocessing
from utils.timer import Timer
from pygame_module.game_display import GameDisplay
from episodes.episode_manager import EpisodeManager
from api.requests import end_training, start_training
from utils.common import distribute_episodes, generate_datetime_string


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

        # episode_manager.run_random()

        episode_manager.train_model()

        # episode_manager.run_model()

        running = False

    pygame.quit()


def main():
    timer = Timer()
    timer.start()
    total_num_cores = multiprocessing.cpu_count()
    num_used_cores = 2
    num_episode = 400
    modelname = generate_datetime_string() + "_model"

    # recupérer status et vérifier que 200 avant de continuer
    start_training(modelname)

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

        end_training()
        timer.end()

        print("Total episode duration", timer.get_formatted_duration())


if __name__ == '__main__':
    main()
