from time import sleep
from api.requests import is_model_saved
from episodes.runners import run_multicore_training, run_random, run_trained_model

from utils.timer import Timer


def main():
    timer = Timer(start_now=True)

    num_used_cores = 1
    num_episode = 400

    # run_random(num_eps=200)

    modelname = run_multicore_training(num_used_cores, num_episode)

    exit_counter = 0
    while not is_model_saved(modelname) or exit_counter == 50:
        sleep(10)

    # modelname = '2024-02-02_19-37-01_model'
    run_trained_model(modelname, 100)

    timer.end()
    print("Total duration", timer.get_formatted_duration())


if __name__ == '__main__':
    main()
