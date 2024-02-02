from episodes.runners import run_multicore_training, run_random, run_trained_model

from utils.timer import Timer


def main():
    timer = Timer(start_now=True)

    num_used_cores = 3
    num_episode = 600

    modelname = run_multicore_training(num_used_cores, num_episode)

    # run_trained_model(modelname, 10)

    # wait server finished before running model (waiting loop ?)
    # run_random(num_eps=200)

    timer.end()
    print("Total episode duration", timer.get_formatted_duration())


if __name__ == '__main__':
    main()
