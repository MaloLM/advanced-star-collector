# GAME SETTINGS
GAME_TITLE = "RL star collector game"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
FPS = 60

# Reinforcement Learning settings
NB_OF_EPISODES = 3000

STATE_SIZE = 4 + 3 + 8 + 8 + 8
ACTION_POSSIBILITIES = 8

EPSILON = 1.0
EPSILON_DECAY = 0.999
MIN_EPSILON = EPSILON - EPSILON_DECAY

MAX_STEP_PER_EP = 200  # before no efficiency

# Directorioes
FRAMES_PATH = "/tmp/frames"
EPISODE_SAVING_TO_GIF_PATH = '/tmp/games/'
