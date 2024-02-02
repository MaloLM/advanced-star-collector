import math
import numpy as np
from datetime import datetime


@staticmethod
def distribute_episodes(num_episodes, num_processes):
    """
    Distributes the number of episodes among processes.

    Args:
    num_episodes (int): The total number of episodes to run.
    num_processes (int): The number of processes.

    Returns:
    list: A list containing the number of episodes each process will execute.
    """
    base_count = num_episodes // num_processes
    remainder = num_episodes % num_processes

    episodes_per_process = [base_count for _ in range(num_processes)]

    for i in range(remainder):
        episodes_per_process[i] += 1

    return episodes_per_process


@staticmethod
def generate_datetime_string():
    """
    Generates a string representing the current date and time.

    Returns:
        str: A string formatted as YearMonthDay_HoursMinutesSeconds.
    """
    now = datetime.now()
    # as Year-Month-Day_Hours-Minutes-Seconds
    return now.strftime("%Y-%m-%d_%H-%M-%S")


@staticmethod
def distance(pos1, pos2):
    """
    Calculates the Euclidean distance between two points.

    Args:
        pos1 (tuple): The first point (x, y).
        pos2 (tuple): The second point (x, y).

    Returns:
        float: The Euclidean distance between pos1 and pos2.
    """
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


@staticmethod
def flatten_list(list_to_flatten):
    """
    Flattens a nested list structure into a single list of floats.

    This method is useful for preprocessing state representations that 
    are initially in a nested list format, commonly encountered in 
    environments with multi-dimensional state spaces.

    Args:
        state (list): A list that may contain nested lists. Elements can be 
                    of any type that can be converted to a float.

    Returns:
        list: A single-level list where all values from the original nested 
            list structure are converted to floats. This flat list is 
            suitable for use in ML models that require flat input features.
    """
    flattened_state = []
    for item in list_to_flatten:
        if isinstance(item, list):
            flattened_state.extend(item)
        else:
            flattened_state.append(item)
    return [float(i) for i in flattened_state]


@staticmethod
def normalize_group(data, min_val, max_val):
    data_array = np.array(data)
    normalized_array = (data_array - min_val) / (max_val - min_val)
    return normalized_array.tolist()


@staticmethod
def one_hot_encode(value, num_classes):
    one_hot = np.zeros(num_classes)
    one_hot[value] = 1
    return one_hot.tolist()


@staticmethod
def calculate_decay_lambda(factor, total_episodes, target_y=0.01):
    """
    Calcule le taux de décroissance lambda pour une fonction de décroissance exponentielle.

    Args:
        factor (float): Le facteur du total des épisodes à quel point y doit être proche de 0.
        total_episodes (int): Le nombre total d'épisodes.
        target_y (float, optional): La valeur cible de y à ce point. Valeur par défaut à 0.01.

    Returns:
        float: Le taux de décroissance lambda calculé.
    """
    critical_point = factor * total_episodes
    lambda_value = -math.log(target_y) / critical_point
    return lambda_value


@staticmethod
def exponential_decay(x, total_episodes, factor=0.85, target_y=0.01):
    """
    Calcule la valeur de décroissance exponentielle pour un x donné, un facteur, et un total d'épisodes.

    Args:
        x (float): La valeur actuelle de x (par exemple, le numéro de l'épisode).
        factor (float): Le facteur du total des épisodes à quel point y doit être proche de 0.
        total_episodes (int): Le nombre total d'épisodes.
        target_y (float, optional): La valeur cible de y à ce point. Valeur par défaut à 0.01.

    Returns:
        float: La valeur de y après décroissance.
    """
    lambda_value = calculate_decay_lambda(factor, total_episodes, target_y)
    return math.exp(-lambda_value * x)
