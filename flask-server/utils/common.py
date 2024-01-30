from datetime import datetime
import math

import numpy as np


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
