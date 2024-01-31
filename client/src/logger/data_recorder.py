import os
import imageio
import datetime
from utils.common import generate_datetime_string
from settings import EPISODE_SAVING_TO_GIF_PATH, FRAMES_PATH


def create_gif(frames_dir=FRAMES_PATH, gif_path=EPISODE_SAVING_TO_GIF_PATH, gif_name="_world", delete_frames=True, loop=3):
    """
    Creates a GIF from the saved frames in the specified directory.

    Args:
        frame_dir (str): The directory where frames are saved.
        gif_path (str): The path where the GIF should be saved.
        delete_frames (bool): If True, deletes the original frames after creating the GIF. Defaults to True.
        loop (int): The number of times the GIF should loop; 0 means infinite loop. Defaults to 0.
    """
    images = []

    file_names = sort_files_chronologically(frames_dir)

    for filename in file_names:
        if filename.endswith('.png'):
            file_path = os.path.join(frames_dir, filename)
            images.append(imageio.imread(file_path))
            if delete_frames:
                os.remove(file_path)

    datestr = generate_datetime_string()

    if not os.path.exists(gif_path):
        os.makedirs(gif_path)

    gif_path = gif_path + datestr + gif_name + ".gif"

    imageio.mimsave(gif_path, images, duration=1, loop=loop)

    delete_frames_directory(frames_dir)


def delete_frames_directory(frames_dir):
    """
    Deletes all files in the specified directory and then removes the directory.

    :param frames_dir: Path to the directory containing frame files.
    """
    if os.path.exists(frames_dir):
        for filename in os.listdir(frames_dir):
            file_path = os.path.join(frames_dir, filename)
            os.remove(file_path)

        os.rmdir(frames_dir)


def sort_files_chronologically(frame_dir):
    file_names = os.listdir(frame_dir)

    file_names = [f for f in file_names if f.startswith(
        'frame_') and f.endswith('.png')]

    def extract_timestamp(file_name):
        timestamp_str = file_name[len('frame_'):-len('.png')]
        return datetime.datetime.strptime(timestamp_str, "%Y-%m-%d-%H-%M-%S-%f")

    file_names.sort(key=extract_timestamp)

    return file_names
