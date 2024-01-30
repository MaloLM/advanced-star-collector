from .surface import Disk
from utils.colors import GOLD, GREEN


class Collectible():
    """
    Represents a collectible item in the game environment.

    This class defines the basic properties and behaviors of a collectible item, 
    including its position, color, and shape. 

    Attributes:
        x_pos (float): The x-coordinate of the collectible's position. Initially None.
        y_pos (float): The y-coordinate of the collectible's position. Initially None.
        color (str): The color of the collectible. Defaults to 'GOLD'.
        shape (Disk): The geometric shape of the collectible, represented as a disk.
    """
    color = GOLD

    def __init__(self, color: tuple[int, int, int] = GOLD):
        self.x_pos = None
        self.y_pos = None
        self.color = color
        self.shape = Disk(35, color)


class ExitDoor(Collectible):
    """
    Represents the exit door in the game environment.

    The ExitDoor is a special type of Collectible that signifies the end of a level or game.
    It inherits from the Collectible class and can have its own unique properties and behaviors.

    Attributes:
        color (str): The color of the exit door. Defaults to 'green'.
    """

    def __init__(self, color: tuple[int, int, int] = GREEN):
        super().__init__(color)
