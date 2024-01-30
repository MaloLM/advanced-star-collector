from abc import ABC, abstractmethod
import matplotlib.patches as patches
import random
import math
from utils.colors import GRAY


class Shape(ABC):
    """
    An abstract base class representing a geometric shape.

    This class provides a framework for different shapes that can be used in a simulation or game environment.
    Subclasses should implement methods for random position generation, boundary checking, and drawing.

    Attributes:
        color (str): The color of the shape.
    """

    def __init__(self, color: tuple[int, int, int] = GRAY):
        self.color = color

    @abstractmethod
    def get_random_position(self):
        """
        Abstract method to get a random position within the shape.

        Returns:
            tuple: A tuple representing a random position (x, y) within the shape.
        """
        pass

    @abstractmethod
    def is_inside(self, x: int, y: int):
        """
        Abstract method to check if a point is inside the shape.

        Args:
            x (float): The x-coordinate of the point.
            y (float): The y-coordinate of the point.

        Returns:
            bool: True if the point is inside the shape, False otherwise.
        """
        pass


class Disk(Shape):
    """
    A subclass of Shape representing a disk.

    Attributes:
        radius (float): The radius of the disk.
    """

    def __init__(self, radius: int, color: tuple[int, int, int] = GRAY):
        super().__init__(color)
        self.radius = radius

    def get_random_position(self) -> tuple[int, int]:
        angle = random.uniform(0, 2 * math.pi)
        r = self.radius * math.sqrt(random.uniform(0, 1))
        return (r * math.cos(angle), r * math.sin(angle))

    def is_inside(self, x: int, y: int) -> bool:
        return x**2 + y**2 <= self.radius**2


class Surface:
    """
    Represents a surface defined by a shape.

    This class is used to define a surface area in a simulation or game environment, 
    which can be of various shapes like square, disk, etc.

    Attributes:
        shape (Shape): The geometric shape defining the surface.
    """

    def __init__(self, shape: Shape = Disk(250), x_pos: int = 400, y_pos: int = 400):
        self.shape = shape
        self.x_pos = x_pos
        self.y_pos = y_pos

    def get_random_position(self) -> tuple[int, int]:
        rel_x, rel_z = self.shape.get_random_position()

        x = int(self.x_pos + rel_x)
        y = int(self.y_pos + rel_z)

        return x, y

    def is_inside(self, x: int, y: int) -> bool:
        adjusted_x = x - self.x_pos
        adjusted_z = y - self.y_pos

        return self.shape.is_inside(adjusted_x, adjusted_z)
