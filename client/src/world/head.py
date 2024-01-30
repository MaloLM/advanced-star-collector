import math
from utils.colors import BLUE, PALE_GRAY


class Head:
    """
    Represents the head of an agent, positioned relative to the center of the agent.

    Attributes:
        distance_to_center (int): The distance from the center of the agent to the head.
        angle (int): The angle of the head relative to the agent's forward direction, in degrees.

    Methods:
        get_seen_position(agent_center_pos): Calculates the position of the head based on the agent's center position.
    """

    def __init__(self, distance_to_center: int, angle: int, color: tuple = BLUE, radius: int = 5):
        """
        Initializes a new instance of the Head class.

        Args:
            distance_to_center (int): The distance from the agent's center to the head.
            angle (int): The angle of the head relative to the agent's forward direction, in degrees.
        """
        self.distance_to_center = distance_to_center
        self.angle = angle
        self.color = color
        self.radius = radius
        self.intersection_with_circle_pos = (None, None)
        self.is_within_surface = False
        self.sensing_color = PALE_GRAY

    def get_seen_position(self, agent_center_pos: tuple[int, int]) -> tuple[int, int]:
        """
        Calculates the position of the head based on the agent's center position, the distance from the center, and the angle.

        Args:
            agent_center_pos (tuple): The (x, y) coordinates of the agent's center position.

        Returns:
            tuple: The (x, y) coordinates of the head's position.
        """
        x_center_pos, z_center_pos = agent_center_pos

        angle_rad = math.radians(self.angle)

        head_x_pos = x_center_pos + \
            self.distance_to_center * math.cos(angle_rad)
        head_y_pos = z_center_pos + \
            self.distance_to_center * math.sin(angle_rad)

        return (head_x_pos, head_y_pos)
