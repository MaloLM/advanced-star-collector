from .head import Head
from .surface import Disk
from utils.colors import BLUE


class Agent:
    """
    update me
    """

    def __init__(self, radius: int = 30, color: str = BLUE, step: int = 40):
        self.x_pos = None
        self.y_pos = None
        self.door_found = 0
        self.step = step
        self.shape = Disk(radius, color=color)
        self.heads: list = []
        self.head_detection = None
        self.head_distance_to_a_collectible = None

        angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165,
                  180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]

        # [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        for angle in angles:
            self.add_head(Head(self.step, angle))

    def add_head(self, head: Head):
        self.heads.append(head)

    def remove_head(self, head: Head):
        if head in self.heads:
            self.heads.remove(head)
        else:
            raise ValueError(
                "The specified head is not attached to this agent.")

    def calculate_new_position(self, head: Head):
        """
        update me
        """
        x_pos, y_pos = head.get_seen_position()

        return x_pos, y_pos

    def move(self, head_index: int):
        """
        update me
        """
        head: Head = self.heads[head_index]
        self.x_pos, self.y_pos = head.get_seen_position(
            (self.x_pos, self.y_pos))
