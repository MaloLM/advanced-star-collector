import math
from utils.common import distance
from utils.colors import PALE_GRAY
from rl_module.world.head import Head
from rl_module.world.agent import Agent
from rl_module.world.surface import Surface
from rl_module.world.collectible import Collectible, ExitDoor
from utils.game_states import ON_EXIT_DOOR, ONTO_SURFACE, OUT_OF_BOUNDS, STAR_COLLECTED


class World:
    """
    Represents the game world in a simulation environment.

    This class manages the game elements such as the surface, agent, collectibles, and exit door.
    It provides methods to handle movements and interactions within the world, check for collisions, and draw the game state.
    """

    def __init__(self, surface: Surface = Surface()):
        self.surface: Surface = surface
        self.collectibles: list[Collectible] = []
        self.exit_door: ExitDoor = self.set_exit_door()
        self.agent: Agent = self.set_agent()

    def reset(self, nb_collectibles=4):
        self.set_exit_door()

        for _ in range(nb_collectibles):
            collectible = Collectible()
            collectible_pos = self.get_free_random_position(
                collectible.shape.radius)

            self.set_collectible(
                collectible, collectible_pos[0], collectible_pos[1])

        self.set_agent()

    def is_collision(self, position: tuple, radius: int):
        """
        Checks if a given position and radius collide with any game element in the world.

        Args:
            position (tuple): The position (x, y) to check for collisions.
            radius (float): The radius to consider for collision detection.

        Returns:
            bool: True if there is a collision, False otherwise.
        """
        for collectible in self.collectibles:
            if distance(position, (collectible.x_pos, collectible.y_pos)) < radius + collectible.shape.radius:
                return True

        if hasattr(self, 'exit_door') and isinstance(self.exit_door, ExitDoor):
            if distance(position, (self.exit_door.x_pos, self.exit_door.y_pos)) < radius + self.exit_door.shape.radius:
                return True
        return False

    def get_free_random_position(self, radius: int) -> tuple[int, int]:
        """
        Finds a random position in the world that is not in collision with any game element.

        Args:
            radius (float): The radius to use for collision detection.

        Returns:
            tuple: A free random position (x, y) in the world.
        """
        while True:
            random_pos = self.surface.get_random_position()
            if not self.is_collision(random_pos, radius):
                return random_pos

    def set_collectible(self, collectible: Collectible, x: int = None, y: int = None):
        """
        Places a collectible in the world at a specified position.

        Args:
            collectible (Collectible): The collectible to place in the world.
            x (float, optional): The x-coordinate of the collectible's position. Defaults to None.
            y (float, optional): The y-coordinate of the collectible's position. Defaults to None.
        """
        collectible.x_pos = x
        collectible.y_pos = y
        self.collectibles.append(collectible)

    def remove_collectible(self, collectible: Collectible):
        """
        Removes a collectible from the world.

        Args:
            collectible (Collectible): The collectible to remove.
        """
        for world_collectibles in self.collectibles:
            if id(world_collectibles) == id(collectible):
                del world_collectibles

    def set_exit_door(self):
        """
        Places the exit door in the world at a specified position.

        Args:
            exit_door (ExitDoor): The exit door to place in the world.
            x (float, optional): The x-coordinate of the exit door's position. Defaults to None.
            y (float, optional): The y-coordinate of the exit door's position. Defaults to None.
        """
        exit_door = ExitDoor()

        exit_door_x, exit_door_z = self.get_free_random_position(
            exit_door.shape.radius)
        exit_door.x_pos = exit_door_x
        exit_door.y_pos = exit_door_z
        self.exit_door = exit_door
        return ExitDoor

    def set_agent(self) -> None:
        """
        Places the agent in the world at a specified position.

        Args:
            agent (Agent): The agent to place in the world.
            x (float, optional): The x-coordinate of the agent's position. Defaults to None.
            y (float, optional): The y-coordinate of the agent's position. Defaults to None.
        """
        self.agent = Agent()
        (x, y) = self.get_free_random_position(self.agent.shape.radius)
        self.agent.x_pos = x
        self.agent.y_pos = y
        return self.agent

    def move_agent(self, action: int) -> None:
        """
        Moves the agent based on its current status and updates the game elements accordingly.

        This method handles the logic for agent movement, including collision detection
        with collectibles and the exit door, and updates the game state.
        """
        next_status = self.evaluate_next_position_status(action)

        if next_status in [STAR_COLLECTED, ON_EXIT_DOOR]:
            collisions: Collectible = self.find_collisions_at_next_position(
                self.agent)

            for collision in collisions[action]:
                if isinstance(collision, ExitDoor):
                    self.agent.door_found = 1

                self.handle_collectible_collision(collision)

        self.agent.move(action)

    def handle_collectible_collision(self, collision: Collectible) -> None:
        """
        Handles logic when the agent collides with a collectible.
        """
        for collectible in self.collectibles:
            if id(collectible) == id(collision):
                self.collectibles.remove(collectible)
                break

    def find_collisions_at_next_position(self, agent: Agent) -> list:
        """
        Identifies any game elements in the world that would collide with the agent's next position.

        Args:
            agent (Agent): The agent for which to check potential collisions.

        Returns:
            list: A list of game elements (collectibles, exit door) that are in collision with the agent's next position.
        """
        collisions = []
        agent_center_pos = (agent.x_pos, agent.y_pos)

        for head in agent.heads:
            position = head.get_seen_position(agent_center_pos)
            radius = agent.shape.radius

            current_head_collisions = []

            # Check collision with collectibles
            for collectible in self.collectibles:
                if distance(position, (collectible.x_pos, collectible.y_pos)) < radius + collectible.shape.radius:
                    current_head_collisions.append(collectible)

            # Check collision with exit door
            if self.exit_door and distance(position, (self.exit_door.x_pos, self.exit_door.y_pos)) < radius + self.exit_door.shape.radius:
                current_head_collisions.append(self.exit_door)

            collisions.append(current_head_collisions)
        return collisions

    def evaluate_current_positions_status(self) -> int:
        """
        Evaluates and categorizes the status of the agent's current position in the world.

        The status is determined based on whether the position is out of bounds, free, or colliding with any collectibles, or the exit door.

        Returns:
            int: An integer code representing the status of the next position (0: out_of_bounds, 1: nothing, 2: collectible, 3: exit_door, 4: unknown).
        """
        radius = self.agent.shape.radius

        position = (self.agent.x_pos, self.agent.y_pos)

        if not self.is_within_surface(position):
            return OUT_OF_BOUNDS
        else:
            if self.is_collision_with_collectibles(position, radius):
                return STAR_COLLECTED

            elif self.is_collision_with_exit_door(position, radius):
                return ON_EXIT_DOOR

            else:
                return ONTO_SURFACE

    def evaluate_next_position_status(self, action: int) -> int:
        """
        Evaluates and categorizes the status of the agent's next position in the world.

        The status is determined based on whether the position is out of bounds, free, or colliding with any collectibles, or the exit door.

        Returns:
            int: An integer code representing the status of the next position (0: out_of_bounds, 1: nothing, 2: collectible, 3: exit_door, 4: unknown).
        """
        radius = self.agent.shape.radius
        agent_center = (self.agent.x_pos, self.agent.y_pos)

        head: Head = self.agent.heads[action]

        x, y = head.get_seen_position(agent_center)

        if not self.is_within_surface((x, y)):
            return OUT_OF_BOUNDS
        else:
            if self.is_collision_with_collectibles((x, y), radius):
                return STAR_COLLECTED
            elif self.is_collision_with_exit_door((x, y), radius):
                return ON_EXIT_DOOR
            else:
                return ONTO_SURFACE

    def is_within_surface(self, position: tuple[int, int]) -> bool:
        """
        Checks if a given position is within the bounds of the world's surface.

        Args:
            position (tuple): The position (x, y) to check.

        Returns:
            bool: True if the position is within the surface bounds, False otherwise.
        """
        x, y = position
        return self.surface.is_inside(x, y)

    def get_agent_direction_sensing(self):
        head_detection = []
        head_distance_to_a_collectible = []

        for head in self.agent.heads:
            center = (self.surface.x_pos, self.surface.y_pos)
            head_pos = head.get_seen_position(
                (self.agent.x_pos, self.agent.y_pos))
            radius = self.surface.shape.radius
            angle = head.angle

            if not self.is_within_surface(head_pos):
                head.is_within_surface = False
                head_detection.append(0)
                head_distance_to_a_collectible.append(
                    radius * 4)
            else:
                head.is_within_surface = True

                head.intersection_with_circle_pos = World.find_nearest_intersection_point_to_surface(
                    center, radius, head_pos, angle)

                (nearest_collectible, distance_to_collectible, nearest_point) = self.find_nearest_intersection(
                    head_pos, head.intersection_with_circle_pos)

                if nearest_collectible is not None:
                    head.intersection_with_circle_pos = nearest_point
                    if isinstance(nearest_collectible, ExitDoor):
                        head_detection.append(2)
                        head.sensing_color = self.exit_door.color
                    else:
                        head_detection.append(1)  # basic collectible
                        head.sensing_color = Collectible.color

                    head_distance_to_a_collectible.append(
                        distance_to_collectible)
                else:
                    head_detection.append(0)
                    head.sensing_color = PALE_GRAY
                    head_distance_to_a_collectible.append(
                        distance_to_collectible)

        return head_detection, head_distance_to_a_collectible

    @staticmethod
    def find_nearest_intersection_point_to_surface(center: tuple[int, int], radius: int, point_coords: tuple[int, int], point_angle: int):
        angle_rad = math.radians(point_angle)

        dx, dy = math.cos(angle_rad), math.sin(angle_rad)

        cx, cy = center
        x0, y0 = point_coords

        a = dx**2 + dy**2
        b = 2 * (dx * (x0 - cx) + dy * (y0 - cy))
        c = (x0 - cx)**2 + (y0 - cy)**2 - radius**2

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return None
        else:
            t1 = (-b + math.sqrt(discriminant)) / (2 * a)
            t2 = (-b - math.sqrt(discriminant)) / (2 * a)

            t = t1 if t1 >= 0 else t2

            intersection_point = (x0 + t * dx, y0 + t * dy)

            return intersection_point

    def find_nearest_intersection(self, starting_point, ending_point):
        nearest_collectible = None
        nearest_intersection_point = None
        min_distance = float('inf')

        collectibles_and_exit_door = self.collectibles + [self.exit_door]

        for item in collectibles_and_exit_door:
            collectible_pos, collectible_radius = (item.x_pos,
                                                   item.y_pos), item.shape.radius

            (intersection_point, distance) = self._line_circle_intersection(
                starting_point, ending_point, collectible_pos, collectible_radius)

            if intersection_point is not None and distance < min_distance:
                min_distance = int(distance)
                nearest_collectible = item
                nearest_intersection_point = intersection_point

        return (nearest_collectible, min_distance, nearest_intersection_point) if nearest_collectible else (None, self.surface.shape.radius * 4, None)

    def _line_circle_intersection(self, starting_point, ending_point, collectible_pos, collectible_radius):
        dx = ending_point[0] - starting_point[0]
        dy = ending_point[1] - starting_point[1]

        fx = starting_point[0] - collectible_pos[0]
        fy = starting_point[1] - collectible_pos[1]

        a = dx**2 + dy**2

        if a == 0:
            distance = math.sqrt(fx**2 + fy**2)
            if distance <= collectible_radius:
                return starting_point, 0
            else:
                return None, float('inf')

        b = 2 * (fx * dx + fy * dy)
        c = fx**2 + fy**2 - collectible_radius**2

        discriminant = b**2 - 4 * a * c

        intersection_points = []
        if discriminant >= 0:
            discriminant_sqrt = math.sqrt(discriminant)
            t1 = (-b + discriminant_sqrt) / (2 * a)
            t2 = (-b - discriminant_sqrt) / (2 * a)

            if 0 <= t1 <= 1:
                intersection_points.append(
                    (starting_point[0] + t1 * dx, starting_point[1] + t1 * dy))
            if 0 <= t2 <= 1:
                intersection_points.append(
                    (starting_point[0] + t2 * dx, starting_point[1] + t2 * dy))

        if not intersection_points:
            return None, float('inf')

        nearest_point = min(intersection_points,
                            key=lambda point: (point[0] - starting_point[0])**2 + (point[1] - starting_point[1])**2)
        min_distance = math.sqrt(
            (nearest_point[0] - starting_point[0])**2 + (nearest_point[1] - starting_point[1])**2)

        return nearest_point, min_distance

    @staticmethod
    def is_point_on_segment(point, starting_point, ending_point):
        return min(starting_point[0], ending_point[0]) <= point[0] <= max(starting_point[0], ending_point[0]) \
            and min(starting_point[1], ending_point[1]) <= point[1] <= max(starting_point[1], ending_point[1])

    def is_collision_with_collectibles(self, position: tuple[int, int], radius: int) -> bool:
        """
        Determines if there is a collision with any collectible at the given position.

        Args:
            position (tuple): The position (x, y) to check for collision.
            radius (float): The radius to consider for collision detection.

        Returns:
            bool: True if a collision with a collectible is detected, False otherwise.
        """
        for collectible in self.collectibles:
            if distance(position, (collectible.x_pos, collectible.y_pos)) < radius + collectible.shape.radius:
                return True
        return False

    def is_collision_with_exit_door(self, position: tuple[int, int], radius: int) -> bool:
        """
        Determines if there is a collision with the exit door at the given position.

        Args:
            position (tuple): The position (x, y) to check for collision.
            radius (float): The radius to consider for collision detection.

        Returns:
            bool: True if a collision with the exit door is detected, False otherwise.
        """
        if self.exit_door and distance(position, (self.exit_door.x_pos, self.exit_door.y_pos)) < radius + self.exit_door.shape.radius:
            return True
        return False

    def get_agent_position(self) -> (float, float):
        """
        Retrieves the current position of the agent in the world.

        Returns:
            tuple: A tuple (x, y) representing the agent's current position.
        """
        return (self.agent.x_pos, self.agent.y_pos)

    def remove_agent(self) -> Agent:
        """
        Removes the agent from the world.
        """
        del self.agent
