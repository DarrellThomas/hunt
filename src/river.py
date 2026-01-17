"""
River system for the HUNT ecosystem.
Creates flowing water that affects agent movement.
"""

import numpy as np
from config import *


class River:
    """
    Generates and manages a river with optional island splitting.
    """

    def __init__(self, world_width, world_height):
        """
        Initialize the river.

        Args:
            world_width: Width of the world
            world_height: Height of the world
        """
        self.width = world_width
        self.height = world_height
        self.enabled = RIVER_ENABLED

        if not self.enabled:
            return

        self.river_width = RIVER_WIDTH
        self.flow_speed = RIVER_FLOW_SPEED
        self.curviness = RIVER_CURVINESS
        self.split = RIVER_SPLIT
        self.split_start = RIVER_SPLIT_START
        self.split_end = RIVER_SPLIT_END
        self.island_width = RIVER_ISLAND_WIDTH

        # Generate river path
        self._generate_path()

    def _generate_path(self):
        """Generate the river centerline path."""
        # Create path points from left to right
        num_points = 50
        t = np.linspace(0, 1, num_points)

        # X goes from 0 to world_width
        self.path_x = t * self.width

        # Y starts at middle, curves based on curviness
        base_y = self.height / 2

        # Use sine waves for natural curves
        curve = np.sin(t * np.pi * 4) * self.curviness * self.height * 0.2
        curve += np.sin(t * np.pi * 2.3) * self.curviness * self.height * 0.15

        self.path_y = base_y + curve

        # Clamp to world bounds
        self.path_y = np.clip(self.path_y, self.river_width, self.height - self.river_width)

        # Calculate flow direction at each point (tangent to path)
        dx = np.gradient(self.path_x)
        dy = np.gradient(self.path_y)

        # Normalize to unit vectors
        mag = np.sqrt(dx**2 + dy**2)
        self.flow_dir_x = dx / mag
        self.flow_dir_y = dy / mag

    def is_in_river(self, x, y):
        """
        Check if a position is in the river.

        Args:
            x, y: Position to check

        Returns:
            Boolean indicating if position is in river
        """
        if not self.enabled:
            return False

        # Find nearest point on river path
        distances = np.sqrt((self.path_x - x)**2 + (self.path_y - y)**2)
        nearest_idx = np.argmin(distances)

        # Check if within river width
        dist_to_center = distances[nearest_idx]

        if self.split:
            # Check if in split region
            t = nearest_idx / len(self.path_x)
            if self.split_start <= t <= self.split_end:
                # In split region - check if in top or bottom channel
                center_y = self.path_y[nearest_idx]
                offset = self.island_width / 2

                # Check if ON THE ISLAND (between channels) - NO FLOW HERE!
                dist_from_center = abs(y - center_y)
                if dist_from_center < offset:
                    # On the island - not in river!
                    return False

                # Top channel (above island)
                top_channel_center = center_y - offset
                in_top = abs(y - top_channel_center) < self.river_width / 2

                # Bottom channel (below island)
                bottom_channel_center = center_y + offset
                in_bottom = abs(y - bottom_channel_center) < self.river_width / 2

                return in_top or in_bottom

        # Not in split region, check normal river bounds
        return dist_to_center < self.river_width / 2

    def is_on_island(self, x, y):
        """
        Check if a position is on the island (land sanctuary).

        Args:
            x, y: Position to check

        Returns:
            Boolean indicating if position is on island
        """
        if not self.enabled or not self.split:
            return False

        # Find nearest point on river path
        distances = np.sqrt((self.path_x - x)**2 + (self.path_y - y)**2)
        nearest_idx = np.argmin(distances)

        # Check if in split region
        t = nearest_idx / len(self.path_x)
        if self.split_start <= t <= self.split_end:
            center_y = self.path_y[nearest_idx]
            offset = self.island_width / 2

            # Check if within island bounds
            dist_from_center = abs(y - center_y)
            return dist_from_center < offset

        return False

    def get_flow_at(self, x, y):
        """
        Get flow velocity at a position.
        Returns (0, 0) for land areas including the island.

        Args:
            x, y: Position

        Returns:
            (flow_x, flow_y): Flow velocity components
        """
        # Island is a safe land sanctuary - no flow!
        if self.is_on_island(x, y):
            return 0.0, 0.0

        if not self.enabled or not self.is_in_river(x, y):
            return 0.0, 0.0

        # Find nearest point on river path
        distances = np.sqrt((self.path_x - x)**2 + (self.path_y - y)**2)
        nearest_idx = np.argmin(distances)

        # Get flow direction at this point
        flow_x = self.flow_dir_x[nearest_idx] * self.flow_speed
        flow_y = self.flow_dir_y[nearest_idx] * self.flow_speed

        return flow_x, flow_y

    def get_flow_at_batch(self, positions):
        """
        Vectorized version: Get flow velocity for multiple positions at once.

        Args:
            positions: Nx2 array of positions

        Returns:
            Nx2 array of flow velocities
        """
        if not self.enabled or len(positions) == 0:
            return np.zeros_like(positions)

        flows = np.zeros_like(positions)

        for i, pos in enumerate(positions):
            flows[i] = self.get_flow_at(pos[0], pos[1])

        return flows

    def island_behavior(self, agent_type, x, y):
        """
        Get behavior modifiers for agents on the island.

        Args:
            agent_type: Type of agent ('prey' or 'predator')
            x, y: Position to check

        Returns:
            Dictionary with behavior modifiers, or None if not on island.
            For prey: {'speed_multiplier': float, 'reproduction_multiplier': float}
            For predator: {'speed_multiplier': float, 'hunger_multiplier': float, 'reproduction_multiplier': float}
        """
        if not self.is_on_island(x, y):
            return None

        if agent_type == 'prey':
            return {
                'speed_multiplier': ISLAND_PREY_SPEED_MULTIPLIER,
                'reproduction_multiplier': ISLAND_PREY_REPRODUCTION_MULTIPLIER,
            }
        elif agent_type == 'predator':
            return {
                'speed_multiplier': ISLAND_PRED_SPEED_MULTIPLIER,
                'hunger_multiplier': ISLAND_PRED_HUNGER_MULTIPLIER,
                'reproduction_multiplier': ISLAND_PRED_REPRODUCTION_MULTIPLIER,
            }
        else:
            return None

    def get_render_data(self):
        """
        Get data for rendering the river.

        Returns:
            Dictionary with river rendering info
        """
        if not self.enabled:
            return None

        return {
            'path_x': self.path_x,
            'path_y': self.path_y,
            'width': self.river_width,
            'split': self.split,
            'split_start': self.split_start,
            'split_end': self.split_end,
            'island_width': self.island_width,
        }
