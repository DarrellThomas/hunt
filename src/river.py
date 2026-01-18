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

    def get_river_polygons(self):
        """
        Generate polygons for smooth river rendering.

        Returns:
            dict with keys:
                'river_polygon': List of (x,y) points forming river outline
                'island_polygon': List of (x,y) points forming island outline (or None)
        """
        if not self.enabled:
            return None

        # Calculate perpendicular offsets at each path point
        river_left_bank = []
        river_right_bank = []

        for i in range(len(self.path_x)):
            # Get tangent direction at this point
            if i == 0:
                dx = self.path_x[1] - self.path_x[0]
                dy = self.path_y[1] - self.path_y[0]
            elif i == len(self.path_x) - 1:
                dx = self.path_x[-1] - self.path_x[-2]
                dy = self.path_y[-1] - self.path_y[-2]
            else:
                # Average of forward and backward tangents for smoothness
                dx = (self.path_x[i+1] - self.path_x[i-1]) / 2
                dy = (self.path_y[i+1] - self.path_y[i-1]) / 2

            # Normalize
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx, dy = dx/length, dy/length

            # Perpendicular vector (rotate 90 degrees)
            perp_x, perp_y = -dy, dx

            # Calculate bank positions with smooth transitions
            t = i / len(self.path_x)

            # Determine if we're in or near split region
            if self.split:
                # Smooth transition zones
                transition_width = 0.05  # 5% of path for smooth transition

                if t < self.split_start - transition_width:
                    # Before split - normal river
                    split_factor = 0.0
                elif t < self.split_start:
                    # Transition into split
                    split_factor = (t - (self.split_start - transition_width)) / transition_width
                elif t <= self.split_end:
                    # Full split region
                    split_factor = 1.0
                elif t < self.split_end + transition_width:
                    # Transition out of split
                    split_factor = 1.0 - (t - self.split_end) / transition_width
                else:
                    # After split - normal river
                    split_factor = 0.0

                # Interpolate between normal and split widths
                normal_half_width = self.river_width / 2
                split_half_width = normal_half_width + self.island_width / 2

                effective_half_width = normal_half_width + split_factor * (split_half_width - normal_half_width)
            else:
                effective_half_width = self.river_width / 2

            # Calculate bank positions
            river_left_bank.append((
                self.path_x[i] + perp_x * effective_half_width,
                self.path_y[i] + perp_y * effective_half_width
            ))
            river_right_bank.append((
                self.path_x[i] - perp_x * effective_half_width,
                self.path_y[i] - perp_y * effective_half_width
            ))

        # Create closed polygon: left bank forward, right bank backward
        river_polygon = river_left_bank + river_right_bank[::-1]

        # Generate island polygon if split enabled
        island_polygon = None
        if self.split:
            island_polygon = self._generate_island_polygon()

        return {
            'river_polygon': river_polygon,
            'island_polygon': island_polygon
        }

    def _generate_island_polygon(self):
        """Generate smooth island outline as a single polygon with tapered ends."""
        island_points = []

        # Find indices for split region
        start_idx = int(self.split_start * len(self.path_x))
        end_idx = int(self.split_end * len(self.path_x))

        # Ensure we have enough points
        if end_idx <= start_idx:
            return None

        # Generate points along top edge of island (left to right)
        for i in range(start_idx, end_idx + 1):
            # Get perpendicular direction
            if i == start_idx:
                dx = self.path_x[i+1] - self.path_x[i]
                dy = self.path_y[i+1] - self.path_y[i]
            elif i == end_idx or i >= len(self.path_x) - 1:
                dx = self.path_x[min(i, len(self.path_x)-1)] - self.path_x[min(i-1, len(self.path_x)-2)]
                dy = self.path_y[min(i, len(self.path_y)-1)] - self.path_y[min(i-1, len(self.path_y)-2)]
            else:
                dx = (self.path_x[i+1] - self.path_x[i-1]) / 2
                dy = (self.path_y[i+1] - self.path_y[i-1]) / 2

            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx, dy = dx/length, dy/length
            perp_x, perp_y = -dy, dx

            # Taper island at ends for smooth transition
            t_local = (i - start_idx) / max(1, (end_idx - start_idx))  # 0 to 1 within island
            taper = np.sin(t_local * np.pi)  # 0 at ends, 1 in middle

            half_island = (self.island_width / 2) * taper

            island_points.append((
                self.path_x[i] + perp_x * half_island,
                self.path_y[i] + perp_y * half_island
            ))

        # Generate points along bottom edge (right to left)
        for i in range(end_idx, start_idx - 1, -1):
            if i == start_idx:
                dx = self.path_x[i+1] - self.path_x[i]
                dy = self.path_y[i+1] - self.path_y[i]
            elif i == end_idx or i >= len(self.path_x) - 1:
                dx = self.path_x[min(i, len(self.path_x)-1)] - self.path_x[min(i-1, len(self.path_x)-2)]
                dy = self.path_y[min(i, len(self.path_y)-1)] - self.path_y[min(i-1, len(self.path_y)-2)]
            else:
                dx = (self.path_x[i+1] - self.path_x[i-1]) / 2
                dy = (self.path_y[i+1] - self.path_y[i-1]) / 2

            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx, dy = dx/length, dy/length
            perp_x, perp_y = -dy, dx

            t_local = (i - start_idx) / max(1, (end_idx - start_idx))
            taper = np.sin(t_local * np.pi)
            half_island = (self.island_width / 2) * taper

            island_points.append((
                self.path_x[i] - perp_x * half_island,
                self.path_y[i] - perp_y * half_island
            ))

        return island_points

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
