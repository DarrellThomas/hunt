"""
GPU-resident river system for the HUNT ecosystem.
All river calculations performed on GPU to eliminate CPU-GPU transfers.
"""

import torch
import numpy as np
from config import *


class RiverGPU:
    """
    GPU-accelerated river with optional island splitting.
    All calculations performed in PyTorch on GPU for maximum performance.
    """

    def __init__(self, world_width, world_height, device='cuda'):
        """
        Initialize the GPU river.

        Args:
            world_width: Width of the world
            world_height: Height of the world
            device: PyTorch device ('cuda' or 'cpu')
        """
        self.width = world_width
        self.height = world_height
        self.device = device
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

        # Generate river path (on CPU first)
        self._generate_path()

    def _generate_path(self):
        """Generate the river centerline path and transfer to GPU."""
        # Create path points from left to right (same as CPU version)
        num_points = 50
        t = np.linspace(0, 1, num_points)

        # X goes from 0 to world_width
        path_x = t * self.width

        # Y starts at middle, curves based on curviness
        base_y = self.height / 2

        # Use sine waves for natural curves
        curve = np.sin(t * np.pi * 4) * self.curviness * self.height * 0.2
        curve += np.sin(t * np.pi * 2.3) * self.curviness * self.height * 0.15

        path_y = base_y + curve

        # Clamp to world bounds
        path_y = np.clip(path_y, self.river_width, self.height - self.river_width)

        # Calculate flow direction at each point (tangent to path)
        dx = np.gradient(path_x)
        dy = np.gradient(path_y)

        # Normalize to unit vectors
        mag = np.sqrt(dx**2 + dy**2)
        flow_dir_x = dx / mag
        flow_dir_y = dy / mag

        # Transfer to GPU as tensors
        self.path_x = torch.tensor(path_x, dtype=torch.float32, device=self.device)
        self.path_y = torch.tensor(path_y, dtype=torch.float32, device=self.device)
        self.flow_dir_x = torch.tensor(flow_dir_x, dtype=torch.float32, device=self.device)
        self.flow_dir_y = torch.tensor(flow_dir_y, dtype=torch.float32, device=self.device)

        # Create path points tensor for distance calculations (num_points x 2)
        self.path_points = torch.stack([self.path_x, self.path_y], dim=1)

        # Store num_points for indexing
        self.num_points = len(path_x)

    def get_flow_at_batch_gpu(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Get flow velocity for multiple positions (GPU-resident).

        Args:
            positions: (N, 2) tensor of positions on GPU

        Returns:
            (N, 2) tensor of flow velocities
        """
        if not self.enabled or len(positions) == 0:
            return torch.zeros_like(positions, device=self.device)

        # Check if on island first (island has no flow)
        on_island = self.is_on_island_batch_gpu(positions)

        # Check if in river
        in_river = self.is_in_river_batch_gpu(positions)

        # Compute distances to all path points (N x num_points)
        distances = torch.cdist(positions, self.path_points)

        # Find nearest path point for each position (N,)
        nearest_idx = torch.argmin(distances, dim=1)

        # Get flow directions at nearest points (N, 2)
        flow_x = self.flow_dir_x[nearest_idx] * self.flow_speed
        flow_y = self.flow_dir_y[nearest_idx] * self.flow_speed
        flows = torch.stack([flow_x, flow_y], dim=1)

        # Zero out flow for positions not in river or on island
        mask = in_river & ~on_island
        flows = flows * mask.unsqueeze(1).float()

        return flows

    def is_in_river_batch_gpu(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Check if positions are in the river (GPU-resident).

        Args:
            positions: (N, 2) tensor of positions on GPU

        Returns:
            (N,) boolean tensor indicating if each position is in river
        """
        if not self.enabled or len(positions) == 0:
            return torch.zeros(len(positions), dtype=torch.bool, device=self.device)

        # Compute distances to all path points (N x num_points)
        distances = torch.cdist(positions, self.path_points)

        # Find nearest path point for each position
        dist_to_nearest, nearest_idx = torch.min(distances, dim=1)

        # Get t parameter (position along river, 0 to 1)
        t = nearest_idx.float() / self.num_points

        if self.split:
            # Check if in split region
            in_split_region = (t >= self.split_start) & (t <= self.split_end)

            # Get center y at nearest point
            center_y = self.path_y[nearest_idx]
            offset = self.island_width / 2

            # Distance from center line
            y = positions[:, 1]
            dist_from_center = torch.abs(y - center_y)

            # Check if on island (between channels) - NO FLOW HERE
            on_island = dist_from_center < offset

            # Top channel (above island)
            top_channel_center = center_y - offset
            in_top = torch.abs(y - top_channel_center) < self.river_width / 2

            # Bottom channel (below island)
            bottom_channel_center = center_y + offset
            in_bottom = torch.abs(y - bottom_channel_center) < self.river_width / 2

            # In river if (in top or bottom channel) and NOT on island
            in_river_split = (in_top | in_bottom) & ~on_island

            # Not in split region - normal river check
            in_river_normal = dist_to_nearest < self.river_width / 2

            # Combine: use split logic where applicable, normal otherwise
            in_river = torch.where(in_split_region, in_river_split, in_river_normal)

            return in_river
        else:
            # No split - simple distance check
            return dist_to_nearest < self.river_width / 2

    def is_on_island_batch_gpu(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Check if positions are on the island (GPU-resident).

        Args:
            positions: (N, 2) tensor of positions on GPU

        Returns:
            (N,) boolean tensor indicating if each position is on island
        """
        if not self.enabled or not self.split or len(positions) == 0:
            return torch.zeros(len(positions), dtype=torch.bool, device=self.device)

        # Compute distances to all path points (N x num_points)
        distances = torch.cdist(positions, self.path_points)

        # Find nearest path point for each position
        nearest_idx = torch.argmin(distances, dim=1)

        # Get t parameter (position along river, 0 to 1)
        t = nearest_idx.float() / self.num_points

        # Check if in split region
        in_split_region = (t >= self.split_start) & (t <= self.split_end)

        # Get center y at nearest point
        center_y = self.path_y[nearest_idx]
        offset = self.island_width / 2

        # Distance from center line
        y = positions[:, 1]
        dist_from_center = torch.abs(y - center_y)

        # On island if within island bounds AND in split region
        on_island = (dist_from_center < offset) & in_split_region

        return on_island

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

        # Convert GPU tensors to CPU numpy arrays for polygon generation
        path_x = self.path_x.cpu().numpy()
        path_y = self.path_y.cpu().numpy()

        # Calculate perpendicular offsets at each path point
        river_left_bank = []
        river_right_bank = []

        for i in range(len(path_x)):
            # Get tangent direction at this point
            if i == 0:
                dx = path_x[1] - path_x[0]
                dy = path_y[1] - path_y[0]
            elif i == len(path_x) - 1:
                dx = path_x[-1] - path_x[-2]
                dy = path_y[-1] - path_y[-2]
            else:
                # Average of forward and backward tangents for smoothness
                dx = (path_x[i+1] - path_x[i-1]) / 2
                dy = (path_y[i+1] - path_y[i-1]) / 2

            # Normalize
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx, dy = dx/length, dy/length

            # Perpendicular vector (rotate 90 degrees)
            perp_x, perp_y = -dy, dx

            # Calculate bank positions with smooth transitions
            t = i / len(path_x)

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
                path_x[i] + perp_x * effective_half_width,
                path_y[i] + perp_y * effective_half_width
            ))
            river_right_bank.append((
                path_x[i] - perp_x * effective_half_width,
                path_y[i] - perp_y * effective_half_width
            ))

        # Create closed polygon: left bank forward, right bank backward
        river_polygon = river_left_bank + river_right_bank[::-1]

        # Generate island polygon if split enabled
        island_polygon = None
        if self.split:
            island_polygon = self._generate_island_polygon(path_x, path_y)

        return {
            'river_polygon': river_polygon,
            'island_polygon': island_polygon
        }

    def _generate_island_polygon(self, path_x, path_y):
        """Generate smooth island outline as a single polygon with tapered ends."""
        island_points = []

        # Find indices for split region
        start_idx = int(self.split_start * len(path_x))
        end_idx = int(self.split_end * len(path_x))

        # Ensure we have enough points
        if end_idx <= start_idx:
            return None

        # Generate points along top edge of island (left to right)
        for i in range(start_idx, end_idx + 1):
            # Get perpendicular direction
            if i == start_idx:
                dx = path_x[i+1] - path_x[i]
                dy = path_y[i+1] - path_y[i]
            elif i == end_idx or i >= len(path_x) - 1:
                dx = path_x[min(i, len(path_x)-1)] - path_x[min(i-1, len(path_x)-2)]
                dy = path_y[min(i, len(path_y)-1)] - path_y[min(i-1, len(path_y)-2)]
            else:
                dx = (path_x[i+1] - path_x[i-1]) / 2
                dy = (path_y[i+1] - path_y[i-1]) / 2

            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx, dy = dx/length, dy/length
            perp_x, perp_y = -dy, dx

            # Taper island at ends for smooth transition
            t_local = (i - start_idx) / max(1, (end_idx - start_idx))  # 0 to 1 within island
            taper = np.sin(t_local * np.pi)  # 0 at ends, 1 in middle

            half_island = (self.island_width / 2) * taper

            island_points.append((
                path_x[i] + perp_x * half_island,
                path_y[i] + perp_y * half_island
            ))

        # Generate points along bottom edge (right to left)
        for i in range(end_idx, start_idx - 1, -1):
            if i == start_idx:
                dx = path_x[i+1] - path_x[i]
                dy = path_y[i+1] - path_y[i]
            elif i == end_idx or i >= len(path_x) - 1:
                dx = path_x[min(i, len(path_x)-1)] - path_x[min(i-1, len(path_x)-2)]
                dy = path_y[min(i, len(path_y)-1)] - path_y[min(i-1, len(path_y)-2)]
            else:
                dx = (path_x[i+1] - path_x[i-1]) / 2
                dy = (path_y[i+1] - path_y[i-1]) / 2

            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx, dy = dx/length, dy/length
            perp_x, perp_y = -dy, dx

            t_local = (i - start_idx) / max(1, (end_idx - start_idx))
            taper = np.sin(t_local * np.pi)
            half_island = (self.island_width / 2) * taper

            island_points.append((
                path_x[i] - perp_x * half_island,
                path_y[i] - perp_y * half_island
            ))

        return island_points

    def get_render_data(self):
        """
        Get data for rendering the river (transfer back to CPU/NumPy).

        Returns:
            Dictionary with river rendering info
        """
        if not self.enabled:
            return None

        return {
            'path_x': self.path_x.cpu().numpy(),
            'path_y': self.path_y.cpu().numpy(),
            'width': self.river_width,
            'split': self.split,
            'split_start': self.split_start,
            'split_end': self.split_end,
            'island_width': self.island_width,
        }
