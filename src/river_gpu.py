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
