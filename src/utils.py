"""Shared utilities for HUNT simulation."""

import numpy as np
import torch
from typing import Tuple, Union, Literal


def toroidal_distance_numpy(
    pos1: np.ndarray,
    pos2: np.ndarray,
    world_width: float,
    world_height: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute toroidal (wrap-around) distances using NumPy.

    Args:
        pos1: Reference position (2,) or batch (N, 2)
        pos2: Target positions (M, 2)
        world_width: World width for wrapping
        world_height: World height for wrapping

    Returns:
        distances: (M,) or (N, M) array of distances
        vectors: Direction vectors from pos1 to pos2
    """
    if pos1.ndim == 1:
        pos1 = pos1.reshape(1, 2)
        squeeze = True
    else:
        squeeze = False

    # Compute raw differences
    dx = pos2[:, 0] - pos1[:, 0:1]  # (N, M) or (1, M)
    dy = pos2[:, 1] - pos1[:, 1:2]

    # Apply toroidal wrapping
    dx = np.where(np.abs(dx) > world_width / 2,
                  dx - np.sign(dx) * world_width, dx)
    dy = np.where(np.abs(dy) > world_height / 2,
                  dy - np.sign(dy) * world_height, dy)

    distances = np.sqrt(dx**2 + dy**2)
    vectors = np.stack([dx, dy], axis=-1)

    if squeeze:
        distances = distances.squeeze(0)
        vectors = vectors.squeeze(0)

    return distances, vectors


def bounded_distance_numpy(
    pos1: np.ndarray,
    pos2: np.ndarray,
    world_width: float,
    world_height: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute distances in bounded (walled) world using NumPy.

    Same interface as toroidal_distance_numpy but without wrapping.
    """
    if pos1.ndim == 1:
        pos1 = pos1.reshape(1, 2)
        squeeze = True
    else:
        squeeze = False

    dx = pos2[:, 0] - pos1[:, 0:1]
    dy = pos2[:, 1] - pos1[:, 1:2]

    distances = np.sqrt(dx**2 + dy**2)
    vectors = np.stack([dx, dy], axis=-1)

    if squeeze:
        distances = distances.squeeze(0)
        vectors = vectors.squeeze(0)

    return distances, vectors


def toroidal_distance_torch(
    pos1: torch.Tensor,
    pos2: torch.Tensor,
    world_width: float,
    world_height: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute toroidal distances using PyTorch (GPU-compatible).

    Args:
        pos1: Reference positions (N, 2)
        pos2: Target positions (M, 2)

    Returns:
        distances: (N, M) tensor of distances
        vectors: (N, M, 2) direction vectors
    """
    # Broadcast to (N, M, 2)
    diff = pos1.unsqueeze(1) - pos2.unsqueeze(0)

    # Apply toroidal wrapping
    diff[..., 0] = torch.where(
        torch.abs(diff[..., 0]) > world_width / 2,
        diff[..., 0] - torch.sign(diff[..., 0]) * world_width,
        diff[..., 0]
    )
    diff[..., 1] = torch.where(
        torch.abs(diff[..., 1]) > world_height / 2,
        diff[..., 1] - torch.sign(diff[..., 1]) * world_height,
        diff[..., 1]
    )

    distances = torch.norm(diff, dim=2)
    return distances, diff


def bounded_distance_torch(
    pos1: torch.Tensor,
    pos2: torch.Tensor,
    world_width: float,
    world_height: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute distances in bounded world using PyTorch."""
    diff = pos1.unsqueeze(1) - pos2.unsqueeze(0)
    distances = torch.norm(diff, dim=2)
    return distances, diff


def spawn_offset(
    count: int,
    min_distance: float = 20.0,
    max_distance: float = 150.0,
    framework: Literal['numpy', 'torch'] = 'numpy',
    device: str = 'cpu'
) -> Union[np.ndarray, torch.Tensor]:
    """Generate random spawn offsets for reproduction.

    Args:
        count: Number of offsets to generate
        min_distance: Minimum spawn distance from parent
        max_distance: Maximum spawn distance from parent
        framework: 'numpy' or 'torch'
        device: PyTorch device (only used if framework='torch')

    Returns:
        Array/Tensor of shape (count, 2) with x,y offsets
    """
    if framework == 'numpy':
        distance = np.random.uniform(min_distance, max_distance, size=count)
        angle = np.random.uniform(0, 2 * np.pi, size=count)
        offset_x = distance * np.cos(angle)
        offset_y = distance * np.sin(angle)
        return np.column_stack([offset_x, offset_y])
    else:
        distance = torch.rand(count, device=device) * (max_distance - min_distance) + min_distance
        angle = torch.rand(count, device=device) * 2 * torch.pi
        offset_x = distance * torch.cos(angle)
        offset_y = distance * torch.sin(angle)
        return torch.stack([offset_x, offset_y], dim=1)


def wrap_position_numpy(
    pos: np.ndarray,
    world_width: float,
    world_height: float
) -> np.ndarray:
    """Wrap positions to stay within toroidal world bounds."""
    pos = pos.copy()
    pos[..., 0] = pos[..., 0] % world_width
    pos[..., 1] = pos[..., 1] % world_height
    return pos


def clamp_position_numpy(
    pos: np.ndarray,
    world_width: float,
    world_height: float,
    margin: float = 0.0
) -> np.ndarray:
    """Clamp positions to stay within bounded world."""
    pos = pos.copy()
    pos[..., 0] = np.clip(pos[..., 0], margin, world_width - margin)
    pos[..., 1] = np.clip(pos[..., 1], margin, world_height - margin)
    return pos


def wrap_position_torch(
    pos: torch.Tensor,
    world_width: float,
    world_height: float
) -> torch.Tensor:
    """Wrap positions for toroidal world (GPU)."""
    pos = pos.clone()
    pos[..., 0] = pos[..., 0] % world_width
    pos[..., 1] = pos[..., 1] % world_height
    return pos


def clamp_position_torch(
    pos: torch.Tensor,
    world_width: float,
    world_height: float,
    margin: float = 0.0
) -> torch.Tensor:
    """Clamp positions for bounded world (GPU)."""
    pos = pos.clone()
    pos[..., 0] = pos[..., 0].clamp(margin, world_width - margin)
    pos[..., 1] = pos[..., 1].clamp(margin, world_height - margin)
    return pos
