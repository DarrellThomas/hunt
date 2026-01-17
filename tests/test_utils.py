"""Unit tests for shared utilities."""

import numpy as np
import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    toroidal_distance_numpy,
    bounded_distance_numpy,
    toroidal_distance_torch,
    bounded_distance_torch,
    spawn_offset,
    wrap_position_numpy,
    clamp_position_numpy,
    wrap_position_torch,
    clamp_position_torch
)


def test_toroidal_distance_numpy():
    """Test toroidal distance calculation with NumPy."""
    # Test case 1: Simple distance
    pos1 = np.array([10.0, 10.0])
    pos2 = np.array([[15.0, 15.0]])
    distances, vectors = toroidal_distance_numpy(pos1, pos2, 100, 100)
    expected_dist = np.sqrt(5**2 + 5**2)
    assert np.isclose(distances[0], expected_dist), f"Expected {expected_dist}, got {distances[0]}"

    # Test case 2: Wrapping (closest distance is across boundary)
    pos1 = np.array([5.0, 5.0])
    pos2 = np.array([[95.0, 95.0]])
    distances, vectors = toroidal_distance_numpy(pos1, pos2, 100, 100)
    # Wrapped distance should be sqrt(10^2 + 10^2) = 14.14
    expected_dist = np.sqrt(10**2 + 10**2)
    assert np.isclose(distances[0], expected_dist), f"Expected {expected_dist}, got {distances[0]}"

    # Test case 3: Empty positions
    pos1 = np.array([10.0, 10.0])
    pos2 = np.array([]).reshape(0, 2)
    distances, vectors = toroidal_distance_numpy(pos1, pos2, 100, 100)
    assert len(distances) == 0, "Expected empty array for empty positions"

    print("✓ test_toroidal_distance_numpy passed")


def test_bounded_distance_numpy():
    """Test bounded distance calculation with NumPy."""
    # Test case 1: Simple distance (no wrapping)
    pos1 = np.array([10.0, 10.0])
    pos2 = np.array([[15.0, 15.0]])
    distances, vectors = bounded_distance_numpy(pos1, pos2, 100, 100)
    expected_dist = np.sqrt(5**2 + 5**2)
    assert np.isclose(distances[0], expected_dist), f"Expected {expected_dist}, got {distances[0]}"

    # Test case 2: Large distance (no wrapping, unlike toroidal)
    pos1 = np.array([5.0, 5.0])
    pos2 = np.array([[95.0, 95.0]])
    distances, vectors = bounded_distance_numpy(pos1, pos2, 100, 100)
    # Bounded distance should be full sqrt(90^2 + 90^2) = 127.28
    expected_dist = np.sqrt(90**2 + 90**2)
    assert np.isclose(distances[0], expected_dist), f"Expected {expected_dist}, got {distances[0]}"

    print("✓ test_bounded_distance_numpy passed")


def test_toroidal_distance_torch():
    """Test toroidal distance calculation with PyTorch."""
    # Test case 1: Simple distance
    pos1 = torch.tensor([[10.0, 10.0]], dtype=torch.float32)
    pos2 = torch.tensor([[15.0, 15.0]], dtype=torch.float32)
    distances, vectors = toroidal_distance_torch(pos1, pos2, 100, 100)
    expected_dist = torch.tensor(np.sqrt(5**2 + 5**2), dtype=torch.float32)
    assert torch.isclose(distances[0, 0], expected_dist), f"Expected {expected_dist}, got {distances[0, 0]}"

    # Test case 2: Wrapping
    pos1 = torch.tensor([[5.0, 5.0]], dtype=torch.float32)
    pos2 = torch.tensor([[95.0, 95.0]], dtype=torch.float32)
    distances, vectors = toroidal_distance_torch(pos1, pos2, 100, 100)
    expected_dist = torch.tensor(np.sqrt(10**2 + 10**2), dtype=torch.float32)
    assert torch.isclose(distances[0, 0], expected_dist), f"Expected {expected_dist}, got {distances[0, 0]}"

    print("✓ test_toroidal_distance_torch passed")


def test_spawn_offset():
    """Test spawn offset generation."""
    # NumPy version
    offsets_np = spawn_offset(100, min_distance=20, max_distance=150, framework='numpy')
    assert offsets_np.shape == (100, 2), "Expected shape (100, 2)"
    distances_np = np.linalg.norm(offsets_np, axis=1)
    assert np.all(distances_np >= 20) and np.all(distances_np <= 150), "Distances out of range"

    # PyTorch version
    offsets_torch = spawn_offset(100, min_distance=20, max_distance=150, framework='torch', device='cpu')
    assert offsets_torch.shape == (100, 2), "Expected shape (100, 2)"
    distances_torch = torch.norm(offsets_torch, dim=1)
    assert torch.all(distances_torch >= 20) and torch.all(distances_torch <= 150), "Distances out of range"

    print("✓ test_spawn_offset passed")


def test_wrap_position():
    """Test position wrapping for toroidal world."""
    # NumPy version
    pos_np = np.array([[150.0, 250.0], [-10.0, -5.0]])
    wrapped = wrap_position_numpy(pos_np, 100, 200)
    assert np.allclose(wrapped, [[50.0, 50.0], [90.0, 195.0]]), f"Unexpected wrapping: {wrapped}"

    # PyTorch version
    pos_torch = torch.tensor([[150.0, 250.0], [-10.0, -5.0]])
    wrapped_torch = wrap_position_torch(pos_torch, 100, 200)
    expected_torch = torch.tensor([[50.0, 50.0], [90.0, 195.0]])
    assert torch.allclose(wrapped_torch, expected_torch), f"Unexpected wrapping: {wrapped_torch}"

    print("✓ test_wrap_position passed")


def test_clamp_position():
    """Test position clamping for bounded world."""
    # NumPy version
    pos_np = np.array([[150.0, 250.0], [-10.0, -5.0], [50.0, 100.0]])
    clamped = clamp_position_numpy(pos_np, 100, 200)
    assert np.allclose(clamped, [[100.0, 200.0], [0.0, 0.0], [50.0, 100.0]]), f"Unexpected clamping: {clamped}"

    # PyTorch version
    pos_torch = torch.tensor([[150.0, 250.0], [-10.0, -5.0], [50.0, 100.0]])
    clamped_torch = clamp_position_torch(pos_torch, 100, 200)
    expected_torch = torch.tensor([[100.0, 200.0], [0.0, 0.0], [50.0, 100.0]])
    assert torch.allclose(clamped_torch, expected_torch), f"Unexpected clamping: {clamped_torch}"

    print("✓ test_clamp_position passed")


def test_numpy_torch_consistency():
    """Test that NumPy and PyTorch versions produce consistent results."""
    pos1_np = np.array([25.0, 75.0], dtype=np.float32)
    pos2_np = np.array([[30.0, 80.0], [90.0, 10.0], [5.0, 95.0]], dtype=np.float32)

    pos1_torch = torch.tensor([[25.0, 75.0]], dtype=torch.float32)
    pos2_torch = torch.tensor([[30.0, 80.0], [90.0, 10.0], [5.0, 95.0]], dtype=torch.float32)

    # Test toroidal distances
    distances_np, vectors_np = toroidal_distance_numpy(pos1_np, pos2_np, 100, 100)
    distances_torch, vectors_torch = toroidal_distance_torch(pos1_torch, pos2_torch, 100, 100)

    distances_torch_np = distances_torch[0].numpy()
    assert np.allclose(distances_np, distances_torch_np, atol=1e-5), \
        f"NumPy and PyTorch toroidal distances don't match: {distances_np} vs {distances_torch_np}"

    print("✓ test_numpy_torch_consistency passed")


if __name__ == "__main__":
    print("Running utils.py unit tests...\n")

    test_toroidal_distance_numpy()
    test_bounded_distance_numpy()
    test_toroidal_distance_torch()
    test_spawn_offset()
    test_wrap_position()
    test_clamp_position()
    test_numpy_torch_consistency()

    print("\n✅ All utils.py tests passed!")
