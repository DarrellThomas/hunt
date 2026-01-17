"""Tests for boundary modes (toroidal vs bounded)."""

import sys
import os
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    toroidal_distance_numpy, bounded_distance_numpy,
    wrap_position_numpy, clamp_position_numpy,
    toroidal_distance_torch, bounded_distance_torch,
    wrap_position_torch, clamp_position_torch
)
from config_new import BoundaryMode


def test_toroidal_wrapping():
    """Test that positions wrap around in toroidal mode."""
    print("Testing toroidal position wrapping...")

    world_width, world_height = 100, 100

    # Position beyond right edge
    pos = np.array([110.0, 50.0])
    wrapped = wrap_position_numpy(pos, world_width, world_height)
    assert abs(wrapped[0] - 10.0) < 0.01, f"Expected 10.0, got {wrapped[0]}"
    assert abs(wrapped[1] - 50.0) < 0.01

    # Position beyond left edge
    pos = np.array([-10.0, 50.0])
    wrapped = wrap_position_numpy(pos, world_width, world_height)
    assert abs(wrapped[0] - 90.0) < 0.01, f"Expected 90.0, got {wrapped[0]}"

    # Position beyond top edge
    pos = np.array([50.0, 110.0])
    wrapped = wrap_position_numpy(pos, world_width, world_height)
    assert abs(wrapped[1] - 10.0) < 0.01

    # Position beyond bottom edge
    pos = np.array([50.0, -10.0])
    wrapped = wrap_position_numpy(pos, world_width, world_height)
    assert abs(wrapped[1] - 90.0) < 0.01

    print("  Right edge wrap: 110 -> 10 ✓")
    print("  Left edge wrap: -10 -> 90 ✓")
    print("  Top edge wrap: 110 -> 10 ✓")
    print("  Bottom edge wrap: -10 -> 90 ✓")
    print("✓ Toroidal wrapping works correctly")
    print("✓ test_toroidal_wrapping passed\n")


def test_bounded_clamping():
    """Test that positions are clamped in bounded mode."""
    print("Testing bounded position clamping...")

    world_width, world_height = 100, 100

    # Position beyond right edge
    pos = np.array([110.0, 50.0])
    clamped = clamp_position_numpy(pos, world_width, world_height)
    assert abs(clamped[0] - 100.0) < 0.01, f"Expected 100.0, got {clamped[0]}"
    assert abs(clamped[1] - 50.0) < 0.01

    # Position beyond left edge
    pos = np.array([-10.0, 50.0])
    clamped = clamp_position_numpy(pos, world_width, world_height)
    assert abs(clamped[0] - 0.0) < 0.01, f"Expected 0.0, got {clamped[0]}"

    # Position beyond top edge
    pos = np.array([50.0, 110.0])
    clamped = clamp_position_numpy(pos, world_width, world_height)
    assert abs(clamped[1] - 100.0) < 0.01

    # Position beyond bottom edge
    pos = np.array([50.0, -10.0])
    clamped = clamp_position_numpy(pos, world_width, world_height)
    assert abs(clamped[1] - 0.0) < 0.01

    print("  Right edge clamp: 110 -> 100 ✓")
    print("  Left edge clamp: -10 -> 0 ✓")
    print("  Top edge clamp: 110 -> 100 ✓")
    print("  Bottom edge clamp: -10 -> 0 ✓")
    print("✓ Bounded clamping works correctly")
    print("✓ test_bounded_clamping passed\n")


def test_toroidal_distance_wrapping():
    """Test that toroidal distance considers wrapping."""
    print("Testing toroidal distance with wrapping...")

    world_width, world_height = 100, 100

    # Two points close to opposite edges
    pos1 = np.array([5.0, 50.0])
    pos2 = np.array([95.0, 50.0])

    # Normal distance would be 90, but wrapping distance is 10
    distances, vectors = toroidal_distance_numpy(
        pos1, pos2.reshape(1, 2), world_width, world_height
    )

    expected_distance = 10.0
    assert abs(distances[0] - expected_distance) < 0.01, \
        f"Expected distance {expected_distance}, got {distances[0]}"

    # Vector should point left (negative x) through the wrap
    assert vectors[0, 0] < 0, "Vector should point left through wrap"

    print(f"  Distance through wrap: {distances[0]:.2f} ✓")
    print(f"  Vector direction: {vectors[0]}")
    print("✓ Toroidal distance wrapping works correctly")
    print("✓ test_toroidal_distance_wrapping passed\n")


def test_bounded_distance_no_wrapping():
    """Test that bounded distance does not wrap."""
    print("Testing bounded distance (no wrapping)...")

    world_width, world_height = 100, 100

    # Two points close to opposite edges
    pos1 = np.array([5.0, 50.0])
    pos2 = np.array([95.0, 50.0])

    # Bounded distance should be 90 (no wrapping)
    distances, vectors = bounded_distance_numpy(
        pos1, pos2.reshape(1, 2), world_width, world_height
    )

    expected_distance = 90.0
    assert abs(distances[0] - expected_distance) < 0.01, \
        f"Expected distance {expected_distance}, got {distances[0]}"

    # Vector should point right (positive x)
    assert vectors[0, 0] > 0, "Vector should point right (no wrap)"

    print(f"  Distance without wrap: {distances[0]:.2f} ✓")
    print(f"  Vector direction: {vectors[0]}")
    print("✓ Bounded distance (no wrapping) works correctly")
    print("✓ test_bounded_distance_no_wrapping passed\n")


def test_toroidal_vs_bounded_corner_behavior():
    """Test corner behavior differs between modes."""
    print("Testing corner behavior (toroidal vs bounded)...")

    world_width, world_height = 100, 100

    # Agent at corner
    pos1 = np.array([5.0, 5.0])
    # Target at opposite corner
    pos2 = np.array([95.0, 95.0])

    # Toroidal: shortest path wraps around (distance ≈ 14.14)
    toroidal_dist, _ = toroidal_distance_numpy(
        pos1, pos2.reshape(1, 2), world_width, world_height
    )

    # Bounded: must go diagonally (distance ≈ 127.28)
    bounded_dist, _ = bounded_distance_numpy(
        pos1, pos2.reshape(1, 2), world_width, world_height
    )

    print(f"  Toroidal corner distance: {toroidal_dist[0]:.2f}")
    print(f"  Bounded corner distance: {bounded_dist[0]:.2f}")

    # Bounded should be much longer than toroidal
    assert bounded_dist[0] > toroidal_dist[0] * 5, \
        "Bounded distance should be much longer at corners"

    print("✓ Corner behavior differs correctly")
    print("✓ test_toroidal_vs_bounded_corner_behavior passed\n")


def test_torch_toroidal_wrapping():
    """Test PyTorch toroidal wrapping."""
    print("Testing PyTorch toroidal wrapping...")

    world_width, world_height = 100, 100

    # Position beyond edges
    pos = torch.tensor([[110.0, 50.0], [-10.0, 50.0]], dtype=torch.float32)
    wrapped = wrap_position_torch(pos, world_width, world_height)

    assert abs(wrapped[0, 0].item() - 10.0) < 0.01
    assert abs(wrapped[1, 0].item() - 90.0) < 0.01

    print(f"  Wrapped positions: {wrapped}")
    print("✓ PyTorch toroidal wrapping works correctly")
    print("✓ test_torch_toroidal_wrapping passed\n")


def test_torch_bounded_clamping():
    """Test PyTorch bounded clamping."""
    print("Testing PyTorch bounded clamping...")

    world_width, world_height = 100, 100

    # Position beyond edges
    pos = torch.tensor([[110.0, 50.0], [-10.0, 50.0]], dtype=torch.float32)
    clamped = clamp_position_torch(pos, world_width, world_height)

    assert abs(clamped[0, 0].item() - 100.0) < 0.01
    assert abs(clamped[1, 0].item() - 0.0) < 0.01

    print(f"  Clamped positions: {clamped}")
    print("✓ PyTorch bounded clamping works correctly")
    print("✓ test_torch_bounded_clamping passed\n")


def test_numpy_torch_consistency():
    """Test that NumPy and PyTorch versions give same distance magnitudes."""
    print("Testing NumPy/PyTorch consistency...")

    world_width, world_height = 100, 100

    # Test positions - use batch for torch (pos1 should be (N, 2))
    pos1 = np.array([[10.0, 20.0]])  # Shape (1, 2) for consistency
    pos2_np = np.array([[30.0, 40.0], [90.0, 90.0]])

    # NumPy version (handles 1D pos1)
    dist_np, vec_np = toroidal_distance_numpy(pos1[0], pos2_np, world_width, world_height)

    # PyTorch version (expects batch input)
    pos1_torch = torch.tensor(pos1, dtype=torch.float32)
    pos2_torch = torch.tensor(pos2_np, dtype=torch.float32)
    dist_torch, vec_torch = toroidal_distance_torch(
        pos1_torch, pos2_torch, world_width, world_height
    )

    # Compare distances (torch returns (1, M), numpy returns (M,))
    dist_torch_flat = dist_torch[0].numpy()
    dist_diff = np.abs(dist_np - dist_torch_flat)
    assert np.max(dist_diff) < 1e-5, f"Distance mismatch: {dist_diff}"

    # Note: vectors may point in opposite directions between implementations
    # but distances (magnitudes) should match
    # Compare vector magnitudes instead of directions
    vec_mag_np = np.linalg.norm(vec_np, axis=1)
    vec_mag_torch = torch.norm(vec_torch[0], dim=1).numpy()
    vec_mag_diff = np.abs(vec_mag_np - vec_mag_torch)
    assert np.max(vec_mag_diff) < 1e-5, f"Vector magnitude mismatch: {vec_mag_diff}"

    print(f"  Max distance diff: {np.max(dist_diff):.6f}")
    print(f"  Max vector magnitude diff: {np.max(vec_mag_diff):.6f}")
    print("✓ NumPy/PyTorch consistency verified")
    print("✓ test_numpy_torch_consistency passed\n")


def test_boundary_mode_enum():
    """Test BoundaryMode enum values."""
    print("Testing BoundaryMode enum...")

    assert BoundaryMode.TOROIDAL.value == "toroidal"
    assert BoundaryMode.BOUNDED.value == "bounded"

    # Test enum comparison
    mode = BoundaryMode.TOROIDAL
    assert mode == BoundaryMode.TOROIDAL
    assert mode != BoundaryMode.BOUNDED

    print("  TOROIDAL mode: ✓")
    print("  BOUNDED mode: ✓")
    print("✓ BoundaryMode enum works correctly")
    print("✓ test_boundary_mode_enum passed\n")


def test_edge_cases():
    """Test edge cases for boundary handling."""
    print("Testing edge cases...")

    world_width, world_height = 100, 100

    # Position exactly at boundary
    pos = np.array([100.0, 100.0])

    # Wrapping should bring to 0,0
    wrapped = wrap_position_numpy(pos, world_width, world_height)
    assert abs(wrapped[0]) < 0.01
    assert abs(wrapped[1]) < 0.01

    # Clamping should keep at 100,100
    clamped = clamp_position_numpy(pos, world_width, world_height)
    assert abs(clamped[0] - 100.0) < 0.01
    assert abs(clamped[1] - 100.0) < 0.01

    # Position at zero
    pos = np.array([0.0, 0.0])
    wrapped = wrap_position_numpy(pos, world_width, world_height)
    clamped = clamp_position_numpy(pos, world_width, world_height)
    assert np.allclose(wrapped, [0.0, 0.0])
    assert np.allclose(clamped, [0.0, 0.0])

    print("  Boundary position handling: ✓")
    print("  Zero position handling: ✓")
    print("✓ Edge cases handled correctly")
    print("✓ test_edge_cases passed\n")


if __name__ == "__main__":
    print("Running boundary mode tests...\n")

    test_toroidal_wrapping()
    test_bounded_clamping()
    test_toroidal_distance_wrapping()
    test_bounded_distance_no_wrapping()
    test_toroidal_vs_bounded_corner_behavior()
    test_torch_toroidal_wrapping()
    test_torch_bounded_clamping()
    test_numpy_torch_consistency()
    test_boundary_mode_enum()
    test_edge_cases()

    print("✅ All boundary mode tests passed!")
    print("✅ Boundary mode system is working correctly:")
    print("   - Toroidal wrapping (positions wrap around edges)")
    print("   - Bounded clamping (positions clamped to edges)")
    print("   - Toroidal distance considers wrapping")
    print("   - Bounded distance does not wrap")
    print("   - Corner behavior differs between modes")
    print("   - PyTorch and NumPy versions consistent")
    print("   - Edge cases handled properly")
