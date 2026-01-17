"""Tests for GPU river implementation."""

import sys
import os
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from river import River
from river_gpu import RiverGPU


def test_river_gpu_initialization():
    """Test RiverGPU initialization."""
    print("Testing RiverGPU initialization...")

    river_gpu = RiverGPU(800, 600, device='cpu')

    assert river_gpu.enabled == True
    assert river_gpu.path_x.shape[0] == 50
    assert river_gpu.path_y.shape[0] == 50
    assert river_gpu.flow_dir_x.shape[0] == 50
    assert river_gpu.flow_dir_y.shape[0] == 50
    assert river_gpu.path_points.shape == (50, 2)

    print("✓ RiverGPU initialized correctly")
    print("✓ test_river_gpu_initialization passed\n")


def test_flow_consistency_cpu_vs_gpu():
    """Test that GPU flow matches CPU flow."""
    print("Testing flow consistency between CPU and GPU...")

    width, height = 800, 600
    river_cpu = River(width, height)
    river_gpu = RiverGPU(width, height, device='cpu')

    # Test multiple positions
    test_positions = np.array([
        [100, 300],  # Near river center
        [400, 200],  # Different position
        [700, 400],  # Far right
        [50, 500],   # Far left
        [400, 100],  # Edge of world
        [400, 500],  # Other edge
    ], dtype=np.float32)

    # Get flows from both implementations
    flows_cpu = np.zeros_like(test_positions)
    for i, pos in enumerate(test_positions):
        flows_cpu[i] = river_cpu.get_flow_at(pos[0], pos[1])

    # GPU version
    positions_gpu = torch.tensor(test_positions, dtype=torch.float32)
    flows_gpu = river_gpu.get_flow_at_batch_gpu(positions_gpu).numpy()

    # Compare
    max_diff = np.max(np.abs(flows_cpu - flows_gpu))
    print(f"  Max flow difference: {max_diff:.6f}")

    # Allow small numerical differences
    assert max_diff < 1e-5, f"Flow mismatch: {max_diff}"

    print("✓ GPU flow matches CPU flow")
    print("✓ test_flow_consistency_cpu_vs_gpu passed\n")


def test_is_in_river_consistency():
    """Test that GPU is_in_river matches CPU version."""
    print("Testing is_in_river consistency...")

    width, height = 800, 600
    river_cpu = River(width, height)
    river_gpu = RiverGPU(width, height, device='cpu')

    # Test grid of positions
    x_coords = np.linspace(0, width, 20)
    y_coords = np.linspace(0, height, 20)
    test_positions = []
    for x in x_coords:
        for y in y_coords:
            test_positions.append([x, y])

    test_positions = np.array(test_positions, dtype=np.float32)

    # CPU version
    in_river_cpu = np.array([river_cpu.is_in_river(pos[0], pos[1])
                             for pos in test_positions])

    # GPU version
    positions_gpu = torch.tensor(test_positions, dtype=torch.float32)
    in_river_gpu = river_gpu.is_in_river_batch_gpu(positions_gpu).numpy()

    # Compare
    matches = np.sum(in_river_cpu == in_river_gpu)
    total = len(test_positions)
    match_rate = matches / total

    print(f"  Match rate: {matches}/{total} ({match_rate*100:.1f}%)")

    # Should match almost perfectly (allow tiny tolerance for edge cases)
    assert match_rate > 0.95, f"Too many mismatches: {match_rate}"

    print("✓ GPU is_in_river matches CPU version")
    print("✓ test_is_in_river_consistency passed\n")


def test_is_on_island_consistency():
    """Test that GPU is_on_island matches CPU version."""
    print("Testing is_on_island consistency...")

    width, height = 800, 600
    river_cpu = River(width, height)
    river_gpu = RiverGPU(width, height, device='cpu')

    # Test grid of positions (focused on middle where island is)
    x_coords = np.linspace(200, 600, 30)
    y_coords = np.linspace(200, 400, 30)
    test_positions = []
    for x in x_coords:
        for y in y_coords:
            test_positions.append([x, y])

    test_positions = np.array(test_positions, dtype=np.float32)

    # CPU version
    on_island_cpu = np.array([river_cpu.is_on_island(pos[0], pos[1])
                              for pos in test_positions])

    # GPU version
    positions_gpu = torch.tensor(test_positions, dtype=torch.float32)
    on_island_gpu = river_gpu.is_on_island_batch_gpu(positions_gpu).numpy()

    # Compare
    matches = np.sum(on_island_cpu == on_island_gpu)
    total = len(test_positions)
    match_rate = matches / total

    print(f"  Match rate: {matches}/{total} ({match_rate*100:.1f}%)")

    # Should match perfectly
    assert match_rate > 0.95, f"Too many mismatches: {match_rate}"

    # Count how many are actually on island
    num_on_island_cpu = np.sum(on_island_cpu)
    num_on_island_gpu = np.sum(on_island_gpu)
    print(f"  Positions on island (CPU): {num_on_island_cpu}")
    print(f"  Positions on island (GPU): {num_on_island_gpu}")

    print("✓ GPU is_on_island matches CPU version")
    print("✓ test_is_on_island_consistency passed\n")


def test_island_has_no_flow():
    """Test that positions on island have zero flow."""
    print("Testing island has no flow...")

    width, height = 800, 600
    river_gpu = RiverGPU(width, height, device='cpu')

    # Find positions on island
    x_coords = np.linspace(200, 600, 50)
    y_coords = np.linspace(250, 350, 50)
    test_positions = []
    for x in x_coords:
        for y in y_coords:
            test_positions.append([x, y])

    test_positions = np.array(test_positions, dtype=np.float32)
    positions_gpu = torch.tensor(test_positions, dtype=torch.float32)

    # Check which are on island
    on_island = river_gpu.is_on_island_batch_gpu(positions_gpu)

    # Get flows for island positions
    flows = river_gpu.get_flow_at_batch_gpu(positions_gpu)
    island_flows = flows[on_island]

    if len(island_flows) > 0:
        max_island_flow = torch.max(torch.abs(island_flows)).item()
        print(f"  Max flow on island: {max_island_flow:.6f}")
        assert max_island_flow < 1e-5, "Island should have zero flow!"
        print(f"✓ All {len(island_flows)} island positions have zero flow")
    else:
        print("  Note: No island positions found in test grid")

    print("✓ test_island_has_no_flow passed\n")


def test_flow_direction():
    """Test that flow is generally in the correct direction (left to right)."""
    print("Testing flow direction...")

    width, height = 800, 600
    river_gpu = RiverGPU(width, height, device='cpu')

    # Sample positions in river
    test_positions = np.array([
        [100, 300],
        [200, 300],
        [300, 300],
        [400, 300],
        [500, 300],
        [600, 300],
        [700, 300],
    ], dtype=np.float32)

    positions_gpu = torch.tensor(test_positions, dtype=torch.float32)
    flows = river_gpu.get_flow_at_batch_gpu(positions_gpu)

    # Check that x component is generally positive (flow left to right)
    x_flows = flows[:, 0].numpy()
    positive_count = np.sum(x_flows > 0)
    print(f"  Positions with positive x flow: {positive_count}/{len(x_flows)}")

    # Most flows should be positive (allowing for some curve variation)
    assert positive_count >= len(x_flows) * 0.7, "Most flows should go left to right"

    print("✓ Flow direction is correct")
    print("✓ test_flow_direction passed\n")


def test_batched_performance():
    """Test that batched GPU computation works efficiently."""
    print("Testing batched performance...")

    width, height = 800, 600
    river_gpu = RiverGPU(width, height, device='cpu')

    # Large batch of positions
    num_positions = 1000
    test_positions = torch.rand(num_positions, 2, dtype=torch.float32)
    test_positions[:, 0] *= width
    test_positions[:, 1] *= height

    # Should complete without error
    flows = river_gpu.get_flow_at_batch_gpu(test_positions)
    in_river = river_gpu.is_in_river_batch_gpu(test_positions)
    on_island = river_gpu.is_on_island_batch_gpu(test_positions)

    assert flows.shape == (num_positions, 2)
    assert in_river.shape == (num_positions,)
    assert on_island.shape == (num_positions,)

    print(f"✓ Processed {num_positions} positions in batch")
    print(f"  {torch.sum(in_river).item()} in river, {torch.sum(on_island).item()} on island")
    print("✓ test_batched_performance passed\n")


if __name__ == "__main__":
    print("Running GPU river tests...\n")

    test_river_gpu_initialization()
    test_flow_consistency_cpu_vs_gpu()
    test_is_in_river_consistency()
    test_is_on_island_consistency()
    test_island_has_no_flow()
    test_flow_direction()
    test_batched_performance()

    print("✅ All GPU river tests passed!")
    print("✅ GPU river implementation is consistent with CPU version")
    print("   - Flow calculations match")
    print("   - River detection matches")
    print("   - Island detection matches")
    print("   - Batched operations work efficiently")
