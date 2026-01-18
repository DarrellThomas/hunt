#!/usr/bin/env python3
"""
Analyze saved brain weights from HUNT simulation.

Provides statistics and comparison tools for brain checkpoints.
"""

import argparse
import numpy as np
from pathlib import Path


def load_brain_file(filename):
    """Load brain checkpoint file."""
    if not Path(filename).exists():
        print(f"Error: {filename} not found!")
        return None

    return np.load(filename, allow_pickle=True)


def print_summary(data, verbose=False):
    """Print summary statistics for brain weights."""
    print(f"\n{'='*60}")
    print("BRAIN CHECKPOINT SUMMARY")
    print(f"{'='*60}")

    # Metadata
    if 'metadata' in data:
        metadata = data['metadata'].item()
        print(f"\nRun Information:")
        print(f"  Title: {metadata.get('run_title', 'Unknown')}")
        print(f"  Start Time: {metadata.get('start_time', 'Unknown')}")
        print(f"  Save Time: {metadata.get('save_time', 'Unknown')}")
        print(f"  World Size: {metadata.get('world_width', '?')}x{metadata.get('world_height', '?')}")
        print(f"  Device: {metadata.get('device', 'Unknown')}")

    # Population state
    print(f"\nPopulation State:")
    print(f"  Timestep: {data.get('timestep', 'Unknown')}")

    prey_alive = data['prey_alive']
    pred_alive = data['pred_alive']
    print(f"  Prey: {prey_alive.sum()} alive / {len(prey_alive)} capacity")
    print(f"  Predators: {pred_alive.sum()} alive / {len(pred_alive)} capacity")

    # Brain weights statistics
    prey_weights = data['prey_weights']
    pred_weights = data['pred_weights']

    print(f"\nPrey Brain Weights:")
    print(f"  Shape: {prey_weights.shape}")
    print(f"  Parameters per brain: {prey_weights.shape[1]:,}")
    print(f"  Mean: {prey_weights.mean():.4f}")
    print(f"  Std: {prey_weights.std():.4f}")
    print(f"  Range: [{prey_weights.min():.4f}, {prey_weights.max():.4f}]")

    print(f"\nPredator Brain Weights:")
    print(f"  Shape: {pred_weights.shape}")
    print(f"  Parameters per brain: {pred_weights.shape[1]:,}")
    print(f"  Mean: {pred_weights.mean():.4f}")
    print(f"  Std: {pred_weights.std():.4f}")
    print(f"  Range: [{pred_weights.min():.4f}, {pred_weights.max():.4f}]")

    # Evolvable traits
    if 'prey_swim_speed' in data:
        prey_swim = data['prey_swim_speed'][prey_alive]
        pred_swim = data['pred_swim_speed'][pred_alive]

        print(f"\nEvolved Traits (Alive Agents):")
        print(f"  Prey Swim Speed: {prey_swim.mean():.2f} ± {prey_swim.std():.2f}")
        print(f"    Range: [{prey_swim.min():.2f}, {prey_swim.max():.2f}]")
        print(f"  Predator Swim Speed: {pred_swim.mean():.2f} ± {pred_swim.std():.2f}")
        print(f"    Range: [{pred_swim.min():.2f}, {pred_swim.max():.2f}]")

    if verbose:
        # Additional detailed statistics
        print(f"\nDetailed Statistics:")

        if 'prey_age' in data:
            prey_age = data['prey_age'][prey_alive]
            print(f"  Prey Age: {prey_age.mean():.1f} ± {prey_age.std():.1f}")

        if 'pred_age' in data and 'pred_energy' in data:
            pred_age = data['pred_age'][pred_alive]
            pred_energy = data['pred_energy'][pred_alive]
            print(f"  Predator Age: {pred_age.mean():.1f} ± {pred_age.std():.1f}")
            print(f"  Predator Energy: {pred_energy.mean():.1f} ± {pred_energy.std():.1f}")


def compare_brains(file1, file2):
    """Compare two brain checkpoint files."""
    data1 = load_brain_file(file1)
    data2 = load_brain_file(file2)

    if data1 is None or data2 is None:
        return

    print(f"\n{'='*60}")
    print("BRAIN CHECKPOINT COMPARISON")
    print(f"{'='*60}")

    print(f"\nFile 1: {file1}")
    print(f"  Timestep: {data1.get('timestep', '?')}")
    print(f"  Prey alive: {data1['prey_alive'].sum()}")
    print(f"  Predators alive: {data1['pred_alive'].sum()}")

    print(f"\nFile 2: {file2}")
    print(f"  Timestep: {data2.get('timestep', '?')}")
    print(f"  Prey alive: {data2['prey_alive'].sum()}")
    print(f"  Predators alive: {data2['pred_alive'].sum()}")

    # Weight divergence
    print(f"\nWeight Divergence (Mean Absolute Difference):")

    prey_diff = np.abs(data1['prey_weights'] - data2['prey_weights']).mean()
    pred_diff = np.abs(data1['pred_weights'] - data2['pred_weights']).mean()

    print(f"  Prey brains: {prey_diff:.4f}")
    print(f"  Predator brains: {pred_diff:.4f}")

    # Trait divergence
    if 'prey_swim_speed' in data1 and 'prey_swim_speed' in data2:
        swim_diff_prey = np.abs(data1['prey_swim_speed'] - data2['prey_swim_speed']).mean()
        swim_diff_pred = np.abs(data1['pred_swim_speed'] - data2['pred_swim_speed']).mean()

        print(f"\nTrait Divergence (Mean Absolute Difference):")
        print(f"  Prey swim speed: {swim_diff_prey:.4f}")
        print(f"  Predator swim speed: {swim_diff_pred:.4f}")

    # Timestep difference
    if 'timestep' in data1 and 'timestep' in data2:
        timestep_diff = abs(data2['timestep'] - data1['timestep'])
        print(f"\nEvolutionary Distance:")
        print(f"  Timesteps apart: {timestep_diff:,}")
        if timestep_diff > 0:
            print(f"  Weight change per timestep (prey): {prey_diff / timestep_diff:.6f}")
            print(f"  Weight change per timestep (pred): {pred_diff / timestep_diff:.6f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze saved brain weights from HUNT simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View summary of brain checkpoint
  python analyze_brains.py brains_island_refuge.npz

  # Detailed summary
  python analyze_brains.py brains_step_100000.npz --verbose

  # Compare two checkpoints
  python analyze_brains.py brains_step_100000.npz --compare brains_step_500000.npz
        """
    )

    parser.add_argument('brain_file', type=str,
                        help='Path to brain .npz file')
    parser.add_argument('--summary', action='store_true',
                        help='Print detailed weight statistics (same as --verbose)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed statistics')
    parser.add_argument('--compare', type=str,
                        help='Compare with another brain file')

    args = parser.parse_args()

    # Load and analyze
    data = load_brain_file(args.brain_file)
    if data is None:
        return 1

    # Print summary
    verbose = args.summary or args.verbose
    print_summary(data, verbose=verbose)

    # Compare if requested
    if args.compare:
        print()  # Blank line
        compare_brains(args.brain_file, args.compare)

    print(f"\n{'='*60}\n")
    return 0


if __name__ == '__main__':
    exit(main())
