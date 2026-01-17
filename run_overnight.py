"""
Overnight training run with automatic logging and periodic saves.
No pygame window - pure simulation, writes stats to disk.
"""

import torch
import time
import numpy as np
from simulation_gpu import GPUEcosystem
from datetime import datetime

print("="*70)
print("HUNT - OVERNIGHT TRAINING RUN")
print("="*70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Create 4K ecosystem
ecosystem = GPUEcosystem(
    width=3840,
    height=2160,
    num_prey=9600,
    num_predators=2400,
    device='cuda'
)

# Training parameters
TARGET_TIMESTEPS = 50000  # Run for 50K timesteps (adjust as needed)
SAVE_INTERVAL = 1000      # Save stats every 1000 steps
LOG_INTERVAL = 100        # Print progress every 100 steps

print(f"Target timesteps: {TARGET_TIMESTEPS:,}")
print(f"Save interval: {SAVE_INTERVAL}")
print(f"Log interval: {LOG_INTERVAL}")
print()

# Statistics tracking
stats = {
    'timesteps': [],
    'prey_count': [],
    'pred_count': [],
    'prey_avg_age': [],
    'pred_avg_age': [],
    'pred_avg_energy': [],
}

start_time = time.time()
last_save = time.time()

try:
    for step in range(TARGET_TIMESTEPS):
        ecosystem.step(mutation_rate=0.1)

        # Log progress
        if (step + 1) % LOG_INTERVAL == 0:
            state = ecosystem.get_state_cpu()
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            eta_seconds = (TARGET_TIMESTEPS - step - 1) / steps_per_sec if steps_per_sec > 0 else 0
            eta_hours = eta_seconds / 3600

            print(f"[{step+1:6d}/{TARGET_TIMESTEPS}] "
                  f"Prey: {state['prey_count']:5d} | "
                  f"Pred: {state['pred_count']:5d} | "
                  f"Age: {state['prey_avg_age']:6.1f}/{state['pred_avg_age']:6.1f} | "
                  f"Energy: {state['pred_avg_energy']:5.1f} | "
                  f"Speed: {steps_per_sec:4.1f} steps/s | "
                  f"ETA: {eta_hours:.1f}h")

            # Record stats
            stats['timesteps'].append(step + 1)
            stats['prey_count'].append(state['prey_count'])
            stats['pred_count'].append(state['pred_count'])
            stats['prey_avg_age'].append(state['prey_avg_age'])
            stats['pred_avg_age'].append(state['pred_avg_age'])
            stats['pred_avg_energy'].append(state['pred_avg_energy'])

        # Periodic save
        if (step + 1) % SAVE_INTERVAL == 0:
            np.savez('overnight_stats.npz', **stats)
            save_time = time.time()
            print(f"  → Stats saved to overnight_stats.npz (last save: {save_time - last_save:.1f}s ago)")
            last_save = save_time

except KeyboardInterrupt:
    print("\n\n! Interrupted by user")

finally:
    # Final save
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)

    end_time = time.time()
    elapsed = end_time - start_time

    final_state = ecosystem.get_state_cpu()

    print(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    print(f"Total timesteps: {ecosystem.timestep:,}")
    print(f"Final prey: {final_state['prey_count']}")
    print(f"Final predators: {final_state['pred_count']}")
    print(f"Prey avg age: {final_state['prey_avg_age']:.1f}")
    print(f"Predator avg age: {final_state['pred_avg_age']:.1f}")
    print(f"Predator avg energy: {final_state['pred_avg_energy']:.1f}")

    # Save final statistics
    np.savez('overnight_stats.npz', **stats)
    print(f"\n✓ Statistics saved to: overnight_stats.npz")
    print(f"✓ Log this terminal output for full results")
    print("\n" + "="*70)
