"""
View results from overnight training run.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

print("="*70)
print("OVERNIGHT TRAINING RESULTS")
print("="*70)

# Load stats
if not Path('overnight_stats.npz').exists():
    print("\n✗ No overnight_stats.npz found!")
    print("  Run the overnight training first: python3 run_overnight.py")
    exit(1)

stats = np.load('overnight_stats.npz')

print(f"\n✓ Loaded stats from overnight_stats.npz")
print(f"  Total data points: {len(stats['timesteps'])}")
print(f"  Timesteps recorded: {stats['timesteps'][0]} to {stats['timesteps'][-1]}")
print()

# Summary statistics
print("SUMMARY")
print("-" * 70)
print(f"Prey population:")
print(f"  Min: {min(stats['prey_count'])}")
print(f"  Max: {max(stats['prey_count'])}")
print(f"  Final: {stats['prey_count'][-1]}")
print()
print(f"Predator population:")
print(f"  Min: {min(stats['pred_count'])}")
print(f"  Max: {max(stats['pred_count'])}")
print(f"  Final: {stats['pred_count'][-1]}")
print()
print(f"Prey average age:")
print(f"  Min: {min(stats['prey_avg_age']):.1f}")
print(f"  Max: {max(stats['prey_avg_age']):.1f}")
print(f"  Final: {stats['prey_avg_age'][-1]:.1f}")
print()
print(f"Predator average age:")
print(f"  Min: {min(stats['pred_avg_age']):.1f}")
print(f"  Max: {max(stats['pred_avg_age']):.1f}")
print(f"  Final: {stats['pred_avg_age'][-1]:.1f}")
print()
print(f"Predator average energy:")
print(f"  Min: {min(stats['pred_avg_energy']):.1f}")
print(f"  Max: {max(stats['pred_avg_energy']):.1f}")
print(f"  Final: {stats['pred_avg_energy'][-1]:.1f}")

# Create detailed plots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Overnight Training Results - 4K Ecosystem Evolution', fontsize=16, fontweight='bold')

# Population dynamics
ax = axes[0, 0]
ax.plot(stats['timesteps'], stats['prey_count'], 'g-', linewidth=2, label='Prey', alpha=0.8)
ax.plot(stats['timesteps'], stats['pred_count'], 'r-', linewidth=2, label='Predators', alpha=0.8)
ax.set_xlabel('Timestep')
ax.set_ylabel('Population')
ax.set_title('Population Dynamics')
ax.legend()
ax.grid(True, alpha=0.3)

# Average ages
ax = axes[0, 1]
ax.plot(stats['timesteps'], stats['prey_avg_age'], 'g-', linewidth=2, label='Prey', alpha=0.8)
ax.plot(stats['timesteps'], stats['pred_avg_age'], 'r-', linewidth=2, label='Predators', alpha=0.8)
ax.set_xlabel('Timestep')
ax.set_ylabel('Average Age (timesteps)')
ax.set_title('Evolution of Average Age')
ax.legend()
ax.grid(True, alpha=0.3)

# Predator energy
ax = axes[1, 0]
ax.plot(stats['timesteps'], stats['pred_avg_energy'], 'r-', linewidth=2, alpha=0.8)
ax.axhline(y=120, color='orange', linestyle='--', alpha=0.5, label='Reproduction threshold')
ax.axhline(y=60, color='yellow', linestyle='--', alpha=0.5, label='Half energy')
ax.set_xlabel('Timestep')
ax.set_ylabel('Average Energy')
ax.set_title('Predator Energy Management')
ax.legend()
ax.grid(True, alpha=0.3)

# Predator/Prey ratio
ax = axes[1, 1]
ratio = np.array(stats['pred_count']) / (np.array(stats['prey_count']) + 1)
ax.plot(stats['timesteps'], ratio, 'purple', linewidth=2, alpha=0.8)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='1:1 ratio')
ax.set_xlabel('Timestep')
ax.set_ylabel('Predator/Prey Ratio')
ax.set_title('Population Ratio')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('overnight_results.png', dpi=150, bbox_inches='tight')
print("\n" + "="*70)
print("✓ Visualization saved to: overnight_results.png")
print("="*70)
