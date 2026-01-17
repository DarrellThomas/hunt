"""Analyze the saved statistics from the simulation run."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Load statistics
stats = np.load('stats.npz')

print("=== SIMULATION ANALYSIS ===\n")
print(f"Total timesteps: {len(stats['prey_count'])}")
print(f"\nPopulation ranges:")
print(f"  Prey: {min(stats['prey_count'])} - {max(stats['prey_count'])}")
print(f"  Predators: {min(stats['predator_count'])} - {max(stats['predator_count'])}")

print(f"\nFinal state:")
print(f"  Prey: {stats['prey_count'][-1]}")
print(f"  Predators: {stats['predator_count'][-1]}")

print(f"\nAverage ages (final):")
print(f"  Prey: {stats['prey_avg_age'][-1]:.1f} timesteps")
print(f"  Predators: {stats['predator_avg_age'][-1]:.1f} timesteps")

print(f"\nAverage fitness (final):")
print(f"  Prey: {stats['prey_avg_fitness'][-1]:.1f}")
print(f"  Predators: {stats['predator_avg_fitness'][-1]:.1f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('HUNT - Co-Evolution Statistics', fontsize=14, fontweight='bold')

# Population counts
ax = axes[0, 0]
ax.plot(stats['prey_count'], color='green', label='Prey', linewidth=2)
ax.plot(stats['predator_count'], color='red', label='Predators', linewidth=2)
ax.set_xlabel('Timestep')
ax.set_ylabel('Population')
ax.set_title('Population Dynamics')
ax.legend()
ax.grid(True, alpha=0.3)

# Average age
ax = axes[0, 1]
ax.plot(stats['prey_avg_age'], color='green', label='Prey', linewidth=2)
ax.plot(stats['predator_avg_age'], color='red', label='Predators', linewidth=2)
ax.set_xlabel('Timestep')
ax.set_ylabel('Average Age')
ax.set_title('Average Age Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Average fitness
ax = axes[1, 0]
ax.plot(stats['prey_avg_fitness'], color='green', label='Prey', linewidth=2)
ax.plot(stats['predator_avg_fitness'], color='red', label='Predators', linewidth=2)
ax.set_xlabel('Timestep')
ax.set_ylabel('Average Fitness')
ax.set_title('Average Fitness Over Time')
ax.legend()
ax.grid(True, alpha=0.3)

# Predator/Prey ratio
ax = axes[1, 1]
ratio = np.array(stats['predator_count']) / (np.array(stats['prey_count']) + 1)  # +1 to avoid div by 0
ax.plot(ratio, color='purple', linewidth=2)
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal populations')
ax.set_xlabel('Timestep')
ax.set_ylabel('Predator/Prey Ratio')
ax.set_title('Predator/Prey Ratio')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ecosystem_stats.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved to ecosystem_stats.png")
