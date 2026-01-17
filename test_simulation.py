"""Quick test of the simulation logic without visualization."""

from world import World
import numpy as np

# Create world
print("Creating world...")
world = World(width=800, height=600, initial_prey=50, initial_predators=10)

print(f"Initial state: {len(world.prey)} prey, {len(world.predators)} predators")

# Track kills
total_kills = 0

# Run more timesteps
print("\nRunning simulation for 500 timesteps...")
for i in range(500):
    prev_prey = len(world.prey)
    world.step(mutation_rate=0.1)
    kills_this_step = prev_prey - len(world.prey)
    if kills_this_step > 0:
        total_kills += kills_this_step

    if i % 100 == 0:
        world.print_stats()

        # Check predator-prey distances
        if len(world.prey) > 0 and len(world.predators) > 0:
            min_distance = min(
                world.predators[0].distance_to(prey)
                for prey in world.prey
            )
            print(f"Closest predator-prey distance: {min_distance:.2f}")
            print(f"Total kills so far: {total_kills}")

print(f"\n✓ Test complete! Total kills: {total_kills}")
print(f"Final state: {len(world.prey)} prey, {len(world.predators)} predators")

# Check if any births happened
if len(world.prey) + total_kills > 50 or len(world.predators) > 10:
    print("✓ Reproduction occurred!")
else:
    print("✗ No reproduction detected")
