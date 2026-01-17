"""Quick test of the larger world without visualization."""

from world import World
import time

print("Creating 4x larger world (1600x1200)...")
world = World(width=1600, height=1200, initial_prey=200, initial_predators=40)

print(f"Initial state: {len(world.prey)} prey, {len(world.predators)} predators")
print("\nRunning 100 timesteps to test performance...")

start_time = time.time()
for i in range(100):
    world.step(mutation_rate=0.1)
    if i % 20 == 0:
        print(f"Timestep {i}: {len(world.prey)} prey, {len(world.predators)} predators")

elapsed = time.time() - start_time
print(f"\n✓ Test complete!")
print(f"Time: {elapsed:.2f}s for 100 timesteps ({elapsed/100*1000:.1f}ms per step)")
print(f"Final: {len(world.prey)} prey, {len(world.predators)} predators")
