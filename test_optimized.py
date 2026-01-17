"""Test performance comparison between original and optimized versions."""

import time
import sys

print("Testing OPTIMIZED version...")
from world_optimized import World as WorldOptimized

world = WorldOptimized(width=1600, height=1200, initial_prey=200, initial_predators=40)
print(f"Initial: {len(world.prey)} prey, {len(world.predators)} predators")

start = time.time()
for i in range(100):
    world.step(mutation_rate=0.1)
    if i % 20 == 0:
        print(f"  Step {i}: {len(world.prey)} prey, {len(world.predators)} predators")

elapsed = time.time() - start
print(f"\n✓ Optimized: {elapsed:.2f}s for 100 steps ({elapsed/100*1000:.1f}ms per step)")
print(f"  Target: 30 FPS = 33ms per step")
print(f"  Performance: {'✓ GOOD' if elapsed/100 < 0.033 else '✗ NEEDS MORE OPTIMIZATION'}")
