"""Test massive GPU simulation performance."""

import torch
import time
from simulation_gpu import GPUEcosystem

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.cuda.is_available()}\n")

print("="*70)
print("CREATING MASSIVE ECOSYSTEM")
print("="*70)

ecosystem = GPUEcosystem(
    width=3200,
    height=2400,
    num_prey=8000,
    num_predators=2000,
    device='cuda'
)

print("\nWarming up...")
for _ in range(10):
    ecosystem.step()

print("\nBenchmarking 100 steps...")
start = time.time()
for i in range(100):
    ecosystem.step()
    if i % 20 == 0:
        state = ecosystem.get_state_cpu()
        print(f"  Step {i}: {state['prey_count']} prey, {state['pred_count']} predators")

elapsed = time.time() - start

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Time: {elapsed:.2f}s for 100 steps")
print(f"Per step: {elapsed/100*1000:.1f}ms")
print(f"Target: 60 FPS = 16.7ms per step")
print(f"Achieved: {1/(elapsed/100):.1f} FPS")
print(f"Status: {'✓ READY TO RUN' if elapsed/100 < 0.017 else '✗ NEEDS OPTIMIZATION' if elapsed/100 < 0.033 else '⚠ SLOW BUT WATCHABLE'}")
print("="*70)
