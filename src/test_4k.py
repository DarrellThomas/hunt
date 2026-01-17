"""Test 4K resolution performance."""

import torch
import time
from simulation_gpu import GPUEcosystem

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA: {torch.cuda.is_available()}\n")

print("="*70)
print("TESTING 4K (3840x2160) WITH 12,000 AGENTS")
print("="*70)

ecosystem = GPUEcosystem(
    width=3840,
    height=2160,
    num_prey=9600,
    num_predators=2400,
    device='cuda'
)

print("\nWarming up GPU...")
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
print("4K RESULTS")
print("="*70)
print(f"Resolution: 3840x2160 (4K)")
print(f"Total agents: 12,000 (9,600 prey + 2,400 predators)")
print(f"Time: {elapsed:.2f}s for 100 steps")
print(f"Per step: {elapsed/100*1000:.1f}ms")
print(f"FPS: {1/(elapsed/100):.1f}")
print(f"Target: 15+ FPS = 67ms per step")
print(f"Status: {'✓ READY!' if elapsed/100 < 0.067 else '⚠ MIGHT BE SLOW' if elapsed/100 < 0.1 else '✗ TOO SLOW'}")
print("="*70)
