# Running Experiments with HUNT

This guide covers how to run simulations, collect data, and analyze results.

## Quick Start

### Run a Single Simulation

```bash
# CPU mode (visualized)
cd src
python3 main.py

# GPU mode (fast, headless)
python3 main_gpu.py
```

### Command-Line from Project Root

```bash
# CPU mode
python3 run.py

# GPU mode
python3 run_gpu.py
```

## Execution Modes

### 1. Interactive Mode (Default)

Visual simulation with Pygame rendering:

```bash
cd src
python3 main.py
```

**Controls:**
- Close window to stop
- Watch populations evolve in real-time
- Good for debugging and demonstrations

**Performance:** ~30-100 steps/second

### 2. Overnight Training Mode

Automated data collection without visualization:

```bash
cd src
./start_overnight.sh
```

This runs simulation in the background and:
- Collects data every 100 steps
- Saves to `stats_autosave.npz`
- Logs progress to `overnight.log`
- Can run for days/weeks

**Performance:** ~1000-5000 steps/second (CPU)

### 3. GPU Accelerated Mode

Maximum performance using CUDA:

```bash
cd src
python3 main_gpu.py

# Options:
python3 main_gpu.py --headless          # No visualization
python3 main_gpu.py --steps 100000      # Run for N steps
python3 main_gpu.py --collect-data       # Save stats
```

**Performance:** ~50,000-200,000 steps/second (depends on GPU)

## Data Collection

### Automatic Collection

The simulation automatically tracks:
- Population counts (prey, predators)
- Average/min/max energy levels
- Birth and death counts
- Swim speed evolution statistics
- River utilization (% in river, on island)

### Accessing Data During Simulation

```python
from world import World

world = World(width=800, height=600, initial_prey=100, initial_predators=20)

for step in range(1000):
    world.step()

    if step % 100 == 0:
        state = world.get_state()
        print(f"Step {state['timestep']}: "
              f"{state['prey_count']} prey, "
              f"{state['predator_count']} predators")
```

### Saving Data

```python
import numpy as np

# Collect data
timesteps = []
prey_counts = []
pred_counts = []

for step in range(10000):
    world.step()
    state = world.get_state()

    timesteps.append(state['timestep'])
    prey_counts.append(state['prey_count'])
    pred_counts.append(state['predator_count'])

# Save to file
np.savez('my_experiment.npz',
         timesteps=timesteps,
         prey=prey_counts,
         predators=pred_counts)
```

### Loading Data

```python
# Load saved data
data = np.load('my_experiment.npz')

timesteps = data['timesteps']
prey = data['prey']
predators = data['predators']

# Analyze
import matplotlib.pyplot as plt
plt.plot(timesteps, prey, label='Prey')
plt.plot(timesteps, predators, label='Predators')
plt.legend()
plt.show()
```

## Batch Experiments

### Running Multiple Configurations

```python
from config_new import SimulationConfig
import numpy as np

# Define parameter sweep
world_sizes = [(400, 300), (800, 600), (1600, 1200)]
results = []

for width, height in world_sizes:
    print(f"Running {width}x{height}...")

    # Create config
    config = SimulationConfig.default_two_species()
    config.world.width = width
    config.world.height = height

    # Run simulation
    world = World(width=width, height=height)

    # ... run and collect data ...

    results.append({
        'size': (width, height),
        'final_prey': len(world.prey),
        'final_pred': len(world.predators)
    })

# Save results
np.save('size_sweep_results.npy', results)
```

### Parallel Execution

```python
from multiprocessing import Pool
import numpy as np

def run_trial(trial_id):
    """Run a single trial."""
    world = World(width=800, height=600)

    for step in range(50000):
        world.step()

    return {
        'trial': trial_id,
        'final_prey': len(world.prey),
        'final_pred': len(world.predators)
    }

# Run 10 trials in parallel
with Pool(10) as pool:
    results = pool.map(run_trial, range(10))

# Analyze
prey_counts = [r['final_prey'] for r in results]
print(f"Mean final prey: {np.mean(prey_counts):.1f} ± {np.std(prey_counts):.1f}")
```

### GPU Batch Processing

For maximum throughput, run multiple trials on GPU:

```python
import torch
from simulation_gpu import GPUEcosystem

results = []

for trial in range(10):
    print(f"Trial {trial+1}/10")

    sim = GPUEcosystem(
        width=3200,
        height=2400,
        num_prey=8000,
        num_predators=2000,
        device='cuda'
    )

    for step in range(100000):
        sim.step()

        if step % 10000 == 0:
            state = sim.get_state_cpu()
            print(f"  Step {step}: {state['prey_count']} prey")

    final_state = sim.get_state_cpu()
    results.append(final_state)

# Save
np.save('gpu_batch_results.npy', results)
```

## Configuration-Based Experiments

### Using JSON Configs

```python
from config_new import SimulationConfig

# Load config
config = SimulationConfig.from_json_file('configs/three_species.json')

# Modify parameters
config.species[0].initial_count = 500  # More prey
config.world.width = 1600

# Save modified config
config.to_json_file('configs/my_experiment.json')

# Run simulation with config
# (Requires implementing config-based World initialization)
```

### Parameter Sweeps

```python
import itertools

# Define parameter ranges
prey_counts = [100, 200, 400]
pred_counts = [10, 20, 40]
world_sizes = [(800, 600), (1600, 1200)]

# Generate all combinations
experiments = list(itertools.product(prey_counts, pred_counts, world_sizes))

print(f"Total experiments: {len(experiments)}")

# Run each
results = []
for prey_n, pred_n, (w, h) in experiments:
    print(f"Running: prey={prey_n}, pred={pred_n}, size={w}x{h}")

    # ... run simulation ...
    # ... collect results ...

    results.append({
        'params': (prey_n, pred_n, w, h),
        'outcome': '...'
    })
```

## Analysis Scripts

### Population Dynamics

```python
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.load('experiment.npz')

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(data['timesteps'], data['prey'], label='Prey', color='blue')
plt.plot(data['timesteps'], data['predators'], label='Predators', color='red')
plt.xlabel('Time Steps')
plt.ylabel('Population')
plt.legend()
plt.title('Population Over Time')

plt.subplot(1, 2, 2)
plt.plot(data['prey'], data['predators'])
plt.xlabel('Prey Population')
plt.ylabel('Predator Population')
plt.title('Phase Space')

plt.tight_layout()
plt.savefig('population_dynamics.png')
plt.show()
```

### Evolution Tracking

```python
# Track evolved traits over time
import numpy as np
import matplotlib.pyplot as plt

data = np.load('evolution_data.npz')

plt.figure(figsize=(10, 6))
plt.plot(data['timesteps'], data['prey_avg_swim'], label='Prey Swim Speed')
plt.plot(data['timesteps'], data['pred_avg_swim'], label='Predator Swim Speed')
plt.xlabel('Time Steps')
plt.ylabel('Average Swim Speed')
plt.legend()
plt.title('Evolution of Swim Speed')
plt.savefig('swim_speed_evolution.png')
```

### Statistical Analysis

```python
import numpy as np
from scipy import stats

# Load multiple trials
trial_files = ['trial_1.npz', 'trial_2.npz', 'trial_3.npz']
final_prey = []

for file in trial_files:
    data = np.load(file)
    final_prey.append(data['prey'][-1])

# Statistics
mean = np.mean(final_prey)
std = np.std(final_prey)
ci = stats.t.interval(0.95, len(final_prey)-1,
                       loc=mean,
                       scale=stats.sem(final_prey))

print(f"Final prey count: {mean:.1f} ± {std:.1f}")
print(f"95% CI: [{ci[0]:.1f}, {ci[1]:.1f}]")
```

## Performance Optimization

### CPU Optimization

```python
# Use smaller world for faster simulation
world = World(width=400, height=300)  # Instead of 800x600

# Reduce population
world = World(initial_prey=50, initial_predators=10)  # Instead of 200/40

# Disable visualization
# (Use overnight mode or GPU headless mode)
```

### GPU Optimization

```python
from simulation_gpu import GPUEcosystem

# Use larger batches (more agents = better GPU utilization)
sim = GPUEcosystem(
    width=3200,
    height=2400,
    num_prey=16000,  # Large batch
    num_predators=4000,
    device='cuda'
)

# Reduce state transfers
for step in range(100000):
    sim.step()

    # Only get state occasionally
    if step % 10000 == 0:
        state = sim.get_state_cpu()  # Expensive CPU-GPU transfer
        print(f"Step {step}")
```

## Reproducibility

### Random Seeds

```python
import numpy as np
import random

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Now results will be identical across runs
world = World()
```

### Saving Configurations

```python
# Save exact config used
config = SimulationConfig.default_two_species()
config.to_json_file('experiment_config.json')

# Save metadata
metadata = {
    'date': '2026-01-17',
    'version': '1.0',
    'seed': 42,
    'notes': 'Testing population dynamics'
}
np.save('experiment_metadata.npy', metadata)
```

## Debugging

### Verify Populations

```python
world = World(width=800, height=600, initial_prey=100, initial_predators=20)

for step in range(1000):
    world.step()

    # Check for extinction
    if len(world.prey) == 0:
        print(f"Prey extinct at step {step}!")
        break
    if len(world.predators) == 0:
        print(f"Predators extinct at step {step}!")
        break

    # Check for runaway growth
    if len(world.prey) > 10000:
        print(f"Prey population explosion at step {step}!")
        break
```

### Visualize Agent Positions

```python
import matplotlib.pyplot as plt

state = world.get_state()

prey_pos = state['prey_positions']
pred_pos = state['predator_positions']

plt.figure(figsize=(10, 8))
plt.scatter(prey_pos[:, 0], prey_pos[:, 1], c='blue', label='Prey', alpha=0.6)
plt.scatter(pred_pos[:, 0], pred_pos[:, 1], c='red', label='Predators', alpha=0.6)
plt.xlim(0, world.width)
plt.ylim(0, world.height)
plt.legend()
plt.title(f'Positions at Step {state["timestep"]}')
plt.show()
```

## Example Experiments

### 1. Predator-Prey Cycles

Study classic Lotka-Volterra dynamics:

```python
world = World(width=800, height=600, initial_prey=200, initial_predators=40)

# Run for long time
for step in range(50000):
    world.step()

# Expect oscillating populations
```

### 2. Evolution of Speed

Track how speed evolves in river:

```python
# River enabled by default
world = World()

# Track average speeds
speeds = []
for step in range(20000):
    world.step()

    if step % 100 == 0:
        avg_speed = np.mean([p.swim_speed for p in world.prey])
        speeds.append(avg_speed)

# Plot evolution
plt.plot(speeds)
plt.title('Evolution of Swim Speed')
plt.show()
```

### 3. Boundary Mode Comparison

Compare toroidal vs bounded:

```python
# Run both modes
results_toroidal = run_experiment(BoundaryMode.TOROIDAL)
results_bounded = run_experiment(BoundaryMode.BOUNDED)

# Compare stability
compare_population_variance(results_toroidal, results_bounded)
```

## Next Steps

- See `ADDING_SPECIES.md` for multi-species experiments
- See `BOUNDARY_MODES.md` for toroidal vs bounded setups
- See `analyze_evolution.py` for example analysis script
- See `tests/test_integration.py` for programmatic examples

## Troubleshooting

### Simulation Crashes

- Check memory usage (large populations = high RAM)
- Verify CUDA available for GPU mode
- Check for NaN values in positions

### Poor Performance

- Use GPU mode for large simulations
- Reduce population sizes
- Use overnight mode (no visualization)

### Unexpected Results

- Check random seed (set for reproducibility)
- Verify extinction prevention is enabled
- Check interaction configurations
- Review parameter values (too extreme?)

## Resources

- Example configs: `configs/`
- Analysis scripts: `analyze_evolution.py`
- Test examples: `tests/test_integration.py`
- Overnight mode: `src/start_overnight.sh`
