# HUNT - Predator-Prey Co-Evolution

A neuroevolution simulation where predators and prey both learn to survive through natural selection.

## Overview

This project implements a 2D ecosystem where:
- **Prey** (green) learn to evade predators and survive long enough to reproduce
- **Predators** (red) learn to hunt prey or starve to death
- Both populations evolve through **neuroevolution** - successful agents reproduce and pass their neural network weights to offspring with mutations
- No explicit rewards or training - just survival of the fittest

## Quick Start

```bash
python3 main.py
```

The Pygame window will open showing the live ecosystem. Watch as both populations co-evolve!

### Controls
- **SPACE** - Pause/Resume simulation
- **S** - Save statistics to `stats.npz`
- **ESC** - Quit

## How It Works

### Agents
Each agent (predator or prey) has:
- A **neural network brain** (3 layers, 32 hidden units)
- Sensory observations of nearby agents
- Movement controlled by neural network output

### Evolution
- Agents that survive longer accumulate higher fitness
- Successful agents reproduce, creating offspring with mutated brains
- Failed strategies die out
- Over generations, effective hunting and evasion behaviors emerge

### Survival Mechanics

**Prey:**
- Feed on grass (infinite food)
- Die from: being caught, old age (500 timesteps)
- Reproduce every 200 timesteps if alive

**Predators:**
- Must catch prey to restore energy
- Die from: starvation (energy reaches 0), old age (800 timesteps)
- Reproduce when energy > 120 (cooldown: 150 timesteps)

## Architecture

- `brain.py` - Neural network implementation
- `agent.py` - Prey and Predator classes
- `world.py` - Ecosystem simulation engine
- `main.py` - Pygame visualization and training loop
- `THESIS.md` - Design document and approach

## Expected Behaviors

As the simulation runs, you should observe:
- **Prey**: Flocking, evasive maneuvers, maintaining distance from predators
- **Predators**: Chase behavior, improved pursuit strategies
- **Population dynamics**: Boom-bust cycles or equilibrium

## Parameters

Key parameters (in `agent.py`):
- Mutation rate: 0.1 (10% weight change)
- Catch radius: 8.0 units
- Predator speed: 2.5 (slightly slower than prey at 3.0)
- Energy mechanics: 150 max, 0.3 cost per step, 60 gain per kill

Adjust these to experiment with different ecosystem dynamics!

## Statistics

Press 'S' to save statistics during the run. Stats include:
- Population counts over time
- Average age and fitness for each population

Load with NumPy:
```python
import numpy as np
stats = np.load('stats.npz')
print(stats['prey_count'])
```

## Testing

Run headless test without visualization:
```bash
python3 test_simulation.py
```

## Credits

Built with NumPy, Pygame, and emergent complexity.
