# HUNT Architecture Documentation

## Overview

HUNT is a neuroevolution-based predator-prey ecosystem simulator with two parallel implementations: CPU (NumPy) and GPU (PyTorch). Agents possess neural network brains that evolve through genetic algorithms without backpropagation.

## File Responsibilities

### Core Simulation

**`brain.py`** (123 lines)
- Pure NumPy feedforward neural network (3 layers)
- Methods: `forward()`, `mutate()`, `copy()`, `get_weights()`, `set_weights()`
- Architecture: input → hidden(32) → hidden(32) → output(2)
- No PyTorch dependency - CPU only

**`agent.py`** (335 lines)
- Base `Agent` class with shared physics/reproduction logic
- `Prey` class: observes predators/prey, evades, reproduces by age
- `Predator` class: observes prey, hunts, energy-based reproduction
- `vectorized_distances()`: toroidal distance calculations
- Uses config.py constants for all parameters

**`world.py`** (246 lines)
- CPU-based ecosystem simulation
- Manages lists of agent instances
- Vectorized operations for observations/collisions
- Statistics tracking
- Extinction prevention mechanisms

**`simulation_gpu.py`** (561 lines)
- Fully GPU-resident ecosystem (PyTorch tensors)
- `NeuralNetBatch`: batched networks for all agents
- `GPUEcosystem`: parallelize everything on GPU
- Sampled observations (avoids O(n²) for 10K agents)
- Direct tensor operations for physics/reproduction

### Environmental Features

**`river.py`** (244 lines)
- `River` class: flowing water with optional island
- Methods: `is_in_river()`, `is_on_island()`, `get_flow_at()`
- Curved path generation with sine waves
- Island behavior modifiers (speed, hunger, reproduction)
- River splits into two channels around island

**`config.py`** (67 lines)
- Centralized configuration for all parameters
- Prey: speed, acceleration, lifespan, reproduction timing
- Predators: energy, hunger, reproduction thresholds
- River: dimensions, flow speed, island configuration
- Island behavior multipliers (recently added)

### Visualization

**`main.py`** (273 lines)
- Pygame visualization for CPU simulation
- `Visualizer` class: rendering, UI, event handling
- Real-time display at 30 FPS
- River/island rendering
- Stats overlay

**`main_gpu.py`** (274 lines)
- Pygame visualization for GPU simulation
- `GPUVisualizer`: similar to CPU version
- Auto-saves stats every 1000 steps
- Optimized for 10K+ agents
- Supports fullscreen 4K

### Analysis Tools

**`analyze_evolution.py`** (242 lines)
- Loads stats from `.npz` files
- Plots: population, swim speed evolution, habitat preference
- Speciation detection based on diversity metrics
- Comprehensive 6-plot visualization

**`analyze_stats.py`** (77 lines)
- Simpler analysis for basic stats
- 4-plot visualization
- Population dynamics and fitness tracking

## Class Hierarchy

```
Agent (base class)
├── pos, vel, acc: np.array
├── age, fitness: int/float
├── brain: Brain instance
├── world_width, world_height: boundaries
├── Methods:
│   ├── observe() [abstract]
│   ├── act(observation)
│   ├── update_physics()
│   └── reproduce(mutation_rate)
│
├── Prey
│   ├── swim_speed: float (evolvable)
│   ├── max_lifespan, reproduction_age: individual timing
│   ├── time_since_reproduction: int
│   ├── observe(pred_pos, pred_vel, prey_pos, prey_vel, my_index)
│   │   └── Returns 32-dim vector: 5 nearest predators + 3 nearest prey
│   ├── should_die() → age >= max_lifespan
│   └── can_reproduce() → time >= reproduction_age
│
└── Predator
    ├── swim_speed: float (evolvable)
    ├── energy: float (must hunt to survive)
    ├── max_lifespan, reproduction_cooldown: individual timing
    ├── time_since_reproduction: int
    ├── observe(prey_pos, prey_vel)
    │   └── Returns 21-dim vector: 5 nearest prey + hunger
    ├── update_energy() → decrease per step
    ├── eat() → restore energy
    ├── should_die() → energy <= 0 or age >= max_lifespan
    ├── can_reproduce() → energy >= threshold and time >= cooldown
    └── pay_reproduction_cost() → decrease energy

River
├── path_x, path_y: np.array (centerline)
├── flow_dir_x, flow_dir_y: np.array (normalized directions)
├── enabled, split: booleans
├── Methods:
│   ├── is_in_river(x, y) → bool
│   ├── is_on_island(x, y) → bool
│   ├── get_flow_at(x, y) → (flow_x, flow_y)
│   ├── island_behavior(agent_type, x, y) → dict | None
│   └── get_render_data() → dict

Brain
├── w1, b1, w2, b2, w3, b3: np.array (weights/biases)
├── input_size, hidden_size, output_size: int
├── Methods:
│   ├── forward(x) → output (tanh activations)
│   ├── mutate(mutation_rate)
│   ├── copy() → Brain
│   ├── get_weights() → flat array
│   └── set_weights(weights)
```

## GPU Class Hierarchy

```
GPUEcosystem
├── Tensors (all on GPU):
│   ├── prey_pos, prey_vel, prey_acc: (N, 2)
│   ├── prey_age, prey_repro_timer: (N,)
│   ├── prey_alive: (N,) bool
│   ├── prey_swim_speed: (N,) evolvable
│   ├── prey_max_age_individual, prey_repro_age_individual: (N,)
│   ├── pred_pos, pred_vel, pred_acc: (M, 2)
│   ├── pred_age, pred_energy, pred_repro_timer: (M,)
│   ├── pred_alive: (M,) bool
│   ├── pred_swim_speed: (M,) evolvable
│   └── pred_max_age_individual, pred_repro_cooldown_individual: (M,)
│
├── Neural Networks:
│   ├── prey_brain: NeuralNetBatch (batched forward pass)
│   └── pred_brain: NeuralNetBatch (batched forward pass)
│
├── River: River instance (CPU-side, checked per agent)
│
└── Methods:
    ├── compute_toroidal_distances(pos1, pos2) → distances, vectors
    ├── observe_prey() → observations (N, 32)
    ├── observe_predators() → observations (M, 21)
    ├── get_island_modifiers(positions, agent_type) → dict | None
    ├── step(mutation_rate)
    └── get_state_cpu() → dict (for visualization)

NeuralNetBatch (nn.Module)
├── fc1, fc2, fc3: nn.Linear layers
├── forward(x) → batched output for all agents
└── mutate_random(indices, mutation_rate)
```

## Data Flow: Perception → Decision → Action

### CPU Version (world.py)

```
1. OBSERVATION PHASE
   For each prey:
     world.step() →
       prey_positions = np.array([p.pos for p in self.prey])
       predator_positions = np.array([p.pos for p in self.predators])
       observation = prey.observe(predator_positions, ...)
         ↓
         vectorized_distances() computes all distances
         ↓
         Find 5 nearest predators + 3 nearest prey
         ↓
         Return 32-dim normalized vector

   For each predator:
     observation = predator.observe(prey_positions, ...)
       ↓
       vectorized_distances() computes all distances
       ↓
       Find 5 nearest prey
       ↓
       Return 21-dim vector (20 prey info + 1 hunger)

2. DECISION PHASE
   For each agent:
     action = agent.act(observation)
       ↓
       output = brain.forward(observation)  # 3-layer NN
       ↓
       acc = output * max_acceleration
       ↓
       Return acceleration vector

3. ACTION PHASE
   For each agent:
     agent.update_physics(dt=1.0)
       ↓
       vel += acc * dt
       ↓
       Limit: speed = ||vel||, if speed > max_speed: vel *= max_speed/speed
       ↓
       pos += vel * dt
       ↓
       Toroidal wrap: pos = pos % [world_width, world_height]
       ↓
       Apply river flow (if in river):
         flow_x, flow_y = river.get_flow_at(pos)
         flow_factor = max(0, 1 - swim_speed/5)
         vel += [flow_x, flow_y] * flow_factor
       ↓
       Apply island modifiers (if on island):
         Speed multiplier affects max velocity
         Hunger multiplier affects energy cost
         Reproduction multiplier affects timing
       ↓
       age += 1

4. COLLISION DETECTION
   For each predator:
     vectorized_distances(predator.pos, prey_positions)
     ↓
     Find prey within catch_radius
     ↓
     Catch nearest one
     ↓
     predator.eat() → energy += gain
     ↓
     Remove prey from simulation

5. LIFE CYCLE
   Deaths:
     prey = [p for p in prey if not p.should_die()]
     predators = [p for p in predators if not p.should_die()]

   Births:
     For each agent that can_reproduce():
       child = agent.reproduce(mutation_rate)
         ↓
         Create new agent at offset position
         ↓
         child.brain = parent.brain.copy()
         ↓
         child.brain.mutate(mutation_rate)
         ↓
         Inherit swim_speed with mutation
       ↓
       Add child to population
       ↓
       Parent pays reproduction cost (energy/timer reset)

   Extinction prevention:
     if len(prey) < min_threshold:
       spawn random prey
     if len(predators) < min_threshold:
       spawn random predators
```

### GPU Version (simulation_gpu.py)

```
1. OBSERVATION PHASE (Batched)
   observe_prey():
     alive_prey_pos = prey_pos[prey_alive]  # (N, 2) tensor
     alive_pred_pos = pred_pos[pred_alive]  # (M, 2) tensor
     ↓
     Sample predators (max 100) to reduce O(n²)
     ↓
     compute_toroidal_distances(prey_pos, sampled_pred_pos)
       → distances (N, M), vectors (N, M, 2)
     ↓
     torch.topk() finds 5 nearest for each prey
     ↓
     Normalize and pack into observations tensor (N, 32)

   observe_predators():
     Sample prey (max 200)
     ↓
     Similar batched distance computation
     ↓
     Pack into observations tensor (M, 21)

2. DECISION PHASE (Batched)
   prey_actions = prey_brain(prey_obs)
     ↓
     Batched forward pass through PyTorch network
     ↓
     All prey get actions simultaneously

   pred_actions = pred_brain(pred_obs)

3. ACTION PHASE (Vectorized)
   prey_vel[prey_alive] += prey_acc[prey_alive]
   ↓
   Vectorized speed limiting:
     speed = torch.norm(vel, dim=1)
     vel = torch.where(speed > max_speed, vel/speed * max_speed, vel)
   ↓
   Apply island speed modifiers:
     island_mods = get_island_modifiers(prey_pos[prey_alive], 'prey')
     if island_mods:
       speed_limit = max_speed * island_mods['speed']
       # Apply per-agent limits
   ↓
   pos = (pos + vel) % [width, height]
   ↓
   Apply river flow (batch):
     pos_cpu = prey_pos[prey_alive].cpu().numpy()
     flows = river.get_flow_at_batch(pos_cpu)
     flow_tensor = torch.tensor(flows, device='cuda')
     flow_factors = 1.0 - swim_speed/5.0
     vel += flow_tensor * flow_factors

4. COLLISION DETECTION (Vectorized)
   distances, _ = compute_toroidal_distances(pred_pos, prey_pos)
     → (M, N) distance matrix
   ↓
   catches = distances < catch_radius  # (M, N) bool matrix
   ↓
   For each predator:
     Find which prey are caught
     Select nearest one
     Mark prey as caught
   ↓
   Batch update energy:
     pred_energy[predator_indices] += energy_gain
   ↓
   Batch remove prey:
     prey_alive[caught_indices] = False

5. LIFE CYCLE (Vectorized)
   Deaths:
     prey_alive &= prey_age < prey_max_age_individual
     pred_alive &= (pred_age < pred_max_age_individual) & (pred_energy > 0)

   Births:
     Get island reproduction modifiers
     ↓
     can_repro_prey = prey_alive & (prey_repro_timer >= repro_age * island_mult)
     can_repro_pred = pred_alive & (pred_energy >= threshold) &
                      (pred_repro_timer >= cooldown * island_mult)
     ↓
     dead_prey_idx = torch.where(~prey_alive)[0]
     alive_prey_idx = torch.where(can_repro_prey)[0]
     ↓
     Randomly select parents
     ↓
     Batch spawn at offset positions:
       spawn_distance = torch.rand(...) * 130 + 20
       spawn_angle = torch.rand(...) * 2π
       new_pos = parent_pos + [spawn_distance * cos(angle), ...]
     ↓
     Reset all offspring attributes
     ↓
     Inherit and mutate swim_speed:
       offspring_swim = parent_swim + torch.randn(...) * mutation_rate
     ↓
     Mark as alive

   Extinction prevention:
     if prey_alive.sum() < 1:
       Respawn 5 random prey in dead slots
     if pred_alive.sum() < 1:
       Respawn 5 random predators
```

## GPU vs CPU Boundaries

### CPU-Only Code
- `brain.py`: Pure NumPy implementation
- `agent.py`: Agent classes with NumPy arrays
- `world.py`: List-based simulation
- `river.py`: NumPy path generation, CPU-side checks
- `main.py`: Pygame visualization

### GPU Code
- `simulation_gpu.py`: PyTorch tensors, CUDA operations
- `NeuralNetBatch`: PyTorch nn.Module
- All agent state as tensors on GPU

### GPU-CPU Transfers
1. **Every render frame**:
   - `get_state_cpu()`: Transfer alive agent positions for rendering
   - ~10K tensors → NumPy: `prey_pos[prey_alive].cpu().numpy()`

2. **River flow computation** (every step):
   - Positions GPU → CPU: `prey_pos[prey_alive].cpu().numpy()`
   - `river.get_flow_at_batch()` on CPU
   - Flows CPU → GPU: `torch.tensor(flows, device='cuda')`

3. **Island checks** (every step):
   - Positions GPU → CPU for `is_on_island()`
   - Creates boolean mask on CPU
   - Mask CPU → GPU: `torch.tensor(on_island, device='cuda')`

### Performance Bottleneck
The river/island CPU checks are the main GPU-CPU transfer bottleneck. With 10K agents:
- 2x per step: prey flow + predator flow
- 2x per step: prey island mods + predator island mods
- Total: 4 × 10K array transfers per step

## Current Coupling & Dependencies

### Tight Coupling

1. **Two-Species Assumption**
   - `world.py`: Hardcoded `self.prey` and `self.predators` lists
   - `simulation_gpu.py`: Separate tensors for prey/pred
   - `Visualizer`: Hardcoded colors/rendering for 2 species
   - `Config`: Separate constants for PREY_* and PRED_*

2. **Observation Hardcoding**
   - `Prey.observe()`: Returns exactly 32 dimensions (5 pred + 3 prey)
   - `Predator.observe()`: Returns exactly 21 dimensions (5 prey + hunger)
   - Brain input_size must match these exactly
   - Cannot add new sensors without changing all agents

3. **Reproduction Logic**
   - `Prey`: Age-based reproduction
   - `Predator`: Energy + cooldown based
   - Different logic for each species (not extensible)

4. **River-Simulation Coupling**
   - River uses CPU arrays
   - GPU simulation transfers positions every step
   - Island modifiers hardcode 'prey' vs 'predator' strings
   - Cannot easily add new environment features

5. **Config-Agent Coupling**
   - Agents import `from config import *`
   - Global constants used directly in code
   - Cannot easily override per-experiment
   - No validation of config values

### Loose Coupling (Good)

1. **Brain-Agent Separation**
   - Brain is independent neural network
   - Agent uses brain via `forward()` interface
   - Can swap brain implementations

2. **World-Agent Separation**
   - World manages agents through abstract interfaces
   - Agents don't know about World internals
   - Vectorized operations keep boundary clean

3. **Visualization-Simulation Separation**
   - Visualizer gets state dict, doesn't modify simulation
   - Can run simulation headless
   - Multiple visualizer implementations

## Module Dependencies

```
Config (config.py)
  ↓
Agent (agent.py) → Brain (brain.py)
  ↓                  ↑
World (world.py) ----+
  ↓
River (river.py)
  ↓
Visualizer (main.py)

GPU Path:
Config → simulation_gpu.py → River
           ↓
        GPUVisualizer (main_gpu.py)

Analysis:
stats.npz → analyze_evolution.py
         → analyze_stats.py
```

### Import Graph

```
config.py: (no imports)

brain.py: numpy

agent.py: numpy, brain, config

river.py: numpy, config

world.py: numpy, agent, river

main.py: pygame, numpy, sys, world

simulation_gpu.py: torch, torch.nn, numpy, config, river

main_gpu.py: pygame, numpy, simulation_gpu

analyze_*.py: numpy, matplotlib, pathlib
```

### Circular Dependencies
None currently. Clean import hierarchy.

## Configuration System

**Current Approach:**
- Single `config.py` file
- Global constants
- Imported with `from config import *`

**Limitations:**
1. Cannot easily run experiments with different configs
2. No config validation
3. No config versioning
4. Difficult to add species-specific configs
5. No runtime config changes

**Benefits:**
- Simple and direct
- Easy to read/modify
- No boilerplate
- Fast access (no lookups)

## Statistics & Logging

**CPU Version** (`world.py`):
- Tracks: prey_count, predator_count, avg_age, avg_fitness
- Methods: `record_stats()`, `save_stats()`, `print_stats()`
- Saves to: `stats.npz`

**GPU Version** (`main_gpu.py`):
- Extended tracking: swim speeds, habitat preferences
- Auto-saves every 1000 steps
- Saves to: `stats_autosave.npz`

**Analysis:**
- `analyze_evolution.py`: Comprehensive plots + speciation detection
- `analyze_stats.py`: Basic 4-plot visualization

## Current Limitations for Extension

### Cannot Easily Add:
1. **New Species**: Hardcoded prey/predator everywhere
2. **New Sensors**: Observation dimensions fixed
3. **New Behaviors**: Reproduction logic species-specific
4. **New Traits**: Must manually add to agent classes
5. **New Environments**: River hardcoded, island modifiers hardcoded
6. **Batch Experiments**: Config system not designed for this

### Can Easily Add:
1. **New Brain Architectures**: Clean interface
2. **New Visualizations**: State dict is flexible
3. **New Analysis**: Stats format is extensible
4. **New Physics**: Contained in update_physics()

## Performance Characteristics

### CPU Version
- **Agents**: 100-500 optimal
- **Bottleneck**: Python loops for agents
- **Optimization**: Vectorized distance calculations
- **FPS**: ~91 FPS with 240 agents (1600x1200)

### GPU Version
- **Agents**: 1000-12000 optimal
- **Bottleneck**: GPU-CPU transfers for river/island
- **Optimization**: Batched neural networks, sampled observations
- **FPS**: ~14 FPS with 12000 agents (3840x2160)

### Scaling Characteristics
- CPU: Linear degradation with agent count (O(n²) collisions mitigated)
- GPU: Sublinear until memory limit, then crashes
- River checks: O(n) CPU overhead per step

## Summary

**Strengths:**
- Clean separation of concerns (mostly)
- Two complementary implementations (CPU/GPU)
- Working neuroevolution with emergent behaviors
- Extensible analysis and visualization
- Well-documented design decisions

**Weaknesses:**
- Hardcoded to exactly 2 species
- Fixed observation dimensions
- Tight coupling between config and agents
- GPU-CPU transfers for environmental features
- No batch experiment infrastructure
- Species-specific reproduction logic

**Next Steps:**
See TECHNICAL_DEBT.md and REFACTOR_PLAN.md for actionable improvements.
