# Boundary Modes: Toroidal vs Bounded

HUNT supports two different world boundary behaviors: **toroidal** (wrap-around) and **bounded** (walled). This document explains the differences and when to use each.

## Overview

The boundary mode determines what happens when agents reach the edge of the world:

- **Toroidal**: Agents wrap around to the opposite edge (like Pac-Man)
- **Bounded**: Agents are stopped by walls at the edges

## Toroidal Mode (Default)

### Behavior

In toroidal mode, the world "wraps around" at the edges:

```
  ┌──────────────────┐
  │                  │
  │   Agent →        │ ← wraps to here
  │                  │
  └──────────────────┘
  ← wraps from here
```

- Agent at position (width+10, y) wraps to (10, y)
- Agent at position (x, -5) wraps to (x, height-5)
- Distances consider the shortest path (which may wrap around)

### Distance Calculation

```python
# Distance between agents near opposite edges:
agent1_pos = (10, 50)
agent2_pos = (790, 50)

# In 800-wide world:
# - Direct distance: 780 pixels
# - Wrapped distance: 20 pixels (shorter!)

# toroidal_distance_numpy chooses wrapped: 20 pixels
```

### Use Cases

Toroidal mode is best for:
- **Studying pure population dynamics** without edge effects
- **Avoiding corner/edge artifacts** in evolution
- **Continuous environments** (simulating infinite space)
- **Traditional predator-prey models** (most research uses toroidal)

### Advantages

- No edge bias (all positions equivalent)
- No corners where agents can get "trapped"
- Simpler analysis (no boundary-specific behaviors)
- Matches most theoretical models

### Disadvantages

- Less intuitive (harder to visualize)
- Not realistic for bounded habitats
- Can't study edge/territory behaviors

## Bounded Mode

### Behavior

In bounded mode, the world has solid walls:

```
  ┌──────────────────┐
  │                  │
  │   Agent → ║      │ ← hits wall, stops
  │                  │
  └──────────────────┘
```

- Agent positions are clamped to [0, width] × [0, height]
- Distances are calculated along straight lines (no wrapping)
- Agents can learn wall-avoidance behaviors

### Distance Calculation

```python
# Distance between agents near opposite edges:
agent1_pos = (10, 50)
agent2_pos = (790, 50)

# In 800-wide world:
# - Only one path: 780 pixels (no wrapping)

# bounded_distance_numpy returns: 780 pixels
```

### Use Cases

Bounded mode is best for:
- **Studying territoriality** and edge behaviors
- **Realistic habitats** (islands, enclosed environments)
- **Testing wall-avoidance** evolution
- **Spatial competition** for corners and edges

### Advantages

- More intuitive visualization
- Realistic for actual habitats
- Agents can evolve wall-avoidance
- Can study territorial behaviors

### Disadvantages

- Edge effects bias evolution
- Corners create "safe zones"
- Less mathematically clean
- Need to consider boundary effects in analysis

## Wall Proximity Sensor

In bounded mode, agents can optionally sense walls:

```python
from config_new import ObservationConfig

observation = ObservationConfig(
    observe_species={'prey': 5},
    sense_hunger=True,
    sense_walls=True  # Only meaningful in bounded mode
)
```

The wall proximity sensor provides 4 values:
- Distance to left wall (normalized)
- Distance to right wall (normalized)
- Distance to top wall (normalized)
- Distance to bottom wall (normalized)

This allows agents to evolve wall-avoidance behaviors through neuroevolution.

## Configuration

### Setting Boundary Mode

```python
from config_new import SimulationConfig, WorldConfig, BoundaryMode

# Toroidal (default)
config = SimulationConfig.default_two_species()
# config.world.boundary_mode == BoundaryMode.TOROIDAL

# Bounded
config_bounded = SimulationConfig.default_bounded()
# config_bounded.world.boundary_mode == BoundaryMode.BOUNDED
```

### Manual Configuration

```python
from config_new import WorldConfig, BoundaryMode

# Toroidal world
world_toroidal = WorldConfig(
    width=800,
    height=600,
    boundary_mode=BoundaryMode.TOROIDAL
)

# Bounded world
world_bounded = WorldConfig(
    width=800,
    height=600,
    boundary_mode=BoundaryMode.BOUNDED
)
```

### JSON Configuration

```json
{
  "world": {
    "width": 800,
    "height": 600,
    "boundary_mode": "bounded"
  },
  "species": [...],
  "interactions": [...]
}
```

## Implementation Details

### Position Updates

**Toroidal:**
```python
# After movement
new_pos = old_pos + velocity
wrapped_pos = new_pos % [world_width, world_height]
agent.pos = wrapped_pos
```

**Bounded:**
```python
# After movement
new_pos = old_pos + velocity
clamped_pos = np.clip(new_pos, [0, 0], [world_width, world_height])
agent.pos = clamped_pos
```

### Distance Functions

```python
from utils import toroidal_distance_numpy, bounded_distance_numpy

# Toroidal (considers wrapping)
dist, vec = toroidal_distance_numpy(pos1, pos2, width, height)

# Bounded (no wrapping)
dist, vec = bounded_distance_numpy(pos1, pos2, width, height)
```

Both functions have identical interfaces, making it easy to switch modes.

### GPU Support

Both modes have GPU implementations in PyTorch:

```python
from utils import toroidal_distance_torch, bounded_distance_torch

# GPU tensors
pos1_gpu = torch.tensor([[10, 20]], device='cuda')
pos2_gpu = torch.tensor([[30, 40]], device='cuda')

# Toroidal on GPU
dist, vec = toroidal_distance_torch(pos1_gpu, pos2_gpu, width, height)

# Bounded on GPU
dist, vec = bounded_distance_torch(pos1_gpu, pos2_gpu, width, height)
```

## Comparison Table

| Aspect | Toroidal | Bounded |
|--------|----------|---------|
| Edges | Wrap around | Solid walls |
| Corners | None (topology is a torus) | 4 corners exist |
| Distance | Shortest path (may wrap) | Straight line only |
| Edge effects | None | Significant |
| Realism | Low | High |
| Analysis | Simpler | More complex |
| Wall sensing | N/A | Optional |
| Evolution bias | Uniform | Edge-dependent |

## Experimental Design

### When to Use Toroidal

Use toroidal mode when:
- You want to study "pure" population dynamics
- You're comparing to theoretical models (most use toroidal)
- You want to avoid confounding edge effects
- You're studying predator-prey cycles, coevolution, etc.

### When to Use Bounded

Use bounded mode when:
- You're modeling a specific real habitat
- You want to study territorial behaviors
- You're testing wall-avoidance evolution
- You want more realistic constraints

### Running Both

For robust results, consider running experiments in both modes:

```python
# Run same experiment in both modes
results_toroidal = run_experiment(config_toroidal)
results_bounded = run_experiment(config_bounded)

# Compare outcomes
compare_results(results_toroidal, results_bounded)
```

This helps determine if your findings are robust to boundary conditions.

## Performance

Both modes have similar performance:
- CPU: Slight overhead for toroidal wrapping (negligible)
- GPU: No measurable difference (both vectorized)

The choice should be based on experimental goals, not performance.

## Evolution Differences

Agents evolve different strategies in each mode:

**Toroidal:**
- No wall-avoidance needed
- Uniform space utilization
- Pure pursuit/evasion behaviors

**Bounded:**
- Wall-avoidance evolves if wall sensor enabled
- Agents may learn to use corners
- Territorial behaviors may emerge

## Migration Between Modes

To convert an existing config:

```python
# Load existing config
config = SimulationConfig.from_json_file('my_config.json')

# Change boundary mode
config.world.boundary_mode = BoundaryMode.BOUNDED

# Add wall sensor to species (optional)
for species in config.species:
    species.observation.sense_walls = True

# Save new config
config.to_json_file('my_config_bounded.json')
```

## Testing

See `tests/test_boundary_modes.py` for comprehensive tests:
- Position wrapping/clamping
- Distance calculations
- Corner behaviors
- NumPy/PyTorch consistency

Run tests:
```bash
python3 tests/test_boundary_modes.py
```

## Visualization

In the GUI:
- **Toroidal**: No visual boundaries (agents wrap around)
- **Bounded**: Walls drawn at edges (optional)

## References

- Classic predator-prey models use toroidal boundaries (Lotka-Volterra)
- Territorial animal studies use bounded environments
- Most ALife research uses toroidal for clean analysis
- Real-world applications should use bounded for realism

## Next Steps

- See `ADDING_SPECIES.md` for species configuration
- See `tests/test_boundary_modes.py` for code examples
- See `config_new.py` for `BoundaryMode` enum definition
