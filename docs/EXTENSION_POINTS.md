# HUNT Extension Points Analysis

This document analyzes the extensibility of the current codebase: what's easy to extend, what requires refactoring, where the natural seams are, and what patterns new code should follow.

## Extension Difficulty Matrix

| Extension Type | Current Difficulty | Blocks | Effort if Refactored |
|----------------|-------------------|--------|---------------------|
| New brain architecture | **Easy** | None | N/A |
| New visualization | **Easy** | None | N/A |
| New analysis metric | **Easy** | None | N/A |
| New physics behavior | **Medium** | None | Easy |
| New environmental feature | **Medium** | GPU transfers | Easy |
| New evolvable trait | **Hard** | Observation dims | Medium |
| New species (3rd, 4th, ...) | **Impossible** | Architecture | Medium |
| New sensor/observation | **Impossible** | Fixed brain inputs | Medium |
| Batch experiments | **Hard** | Config system | Easy |

## Currently Easy to Extend

### 1. Brain Architectures

**Interface**: `brain.py` provides clean contract

**Required methods**:
```python
class CustomBrain:
    def __init__(self, input_size, hidden_size, output_size):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Take observation, return action."""
        pass

    def mutate(self, mutation_rate: float):
        """Modify weights in-place."""
        pass

    def copy(self) -> 'CustomBrain':
        """Deep copy for offspring."""
        pass
```

**Example extensions**:
- Recurrent neural network (LSTM/GRU)
- Convolutional network (for visual input)
- Neuroevolution of augmenting topologies (NEAT)
- Transformer (overkill but possible)
- Hand-coded strategies (no learning)

**How to integrate**:
```python
# agent.py
from custom_brain import CustomBrain

class Agent:
    def __init__(self, ...):
        if brain is None:
            self.brain = CustomBrain(input_size, hidden_size, output_size)
        else:
            self.brain = brain.copy()
```

**Caveats**:
- Must match input_size (currently hardcoded per species)
- Must return 2D output (acceleration vector)
- Mutation must work with offspring reproduction

---

### 2. Visualization Systems

**Interface**: Visualizers receive state dict

**Required**: Read `get_state()` or `get_state_cpu()` output

**Example state dict**:
```python
{
    'timestep': int,
    'prey_positions': np.array (N, 2),
    'predator_positions': np.array (M, 2),
    'prey_count': int,
    'predator_count': int,
    'prey_avg_age': float,
    # ... more stats
}
```

**Example extensions**:
- Web-based visualization (Flask + JS)
- 3D visualization (Panda3D, PyGame3D)
- Headless rendering (PIL/Pillow for images)
- VR visualization (OpenVR)
- Terminal-based ASCII visualization (for SSH sessions)

**How to add**:
```python
class CustomVisualizer:
    def __init__(self, world_or_ecosystem):
        self.sim = world_or_ecosystem

    def draw(self):
        state = self.sim.get_state() # or get_state_cpu()
        # Render however you want
        for pos in state['prey_positions']:
            # Draw prey
        for pos in state['predator_positions']:
            # Draw predator

    def run(self):
        while True:
            self.sim.step()
            self.draw()
```

**Caveats**:
- State dict format might change (no versioning)
- Currently assumes 2 species
- No standardized "color" or "species" field

---

### 3. Analysis Metrics

**Interface**: Stats are saved to `.npz` files

**Current stats**:
```python
# world.py
self.stats = {
    'prey_count': [],
    'predator_count': [],
    'prey_avg_age': [],
    # ... extensible dict
}
```

**Example extensions**:
- Genetic diversity metrics (std of neural weights)
- Spatial clustering analysis
- Predation efficiency (kills per predator)
- Energy economics (energy gained vs spent)
- Coevolutionary dynamics (fitness deltas)
- Speciation detection (already exists in analyze_evolution.py)

**How to add**:
```python
# In world.py or simulation_gpu.py
def record_stats(self):
    self.stats['prey_count'].append(len(self.prey))
    # Add new metric:
    self.stats['spatial_variance'].append(np.var(prey_positions))
```

Then analyze in custom script:
```python
stats = np.load('stats.npz')
spatial_var = stats['spatial_variance']
plt.plot(spatial_var)
```

**Caveats**:
- No schema validation
- Different stats between CPU/GPU versions
- Large arrays can bloat .npz files

---

### 4. Basic Physics Modifications

**Interface**: `update_physics()` method in agents

**Current pattern**:
```python
def update_physics(self, dt=1.0):
    self.vel += self.acc * dt
    # Limit speed
    speed = np.linalg.norm(self.vel)
    if speed > self.max_speed:
        self.vel = (self.vel / speed) * self.max_speed
    # Update position
    self.pos += self.vel * dt
    # Toroidal wrap
    self.pos[0] = self.pos[0] % self.world_width
    self.pos[1] = self.pos[1] % self.world_height
    self.age += 1
```

**Easy modifications**:
- Add friction: `vel *= 0.99`
- Add momentum: `mass` attribute, `acc = force / mass`
- Add drag: `vel -= vel * drag_coefficient * ||vel||`
- Change topology: Bounded instead of toroidal
- Add obstacles: Check collision before position update

**Example**:
```python
def update_physics(self, dt=1.0):
    self.vel += self.acc * dt
    self.vel *= 0.98  # Add friction
    # ... rest same
```

**How to integrate**: Modify `agent.py` or subclass

**Caveats**:
- Must update in both CPU and GPU versions
- GPU version requires tensor operations

---

## Medium Difficulty Extensions

### 5. New Environmental Features

**Current example**: River with flow

**Interface**: World/Ecosystem checks environment and modifies agent state

**Pattern**:
```python
# 1. Define environment class
class NewFeature:
    def __init__(self, world_width, world_height):
        # Initialize

    def affects_agent(self, x, y) -> bool:
        # Check if agent is in affected region
        pass

    def get_effect(self, x, y) -> dict:
        # Return modifiers: {'speed': 0.8, ...}
        pass

# 2. Instantiate in world
self.new_feature = NewFeature(width, height)

# 3. Apply in physics update
for agent in agents:
    if self.new_feature.affects_agent(agent.pos[0], agent.pos[1]):
        effect = self.new_feature.get_effect(agent.pos[0], agent.pos[1])
        # Apply effect
```

**Examples**:
- Temperature zones (hot zones drain energy faster)
- Terrain elevation (hills slow movement)
- Weather (wind affects velocity)
- Resources (food patches, water sources)
- Hazards (traps, predator nests)

**Current barrier**: GPU-CPU transfers

If environment needs per-agent checks:
1. CPU version: Easy, just check in Python loop
2. GPU version: Requires transferring positions to CPU, checking, transferring effects back

**Solutions**:
- Port environment to GPU (PyTorch tensors)
- Use spatial grids for fast lookups
- Check only when agents move significantly

**Effort**: Small (CPU), Medium (GPU)

---

### 6. New Death Conditions

**Current**: Age and energy (predators)

**Interface**: `should_die()` method

**Pattern**:
```python
class Prey:
    def should_die(self) -> bool:
        # Add new condition
        return self.age >= self.max_lifespan or self.some_new_condition()
```

**Examples**:
- Disease (random chance per timestep)
- Poisoning (from toxic food)
- Injury (accumulates from near-misses)
- Hypothermia (if in cold zones)

**Barrier**: Must update both CPU and GPU versions

**GPU challenge**: Boolean logic on tensors
```python
# GPU version
self.prey_alive &= (self.prey_age < max_age) & ~self.prey_diseased
```

**Effort**: Small

---

## Hard Extensions (Require Refactoring)

### 7. New Evolvable Traits

**Current**: `swim_speed` is evolvable

**Requirements**:
1. Add trait to agent initialization
2. Inherit from parent with mutation
3. Use trait in simulation logic
4. Track in statistics

**Barriers**:
- Must update multiple files (agent.py, world.py, simulation_gpu.py, config.py)
- GPU version requires new tensor
- No systematic way to define "evolvable trait"

**Example (adding "vision_range")**:

```python
# config.py
PREY_VISION_RANGE = 100.0

# agent.py (CPU)
class Prey:
    def __init__(self, ..., vision_range=None):
        if vision_range is None:
            self.vision_range = PREY_VISION_RANGE + np.random.randn() * 10
        else:
            self.vision_range = vision_range

    def reproduce(self, mutation_rate):
        child_vision = self.vision_range + np.random.randn() * mutation_rate * 20
        child = Prey(..., vision_range=child_vision)
        # ...

    def observe(self, ...):
        # Use self.vision_range to filter visible agents
        distances, vectors = vectorized_distances(...)
        visible = distances < self.vision_range
        # ...

# simulation_gpu.py
class GPUEcosystem:
    def __init__(self, ...):
        self.prey_vision_range = torch.normal(
            PREY_VISION_RANGE, 10.0,
            size=(num_prey,), device=device
        )

    def observe_prey(self):
        # Filter by vision range
        visible = distances < self.prey_vision_range.unsqueeze(1)
        # ...

    def step(self, ...):
        # Reproduction: inherit and mutate
        child_vision = parent_vision[parents] + torch.randn(...) * mutation_rate * 20
        self.prey_vision_range[dead_indices] = child_vision
```

**Required changes**: 8-10 locations across 4 files

**Effort**: Medium (2-3 hours per trait)

**Improvement needed**: Trait definition system

---

### 8. New Sensors (Observations)

**Current**: Prey sees 5 predators + 3 prey, Predator sees 5 prey + hunger

**To add new sensor** (e.g., "distance to nearest river"):
1. Compute sensor value in `observe()`
2. Append to observation vector
3. Increase observation dimension
4. Update brain input size
5. Update ALL agents to match new dimension
6. Update GPU observation tensor size

**Barriers**:
- **Fixed observation dimensions** are hardcoded
- Brain input_size must match exactly
- All agents of same species must have same observations
- No way to have "optional" sensors

**Example**: Add "smell predator scent"

**What breaks**:
```python
# Current: Prey observations are 32-dim
class Prey:
    def __init__(self, ...):
        super().__init__(..., input_size=32)  # Hardcoded!

    def observe(self, ...):
        # ... compute 32 values
        return np.array(observation, dtype=np.float32)  # Must be 32!

# If we add +1 sensor:
observation.append(scent_value)  # Now 33 dims
return np.array(observation, dtype=np.float32)  # Brain expects 32, CRASH

# Must update:
super().__init__(..., input_size=33)  # Every Prey
```

**Impact**: Cannot easily experiment with sensors

**Solution needed**: Dynamic observation system

**Effort**: Large (requires architecture change)

---

### 9. Batch Experiment Runner

**Goal**: Run experiments with different configs

**Current barrier**: Global config in `config.py`

**Desired**:
```python
experiments = [
    {'PREY_MAX_SPEED': 2.0, 'PRED_MAX_SPEED': 2.5},
    {'PREY_MAX_SPEED': 3.0, 'PRED_MAX_SPEED': 2.5},
    {'PREY_MAX_SPEED': 4.0, 'PRED_MAX_SPEED': 2.5'},
]

for exp in experiments:
    run_experiment(config=exp, output=f'exp_{exp["PREY_MAX_SPEED"]}.npz')
```

**Problems**:
1. Config is imported globally with `from config import *`
2. No way to override at runtime
3. Agents reference global constants directly
4. No config validation or schema

**Solution**: Config object passed to agents

**Effort**: Medium (refactor config system)

---

## Currently Impossible Extensions

### 10. Adding a Third Species

**Goal**: Add "scavengers" that eat dead prey

**Barriers**:
1. **World.py hardcodes two lists**:
   ```python
   self.prey = []
   self.predators = []
   # Where to add self.scavengers?
   ```

2. **Step loop assumes two species**:
   ```python
   prey_positions = np.array([p.pos for p in self.prey])
   predator_positions = np.array([p.pos for p in self.predators])
   # No pattern for N species
   ```

3. **Observations hardcoded**:
   ```python
   # Prey observe predators + other prey
   # What if scavengers also hunt prey?
   # Do prey observe scavengers?
   ```

4. **Visualizer hardcoded**:
   ```python
   self.prey_color = (50, 255, 50)
   self.predator_color = (255, 50, 50)
   # What color for scavengers?
   ```

5. **Config hardcoded**:
   ```python
   PREY_MAX_SPEED = 3.0
   PRED_MAX_SPEED = 2.5
   # SCAV_MAX_SPEED = ???
   ```

6. **GPU version even more hardcoded**:
   ```python
   self.prey_pos = torch.rand(num_prey, 2, ...)
   self.pred_pos = torch.rand(num_predators, 2, ...)
   # Need self.scav_pos = ...
   # And self.scav_brain = ...
   # And in step(): observe_scavengers(), ...
   ```

**Required changes**:
- Refactor to list/dict of species
- Dynamic observation system
- Flexible visualizer
- Per-species configs
- Reproduction must handle any species
- Statistics must scale to N species

**Effort**: Large (1-2 weeks)

---

### 11. Variable-Length Observations

**Goal**: "Prey can see ALL nearby predators, not just 5"

**Current limitation**:
```python
# Brain expects fixed input size
brain = Brain(input_size=32, ...)

# Cannot pass variable-length observation
observation = [...]  # Could be 10 predators, or 3, or 0
brain.forward(observation)  # Requires exactly 32 values
```

**Solutions**:
1. Attention mechanism (transformers)
2. Pad/truncate to fixed size (current approach)
3. Recurrent aggregation (RNN over neighbors)
4. Graph neural network (edges = nearby agents)

**Effort**: Large (requires new brain architecture + observation system)

---

## Natural Extension Seams

### Where to Add Species

**Ideal architecture** (post-refactor):
```python
class Species:
    name: str
    color: tuple
    initial_count: int
    agent_class: type  # Prey, Predator, or custom
    config: SpeciesConfig

class World:
    def __init__(self, species_list: List[Species]):
        self.species = {s.name: [] for s in species_list}

    def step(self):
        for species_name, agents in self.species.items():
            # Generic loop
            for agent in agents:
                agent.step()
```

**Files to change**:
- `world.py`: Replace `self.prey`/`self.predators` with `self.species`
- `agent.py`: Make `Agent` fully generic
- `config.py`: Define `SpeciesConfig` class
- Visualizers: Loop over `species`, use `species.color`

---

### Where to Add Features (Traits)

**Ideal architecture**:
```python
class Trait:
    name: str
    initial_value: float
    mutation_std: float
    min_value: float
    max_value: float

class Agent:
    def __init__(self, traits: Dict[str, Trait]):
        self.traits = {name: trait.sample() for name, trait in traits.items()}

    def reproduce(self):
        child_traits = {
            name: mutate(value, trait.mutation_std)
            for name, (value, trait) in zip(self.traits.items(), parent_traits.items())
        }
        return self.__class__(traits=child_traits, ...)
```

**Usage**:
```python
prey_traits = {
    'speed': Trait('speed', initial=3.0, mutation_std=0.2, min=0.5, max=10.0),
    'vision': Trait('vision', initial=100.0, mutation_std=10.0, min=10, max=500),
}

prey = Prey(traits=prey_traits, ...)
```

**Files to change**:
- `agent.py`: Add trait system
- `config.py`: Define trait configs
- `simulation_gpu.py`: Store traits as separate tensors

---

### Where to Add Environment Features

**Pattern**:
```python
class Environment:
    def __init__(self, features: List[EnvironmentFeature]):
        self.features = features

    def apply_effects(self, agent):
        for feature in self.features:
            if feature.affects(agent.pos):
                effect = feature.get_effect(agent.pos)
                agent.apply_effect(effect)

class EnvironmentFeature:
    def affects(self, pos) -> bool:
        pass

    def get_effect(self, pos) -> dict:
        pass
```

**Examples**:
```python
river = River(width, height, ...)
temperature = TemperatureMap(width, height, hot_zones=[...])
hazards = HazardZones(trap_locations=[...])

env = Environment(features=[river, temperature, hazards])
```

**Files to change**:
- `world.py`: Use `self.environment` instead of `self.river`
- `river.py`: Inherit from `EnvironmentFeature`
- New files for new features

---

### Where to Add Observation Sensors

**Ideal architecture**:
```python
class Sensor:
    name: str
    dimension: int

    def observe(self, agent, world) -> np.ndarray:
        # Return observation vector of size `dimension`
        pass

class Agent:
    def __init__(self, sensors: List[Sensor]):
        self.sensors = sensors
        self.input_size = sum(s.dimension for s in sensors)
        self.brain = Brain(input_size=self.input_size, ...)

    def observe(self, world):
        observations = [sensor.observe(self, world) for sensor in self.sensors]
        return np.concatenate(observations)
```

**Examples**:
```python
nearest_predators = NearestAgentsSensor(agent_type='predator', count=5, features=4)
nearest_prey = NearestAgentsSensor(agent_type='prey', count=3, features=4)
hunger_sensor = HungerSensor(dimension=1)

prey = Prey(sensors=[nearest_predators, nearest_prey])
predator = Predator(sensors=[nearest_prey, hunger_sensor])
```

**Benefits**:
- Add sensors without changing observation dimensions manually
- Mix and match sensors per species
- Easy to experiment ("what if prey could smell predator energy?")

**Files to change**:
- `agent.py`: Add sensor system
- New `sensors.py` module

---

## Recommended Extension Pattern

### For New Code, Follow These Patterns

1. **Configuration**:
   ```python
   # Don't:
   MAGIC_NUMBER = 42

   # Do:
   class Config:
       magic_number: int = 42  # Description of what this controls
   ```

2. **Species**:
   ```python
   # Don't:
   if species == 'prey':
       # ...
   elif species == 'predator':
       # ...

   # Do:
   for species_name, agents in world.species.items():
       # Generic code
   ```

3. **Traits**:
   ```python
   # Don't:
   self.swim_speed = PREY_SWIM_SPEED + np.random.randn() * 0.2

   # Do:
   self.traits['swim_speed'] = traits.sample('swim_speed')
   ```

4. **Observations**:
   ```python
   # Don't:
   observation = np.zeros(32)  # Hardcoded size
   observation[0:20] = predator_info
   observation[20:32] = prey_info

   # Do:
   observation_parts = [sensor.observe(agent, world) for sensor in self.sensors]
   observation = np.concatenate(observation_parts)
   ```

5. **Environment**:
   ```python
   # Don't:
   if self.river.is_in_river(x, y):
       # Apply river effect
   if self.temperature.is_hot(x, y):
       # Apply temperature effect

   # Do:
   for feature in self.environment.features:
       if feature.affects(agent.pos):
           agent.apply_effect(feature.get_effect(agent.pos))
   ```

6. **Statistics**:
   ```python
   # Don't:
   self.stats['prey_count'] = [...]
   self.stats['predator_count'] = [...]

   # Do:
   self.stats = {
       species_name: {
           'count': [],
           'avg_age': [],
           # ...
       }
       for species_name in self.species.keys()
   }
   ```

---

## Easy vs Hard Summary

### ✅ Currently Easy
- Brain architectures (clean interface)
- Visualizations (state dict decoupling)
- Analysis metrics (extensible dict)
- Basic physics (contained methods)

### ⚠️ Medium Difficulty
- New environmental features (GPU transfer overhead)
- New death conditions (update CPU + GPU)
- Evolvable traits (multiple file changes)
- Batch experiments (config refactor needed)

### ❌ Currently Hard/Impossible
- Adding 3rd+ species (architecture blocks)
- New sensors (fixed observation dims)
- Variable-length observations (brain architecture)
- Per-species unique mechanics (hardcoded logic)

---

## Next Steps

See **REFACTOR_PLAN.md** for prioritized refactoring to unblock these extensions.

See **EXTENSION_ARCHITECTURE.md** for proposed patterns to enable N-species, traits, and experiments.
