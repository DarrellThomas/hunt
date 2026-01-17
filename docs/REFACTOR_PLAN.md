# HUNT Refactor Plan

Prioritized refactoring plan to enable platform extensibility. Items ordered by: **Priority** (Critical > Important > Nice-to-have), then by **Dependency** (blocking items first), then by **Risk**.

## Priority Levels

- **Critical**: Must fix before adding N-species or new features
- **Important**: Will cause significant pain if not fixed
- **Nice-to-have**: Improves code quality but not blocking

## Critical Refactors (Must Do First)

### 1. Fix GPU Neuroevolution Bug ⚠️

**Priority**: Critical
**Effort**: Large (1-2 weeks)
**Risk**: High (core algorithm)

**Problem**: GPU version doesn't implement proper neuroevolution (see TECHNICAL_DEBT.md #6)

**Current behavior**:
- All agents share one global network
- Network mutates every 50 steps globally
- No per-agent genetic inheritance

**Correct behavior** (like CPU version):
- Each agent has individual brain weights
- Offspring inherit parent's weights
- Mutation happens at reproduction

**Solution Options**:

**Option A: Per-Agent Weight Storage**
```python
class GPUEcosystem:
    def __init__(self, ...):
        # Store weights per agent
        # Prey brain: 32→32→32→2 = ~3K params per agent
        self.prey_brain_weights = torch.randn(num_prey, weight_count, device='cuda')

    def forward_pass(self):
        # Batch process with per-agent weights
        for agent_idx in range(num_prey):
            # This is slow! Need better approach
```

**Option B: Population of Networks**
```python
# Use multiple small networks instead of one big one
self.prey_brains = [SmallNetwork().to('cuda') for _ in range(num_prey)]
# Still slow due to Python loop
```

**Option C: Hypernetwork**
```python
# Use hypernetwork to generate per-agent weights
self.weight_generator = HyperNetwork(agent_id → weights)
# Complex but maintains GPU parallelism
```

**Recommended**: Option A with optimized batching

**Steps**:
1. Implement per-agent weight storage in GPU tensors
2. Modify forward pass to use per-agent weights (batched)
3. Implement reproduction with weight inheritance
4. Implement mutation at reproduction time
5. Remove periodic global mutation
6. Verify CPU/GPU produce similar evolutionary dynamics
7. Add tests comparing CPU vs GPU evolution

**Tests needed**:
- CPU and GPU produce similar fitness curves
- Offspring inherit parent weights
- Mutation creates diversity

**Risk mitigation**:
- Keep old implementation in separate branch
- Gradual migration with A/B testing
- Profile to ensure performance doesn't degrade

---

### 2. Refactor to N-Species Architecture

**Priority**: Critical
**Effort**: Large (2 weeks)
**Risk**: Medium (large refactor but well-understood)

**Current**: Hardcoded `self.prey` and `self.predators`
**Target**: `self.species: Dict[str, List[Agent]]`

**Phase 1: Introduce Species Abstraction** (3 days)

```python
# New file: species.py
@dataclass
class SpeciesConfig:
    name: str
    initial_count: int
    agent_class: Type[Agent]
    color: Tuple[int, int, int]
    max_speed: float
    max_acceleration: float
    # ... all species-specific params

class SpeciesManager:
    def __init__(self, species_configs: List[SpeciesConfig]):
        self.species = {
            config.name: [
                config.agent_class.from_config(config)
                for _ in range(config.initial_count)
            ]
            for config in species_configs
        }

    def get_all_agents(self) -> List[Agent]:
        return [agent for agents in self.species.values() for agent in agents]

    def get_species(self, name: str) -> List[Agent]:
        return self.species[name]
```

**Phase 2: Migrate World.py** (4 days)

```python
# world.py
class World:
    def __init__(self, width, height, species_configs):
        self.species_manager = SpeciesManager(species_configs)

    def step(self, mutation_rate=0.1):
        # Generic loop instead of separate prey/predator logic
        all_agents = self.species_manager.get_all_agents()

        # Compute positions for all species
        positions_by_species = {
            name: np.array([a.pos for a in agents])
            for name, agents in self.species_manager.species.items()
        }

        # Observations (each agent observes relevant species)
        for agent in all_agents:
            observation = agent.observe(positions_by_species, agent_index)
            agent.act(observation)

        # ... rest of step
```

**Phase 3: Update Observations** (3 days)

```python
class Agent:
    # Define which species this agent observes
    observes_species: List[str] = []  # e.g., ['predator', 'prey']
    observe_counts: Dict[str, int] = {}  # e.g., {'predator': 5, 'prey': 3}

    def observe(self, positions_by_species, my_species, my_index):
        observation = []
        for species_name in self.observes_species:
            count = self.observe_counts[species_name]
            positions = positions_by_species[species_name]
            # Find nearest count agents of this species
            # Add to observation
        return np.array(observation)
```

**Phase 4: Migrate GPU Version** (4 days)

Similar refactor for `simulation_gpu.py`:
- Dictionary of tensors per species
- Dynamic observation computation
- Generic reproduction logic

**Testing Strategy**:
1. Write tests for 2-species (should behave identically to current)
2. Add 3rd species test (scavengers)
3. Verify population dynamics stable
4. Performance benchmark (should not regress)

**Risk Mitigation**:
- Keep current implementation in `world_legacy.py`
- Incremental migration with compatibility layer
- Extensive testing at each phase

---

### 3. Dynamic Observation System

**Priority**: Critical
**Effort**: Medium (1 week)
**Risk**: Low (well-contained change)

**Problem**: Observation dimensions hardcoded, blocks new sensors

**Solution**: Sensor composition

```python
# New file: sensors.py
class Sensor(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def observe(self, agent: Agent, world: World) -> np.ndarray:
        pass

class NearestAgentsSensor(Sensor):
    def __init__(self, species_name: str, count: int, features_per_agent: int = 4):
        self.species_name = species_name
        self.count = count
        self.features = features_per_agent

    @property
    def dimension(self) -> int:
        return self.count * self.features

    def observe(self, agent, world):
        # Find nearest `count` agents of `species_name`
        # Return (count * features) array
        positions = world.get_species_positions(self.species_name)
        distances, vectors = vectorized_distances(agent.pos, positions, ...)
        nearest_indices = np.argsort(distances)[:self.count]
        # Pack into array
        return observation_array

class HungerSensor(Sensor):
    @property
    def dimension(self) -> int:
        return 1

    def observe(self, agent, world):
        if hasattr(agent, 'energy'):
            return np.array([1.0 - agent.energy / agent.max_energy])
        return np.array([0.0])

# In agent.py
class Agent:
    def __init__(self, sensors: List[Sensor], ...):
        self.sensors = sensors
        self.input_size = sum(s.dimension for s in sensors)
        self.brain = Brain(input_size=self.input_size, ...)

    def observe(self, world):
        observations = [sensor.observe(self, world) for sensor in self.sensors]
        return np.concatenate(observations)
```

**Usage**:
```python
prey_sensors = [
    NearestAgentsSensor('predator', count=5),
    NearestAgentsSensor('prey', count=3),
]
predator_sensors = [
    NearestAgentsSensor('prey', count=5),
    HungerSensor(),
]

prey = Prey(sensors=prey_sensors, ...)
predator = Predator(sensors=predator_sensors, ...)
```

**Migration Steps**:
1. Implement `Sensor` base class and basic sensors
2. Add `sensors` parameter to `Agent.__init__`
3. Migrate `Prey.observe()` to use sensors
4. Migrate `Predator.observe()` to use sensors
5. Remove hardcoded observation logic
6. Add tests for sensor composition

**Benefits**:
- Easy to add new sensors
- Easy to experiment (swap sensors without code changes)
- Species-specific observations naturally supported

---

### 4. Refactor Configuration System

**Priority**: Critical
**Effort**: Medium (1 week)
**Risk**: Low (systematic change)

**Current**: Global constants in `config.py`
**Target**: Per-species config objects

```python
# config.py
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PhysicsConfig:
    max_speed: float
    max_acceleration: float
    swim_speed: float

@dataclass
class LifecycleConfig:
    max_lifespan: int
    lifespan_variance: int
    reproduction_age: int
    reproduction_variance: int

@dataclass
class PredatorConfig(LifecycleConfig):
    max_energy: float
    energy_cost: float
    energy_gain: float
    reproduction_threshold: float
    reproduction_cost: float
    reproduction_cooldown: int
    cooldown_variance: int

@dataclass
class SpeciesConfig:
    name: str
    physics: PhysicsConfig
    lifecycle: LifecycleConfig  # or PredatorConfig
    initial_count: int
    agent_class: type
    color: Tuple[int, int, int]

@dataclass
class WorldConfig:
    width: int
    height: int
    catch_radius: float
    species: List[SpeciesConfig]

@dataclass
class RiverConfig:
    enabled: bool
    width: float
    flow_speed: float
    curviness: float
    split: bool
    split_start: float
    split_end: float
    island_width: float

@dataclass
class SimulationConfig:
    world: WorldConfig
    river: RiverConfig
    mutation_rate: float
```

**Usage**:
```python
# Create configs
prey_config = SpeciesConfig(
    name='prey',
    physics=PhysicsConfig(max_speed=3.0, max_acceleration=0.5, swim_speed=2.0),
    lifecycle=LifecycleConfig(max_lifespan=500, ...),
    initial_count=200,
    agent_class=Prey,
    color=(50, 255, 50),
)

predator_config = SpeciesConfig(
    name='predator',
    physics=PhysicsConfig(max_speed=2.5, ...),
    lifecycle=PredatorConfig(max_energy=150, ...),
    initial_count=40,
    agent_class=Predator,
    color=(255, 50, 50),
)

config = SimulationConfig(
    world=WorldConfig(width=1600, height=1200, species=[prey_config, predator_config]),
    river=RiverConfig(...),
    mutation_rate=0.1,
)

# Run simulation
world = World.from_config(config)
```

**Benefits**:
- Easy to define new species (just create new SpeciesConfig)
- Batch experiments: loop over different configs
- Validation built-in (dataclasses + type hints)
- Serializable (save/load configs)

**Migration Steps**:
1. Define config dataclasses
2. Add `from_config()` class methods to World, Agent classes
3. Update existing code to use config objects instead of globals
4. Add config validation
5. Add config save/load (JSON/YAML)
6. Update tests and examples

---

### 5. Unify CPU and GPU Island Mechanics

**Priority**: Critical
**Effort**: Small (2 days)
**Risk**: Low (additive change)

**Problem**: Island behavior modifiers only in GPU version

**Solution**: Integrate into CPU version

```python
# world.py
def step(self, mutation_rate=0.1):
    # ... existing code ...

    # After physics update, apply island modifiers
    for agent in self.prey:
        modifiers = self.river.island_behavior('prey', agent.pos[0], agent.pos[1])
        if modifiers:
            # Speed already applied in max_speed check
            # Apply reproduction modifier
            agent.effective_reproduction_age = (
                agent.reproduction_age * modifiers['reproduction_multiplier']
            )

    for agent in self.predators:
        modifiers = self.river.island_behavior('predator', agent.pos[0], agent.pos[1])
        if modifiers:
            # Speed already handled
            # Apply hunger modifier
            agent.energy -= agent.energy_cost_per_step * (modifiers['hunger_multiplier'] - 1.0)
            # Apply reproduction modifier
            agent.effective_reproduction_cooldown = (
                agent.reproduction_cooldown * modifiers['reproduction_multiplier']
            )
```

**Testing**:
- Verify CPU and GPU have same island effects
- Test edge cases (agent on island boundary)

---

##Important Refactors (High Value)

### 6. Extract Shared Utilities

**Priority**: Important
**Effort**: Small (2 days)
**Risk**: Low (pure refactor)

**Problem**: Distance calculations duplicated in 3 places

**Solution**: Create `utils.py`

```python
# utils.py
def toroidal_distance_numpy(pos1: np.ndarray, pos2: np.ndarray,
                             world_width: float, world_height: float):
    """Compute toroidal distances using NumPy."""
    dx = pos2[:, 0] - pos1[0]
    dy = pos2[:, 1] - pos1[1]
    dx = np.where(np.abs(dx) > world_width / 2,
                  dx - np.sign(dx) * world_width, dx)
    dy = np.where(np.abs(dy) > world_height / 2,
                  dy - np.sign(dy) * world_height, dy)
    distances = np.sqrt(dx**2 + dy**2)
    vectors = np.column_stack([dx, dy])
    return distances, vectors

def toroidal_distance_torch(pos1: torch.Tensor, pos2: torch.Tensor,
                             world_width: float, world_height: float):
    """Compute toroidal distances using PyTorch."""
    diff = pos1.unsqueeze(1) - pos2.unsqueeze(0)
    diff[:, :, 0] = torch.where(
        torch.abs(diff[:, :, 0]) > world_width / 2,
        diff[:, :, 0] - torch.sign(diff[:, :, 0]) * world_width,
        diff[:, :, 0]
    )
    diff[:, :, 1] = torch.where(
        torch.abs(diff[:, :, 1]) > world_height / 2,
        diff[:, :, 1] - torch.sign(diff[:, :, 1]) * world_height,
        diff[:, :, 1]
    )
    distances = torch.norm(diff, dim=2)
    return distances, diff

def spawn_offset(count: int, min_distance: float = 20, max_distance: float = 150,
                 framework='numpy'):
    """Generate random spawn offsets."""
    if framework == 'numpy':
        distance = np.random.uniform(min_distance, max_distance, size=count)
        angle = np.random.uniform(0, 2 * np.pi, size=count)
        offset_x = distance * np.cos(angle)
        offset_y = distance * np.sin(angle)
    elif framework == 'torch':
        distance = torch.rand(count) * (max_distance - min_distance) + min_distance
        angle = torch.rand(count) * 2 * torch.pi
        offset_x = distance * torch.cos(angle)
        offset_y = distance * torch.sin(angle)
    return np.column_stack([offset_x, offset_y]) if framework == 'numpy' else torch.stack([offset_x, offset_y], dim=1)
```

**Migration**:
- Replace all distance calc code with utility calls
- Remove duplicated logic
- Add unit tests for utilities

---

### 7. Port River to GPU

**Priority**: Important
**Effort**: Medium (4 days)
**Risk**: Medium (performance-critical)

**Problem**: GPU-CPU transfers for river checks are bottleneck

**Solution**: Implement river logic in PyTorch

```python
# river_gpu.py
class RiverGPU:
    def __init__(self, world_width, world_height, device='cuda'):
        # Convert path to GPU tensors
        self.path_x = torch.tensor(path_x, device=device)
        self.path_y = torch.tensor(path_y, device=device)
        self.flow_dir_x = torch.tensor(flow_dir_x, device=device)
        self.flow_dir_y = torch.tensor(flow_dir_y, device=device)

    def get_flow_at_batch_gpu(self, positions: torch.Tensor) -> torch.Tensor:
        """Fully GPU-resident flow computation."""
        # Compute distances to all path points (vectorized)
        distances = torch.cdist(positions, torch.stack([self.path_x, self.path_y], dim=1))
        # Find nearest
        nearest_idx = torch.argmin(distances, dim=1)
        # Get flow directions
        flow_x = self.flow_dir_x[nearest_idx] * self.flow_speed
        flow_y = self.flow_dir_y[nearest_idx] * self.flow_speed
        return torch.stack([flow_x, flow_y], dim=1)

    def is_on_island_batch_gpu(self, positions: torch.Tensor) -> torch.Tensor:
        """GPU-resident island detection."""
        distances = torch.cdist(positions, torch.stack([self.path_x, self.path_y], dim=1))
        nearest_idx = torch.argmin(distances, dim=1)
        t = nearest_idx.float() / len(self.path_x)
        in_split_region = (t >= self.split_start) & (t <= self.split_end)
        center_y = self.path_y[nearest_idx]
        dist_from_center = torch.abs(positions[:, 1] - center_y)
        on_island = (dist_from_center < self.island_width / 2) & in_split_region
        return on_island
```

**Benefits**:
- Eliminate GPU-CPU transfers
- 20-30% speedup expected
- Cleaner code

**Risks**:
- Complex GPU logic
- Need to verify correctness matches CPU version

**Testing**:
- Unit tests comparing GPU vs CPU river checks
- Visual tests (render river, verify alignment)
- Performance benchmarks

---

### 8. Add Trait System

**Priority**: Important
**Effort**: Medium (1 week)
**Risk**: Low (additive feature)

**Solution**: Generic trait definition and evolution

```python
# traits.py
@dataclass
class Trait:
    name: str
    initial_value: float
    initial_std: float
    mutation_std: float
    min_value: float
    max_value: float

    def sample_initial(self) -> float:
        value = np.random.normal(self.initial_value, self.initial_std)
        return np.clip(value, self.min_value, self.max_value)

    def mutate(self, parent_value: float, mutation_rate: float) -> float:
        value = parent_value + np.random.randn() * self.mutation_std * mutation_rate
        return np.clip(value, self.min_value, self.max_value)

# agent.py
class Agent:
    def __init__(self, trait_definitions: Dict[str, Trait], trait_values: Dict[str, float] = None):
        self.trait_definitions = trait_definitions
        if trait_values is None:
            self.traits = {name: trait.sample_initial()
                          for name, trait in trait_definitions.items()}
        else:
            self.traits = trait_values

    def reproduce(self, mutation_rate):
        child_traits = {
            name: self.trait_definitions[name].mutate(value, mutation_rate)
            for name, value in self.traits.items()
        }
        return self.__class__(trait_definitions=self.trait_definitions,
                              trait_values=child_traits, ...)
```

**Usage**:
```python
prey_traits = {
    'speed': Trait('speed', initial_value=3.0, initial_std=0.2,
                   mutation_std=0.2, min_value=0.5, max_value=10.0),
    'swim_speed': Trait('swim_speed', initial_value=2.0, initial_std=0.2,
                        mutation_std=0.3, min_value=0.1, max_value=5.0),
    'vision_range': Trait('vision_range', initial_value=100.0, initial_std=10.0,
                          mutation_std=15.0, min_value=10.0, max_value=500.0),
}

prey = Prey(trait_definitions=prey_traits, ...)

# Use traits
max_speed = prey.traits['speed']
vision = prey.traits['vision_range']
```

**Benefits**:
- Easy to add new traits
- Consistent mutation/inheritance
- Automatic tracking in stats

---

## Nice-to-Have Improvements

### 9. Add Type Hints
**Effort**: Small, **Risk**: None, **Value**: IDE support + documentation

### 10. Break Up Large Functions
**Effort**: Small, **Risk**: Low, **Value**: Readability

### 11. Add Config Validation
**Effort**: Small, **Risk**: None, **Value**: Better error messages

### 12. Improve Error Handling
**Effort**: Small, **Risk**: None, **Value**: Easier debugging

---

## Implementation Order

**Phase 1: Foundation** (2-3 weeks)
1. Fix GPU neuroevolution (#1) - 1-2 weeks
2. Refactor config system (#4) - 1 week
3. Extract shared utilities (#6) - 2 days
4. Unify island mechanics (#5) - 2 days

**Phase 2: Core Architecture** (2-3 weeks)
5. N-species architecture (#2) - 2 weeks
6. Dynamic observations (#3) - 1 week

**Phase 3: Optimization** (1 week)
7. Port river to GPU (#7) - 4 days
8. Add trait system (#8) - 1 week

**Phase 4: Quality** (1 week)
9-12. Type hints, refactoring, validation, error handling

**Total**: 6-8 weeks for complete refactor

---

## Risk Mitigation Strategy

1. **Incremental Migration**: Keep old code working while adding new system
2. **Feature Flags**: Use config to toggle new vs old behavior
3. **Extensive Testing**: Unit tests + integration tests at each phase
4. **Performance Benchmarks**: Ensure no regression
5. **Documentation**: Update docs as code changes
6. **Git Branches**: Separate branch per major refactor

---

## Success Metrics

After refactoring, should be able to:
- ✅ Add 3rd species in <1 day
- ✅ Add new trait in <1 hour
- ✅ Add new sensor in <2 hours
- ✅ Run batch experiments with config variations
- ✅ GPU and CPU produce similar evolutionary dynamics
- ✅ No performance regression (ideally 20-30% faster with GPU river)

---

## Quick Wins (Can Do Now)

Before full refactor, these can be done independently:
- #6: Extract utilities (2 days, no risk)
- #5: Unify island mechanics (2 days, low risk)
- #11: Add config validation (1 day, no risk)
- #9: Add type hints to new code going forward (ongoing, no risk)

Start with these to improve code quality immediately while planning larger refactors.
