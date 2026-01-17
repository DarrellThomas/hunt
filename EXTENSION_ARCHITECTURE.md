# HUNT Extension Architecture

This document proposes concrete patterns and interfaces for the four main extension goals after refactoring:
1. Adding new species (arbitrary N)
2. Adding new agent features/traits
3. Adding new environmental features
4. Running batch experiments

All examples assume the refactoring from REFACTOR_PLAN.md has been completed.

## 1. Adding a New Species

### Goal
Add a third species "Scavenger" that eats dead prey corpses.

### Step-by-Step Example

**Step 1: Define the Agent Class**
```python
# scavenger.py
from agent import Agent
from sensors import NearestAgentsSensor, NearestCorpseSensor

class Scavenger(Agent):
    """Scavengers eat dead prey corpses instead of hunting live prey."""

    def __init__(self, x, y, world_width, world_height, brain=None,
                 trait_values=None, trait_definitions=None):
        # Define sensors: see prey corpses and other scavengers
        sensors = [
            NearestCorpseSensor(count=5),  # See dead prey
            NearestAgentsSensor('scavenger', count=3),  # See other scavengers (avoid crowding)
        ]

        super().__init__(x, y, world_width, world_height,
                        sensors=sensors,
                        brain=brain,
                        trait_values=trait_values,
                        trait_definitions=trait_definitions)

    def can_eat(self, corpse):
        """Check if close enough to eat a corpse."""
        distance = np.linalg.norm(self.pos - corpse.pos)
        return distance < self.traits['eat_radius']

    def eat(self, corpse):
        """Consume corpse and gain energy."""
        self.energy = min(self.max_energy,
                         self.energy + self.traits['energy_per_corpse'])

    def should_die(self):
        """Die from starvation or old age."""
        return self.energy <= 0 or self.age >= self.max_lifespan

    def can_reproduce(self):
        """Reproduce when well-fed."""
        return (self.energy >= self.traits['reproduction_threshold'] and
                self.time_since_reproduction >= self.traits['reproduction_cooldown'])
```

**Step 2: Define Species Configuration**
```python
# config.py or experiment script
from scavenger import Scavenger
from traits import Trait

scavenger_traits = {
    'speed': Trait('speed', initial_value=2.0, initial_std=0.1,
                   mutation_std=0.15, min_value=0.5, max_value=8.0),
    'swim_speed': Trait('swim_speed', initial_value=1.5, initial_std=0.2,
                        mutation_std=0.2, min_value=0.1, max_value=4.0),
    'eat_radius': Trait('eat_radius', initial_value=10.0, initial_std=1.0,
                        mutation_std=1.5, min_value=5.0, max_value=30.0),
    'energy_per_corpse': Trait('energy_per_corpse', initial_value=40.0, initial_std=5.0,
                                mutation_std=5.0, min_value=10.0, max_value=100.0),
    'reproduction_threshold': Trait('reproduction_threshold', initial_value=100.0, initial_std=0,
                                    mutation_std=0, min_value=100.0, max_value=100.0),
    'reproduction_cooldown': Trait('reproduction_cooldown', initial_value=200.0, initial_std=20.0,
                                    mutation_std=10.0, min_value=50.0, max_value=500.0),
}

scavenger_config = SpeciesConfig(
    name='scavenger',
    agent_class=Scavenger,
    trait_definitions=scavenger_traits,
    initial_count=30,
    color=(200, 200, 50),  # Yellow
)
```

**Step 3: Add to Simulation**
```python
# main.py or experiment script
prey_config = SpeciesConfig(...)
predator_config = SpeciesConfig(...)
scavenger_config = SpeciesConfig(...)  # From above

config = SimulationConfig(
    world=WorldConfig(
        width=1600,
        height=1200,
        species=[prey_config, predator_config, scavenger_config],  # Add scavenger!
    ),
    river=RiverConfig(...),
    mutation_rate=0.1,
)

world = World.from_config(config)
viz = Visualizer(world)
viz.run()
```

**Step 4: Handle Scavenger-Specific Logic**
```python
# world.py (in step() method)
def step(self, mutation_rate=0.1):
    # ... existing physics, observations, actions ...

    # After predator-prey collisions, create corpses
    for dead_prey in recently_killed_prey:
        self.corpses.append(Corpse(dead_prey.pos, decay_time=100))

    # Scavenger eating
    for scavenger in self.species_manager.get_species('scavenger'):
        for corpse in self.corpses:
            if scavenger.can_eat(corpse):
                scavenger.eat(corpse)
                self.corpses.remove(corpse)
                break

    # Decay corpses
    self.corpses = [c for c in self.corpses if c.age < c.decay_time]
    for corpse in self.corpses:
        corpse.age += 1

    # ... rest of simulation ...
```

**That's it!** Scavengers are now part of the ecosystem.

### What Files Changed?
- **1 new file**: `scavenger.py` (agent class)
- **1 modified file**: `main.py` or experiment script (add scavenger_config)
- **1 modified file**: `world.py` (handle corpses if needed)
- **0 modified files** for sensors/brain/visualization (automatic)

### Time Estimate
- **Before refactor**: Impossible (hardcoded architecture)
- **After refactor**: 2-4 hours

---

## 2. Adding a New Evolvable Trait

### Goal
Add "camouflage" trait to prey that reduces detection range by predators.

### Step-by-Step Example

**Step 1: Define the Trait**
```python
# In prey configuration
prey_traits = {
    # ... existing traits ...
    'camouflage': Trait(
        'camouflage',
        initial_value=1.0,  # 1.0 = normal visibility
        initial_std=0.1,
        mutation_std=0.05,
        min_value=0.3,  # Max 70% reduction
        max_value=1.0,  # Normal visibility
    ),
}

prey_config = SpeciesConfig(
    name='prey',
    trait_definitions=prey_traits,
    # ... rest ...
)
```

**Step 2: Use the Trait in Observations**
```python
# sensors.py
class NearestAgentsSensor(Sensor):
    def observe(self, agent, world):
        # ... existing code ...

        # Apply camouflage if observing prey
        if self.species_name == 'prey':
            for i, prey in enumerate(nearest_agents):
                # Prey with low camouflage are harder to see
                camouflage_factor = prey.traits.get('camouflage', 1.0)
                effective_distance = distances[i] / camouflage_factor
                # If effective distance > vision range, can't see this prey
                if effective_distance > self.vision_range:
                    observation[i*4:(i+1)*4] = [0, 0, 0, 0]  # Invisible

        return observation
```

**Step 3: Track in Statistics**
```python
# world.py
def record_stats(self):
    # Existing stats...
    self.stats['prey_count'].append(...)

    # Add camouflage tracking
    if 'prey' in self.species_manager.species:
        prey_camouflage = [p.traits['camouflage'] for p in self.species_manager.get_species('prey')]
        self.stats['prey_avg_camouflage'].append(np.mean(prey_camouflage))
        self.stats['prey_std_camouflage'].append(np.std(prey_camouflage))
```

**Step 4: Analyze Evolution**
```python
# analyze_evolution.py
# Automatically picks up new stats!
stats = np.load('stats_autosave.npz')
plt.plot(stats['timesteps'], stats['prey_avg_camouflage'])
plt.title('Evolution of Camouflage')
plt.show()
```

### What Files Changed?
- **1 modified file**: Config (add trait definition)
- **1 modified file**: `sensors.py` (use trait in observations)
- **1 modified file**: `world.py` (track in stats, optional)
- **0 brain changes** (automatically evolves)

### Time Estimate
- **Before refactor**: 4-6 hours (manual tracking, reproduction, mutation)
- **After refactor**: 30-60 minutes

---

## 3. Adding a New Environmental Feature

### Goal
Add "temperature zones" that affect agent energy expenditure.

### Step-by-Step Example

**Step 1: Define Environment Feature**
```python
# temperature.py
from environment import EnvironmentFeature

class TemperatureZones(EnvironmentFeature):
    """Hot and cold zones that affect energy usage."""

    def __init__(self, world_width, world_height, hot_zones, cold_zones):
        """
        Args:
            hot_zones: List of (x, y, radius) tuples
            cold_zones: List of (x, y, radius) tuples
        """
        self.width = world_width
        self.height = world_height
        self.hot_zones = hot_zones
        self.cold_zones = cold_zones

    def affects(self, pos: np.ndarray) -> bool:
        """Check if position is in any temperature zone."""
        return self._in_zone(pos, self.hot_zones) or self._in_zone(pos, self.cold_zones)

    def get_effect(self, pos: np.ndarray) -> dict:
        """Return temperature effect multipliers."""
        if self._in_zone(pos, self.hot_zones):
            return {
                'energy_multiplier': 1.5,  # Use 50% more energy in heat
                'speed_multiplier': 0.9,   # Move 10% slower
            }
        elif self._in_zone(pos, self.cold_zones):
            return {
                'energy_multiplier': 1.3,  # Use 30% more energy in cold
                'speed_multiplier': 0.8,   # Move 20% slower
            }
        else:
            return {
                'energy_multiplier': 1.0,
                'speed_multiplier': 1.0,
            }

    def _in_zone(self, pos, zones):
        """Check if position is in any of the zones."""
        for x, y, radius in zones:
            dist = np.linalg.norm(pos - np.array([x, y]))
            if dist < radius:
                return True
        return False

    def get_render_data(self) -> dict:
        """Data for visualization."""
        return {
            'hot_zones': self.hot_zones,
            'cold_zones': self.cold_zones,
        }
```

**Step 2: Add to World Configuration**
```python
# config.py
from temperature import TemperatureZones

# Define zones
hot_zones = [
    (400, 300, 150),  # (x, y, radius)
    (1200, 900, 200),
]

cold_zones = [
    (800, 600, 100),
    (200, 1000, 150),
]

temperature_config = EnvironmentFeatureConfig(
    feature_class=TemperatureZones,
    params={
        'hot_zones': hot_zones,
        'cold_zones': cold_zones,
    }
)

config = SimulationConfig(
    world=WorldConfig(...),
    river=RiverConfig(...),
    environment_features=[
        RiverFeatureConfig(...),  # Existing river
        temperature_config,        # New temperature zones
    ],
)
```

**Step 3: Apply in Simulation**
```python
# world.py (in step() method)
def step(self, mutation_rate=0.1):
    # ... observations, actions, physics ...

    # Apply environmental effects
    for agent in self.species_manager.get_all_agents():
        for feature in self.environment.features:
            if feature.affects(agent.pos):
                effect = feature.get_effect(agent.pos)

                # Apply speed modifier (already in physics)
                if 'speed_multiplier' in effect:
                    agent._effective_max_speed = agent.traits['speed'] * effect['speed_multiplier']

                # Apply energy modifier (for agents with energy)
                if 'energy_multiplier' in effect and hasattr(agent, 'energy'):
                    agent._energy_cost_multiplier = effect['energy_multiplier']

    # ... continue step ...
```

**Step 4: Render Temperature Zones**
```python
# visualizer.py
def draw_environment(self):
    for feature in self.world.environment.features:
        if isinstance(feature, TemperatureZones):
            render_data = feature.get_render_data()

            # Draw hot zones (red circles)
            for x, y, radius in render_data['hot_zones']:
                pygame.draw.circle(self.screen, (255, 100, 100), (int(x), int(y)), int(radius), 2)

            # Draw cold zones (blue circles)
            for x, y, radius in render_data['cold_zones']:
                pygame.draw.circle(self.screen, (100, 100, 255), (int(x), int(y)), int(radius), 2)
```

### What Files Changed?
- **1 new file**: `temperature.py` (feature implementation)
- **1 modified file**: Config (add feature)
- **1 modified file**: `world.py` (apply effects, already generic)
- **1 modified file**: `visualizer.py` (rendering, optional)

### Time Estimate
- **Before refactor**: 1-2 days (tight coupling with existing code)
- **After refactor**: 2-4 hours

---

## 4. Running Batch Experiments

### Goal
Run experiments varying predator speed to find optimal predator-prey speed ratio.

### Step-by-Step Example

**Step 1: Define Experiment Configurations**
```python
# experiment_runner.py
from config import *
from world import World
import numpy as np

# Base configuration
base_config = SimulationConfig(
    world=WorldConfig(
        width=1600,
        height=1200,
        species=[
            SpeciesConfig(
                name='prey',
                trait_definitions={
                    'speed': Trait('speed', initial_value=3.0, ...),
                    # ... other traits ...
                },
                initial_count=200,
                agent_class=Prey,
                color=(50, 255, 50),
            ),
            SpeciesConfig(
                name='predator',
                trait_definitions={
                    'speed': Trait('speed', initial_value=2.5, ...),  # Will vary
                    # ... other traits ...
                },
                initial_count=40,
                agent_class=Predator,
                color=(255, 50, 50),
            ),
        ],
    ),
    river=RiverConfig(...),
    mutation_rate=0.1,
)

# Generate experiment variants
experiments = []
for pred_speed in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
    # Clone base config
    exp_config = copy.deepcopy(base_config)

    # Modify predator speed
    exp_config.world.species[1].trait_definitions['speed'].initial_value = pred_speed

    # Add to experiment list
    experiments.append({
        'name': f'pred_speed_{pred_speed}',
        'config': exp_config,
        'pred_speed': pred_speed,
    })
```

**Step 2: Run Experiments in Parallel**
```python
# experiment_runner.py
from multiprocessing import Pool
import time

def run_experiment(exp):
    """Run a single experiment."""
    print(f"Starting experiment: {exp['name']}")
    start_time = time.time()

    # Create world
    world = World.from_config(exp['config'])

    # Run headless
    for step in range(10000):
        world.step(mutation_rate=exp['config'].mutation_rate)

        if step % 1000 == 0:
            print(f"  {exp['name']}: Step {step}")

    # Save results
    world.save_stats(f"results/{exp['name']}.npz")

    duration = time.time() - start_time
    print(f"Completed {exp['name']} in {duration:.1f}s")

    return exp['name']

# Run experiments in parallel
if __name__ == '__main__':
    with Pool(processes=4) as pool:  # 4 parallel experiments
        results = pool.map(run_experiment, experiments)

    print(f"\nAll experiments complete: {results}")
```

**Step 3: Analyze Results**
```python
# analyze_experiments.py
import numpy as np
import matplotlib.pyplot as plt
import glob

# Load all experiment results
results = {}
for filename in glob.glob('results/pred_speed_*.npz'):
    exp_name = filename.split('/')[-1].replace('.npz', '')
    pred_speed = float(exp_name.replace('pred_speed_', ''))

    stats = np.load(filename)
    results[pred_speed] = {
        'prey_survival': np.mean(stats['prey_count']),
        'pred_survival': np.mean(stats['predator_count']),
        'ecosystem_stability': np.std(stats['prey_count']),
        'pred_fitness': np.mean(stats['pred_avg_age']),
    }

# Plot results
speeds = sorted(results.keys())
prey_counts = [results[s]['prey_survival'] for s in speeds]
pred_counts = [results[s]['pred_survival'] for s in speeds]

plt.figure(figsize=(10, 6))
plt.plot(speeds, prey_counts, 'g-o', label='Avg Prey')
plt.plot(speeds, pred_counts, 'r-o', label='Avg Predators')
plt.xlabel('Predator Speed')
plt.ylabel('Average Population')
plt.title('Effect of Predator Speed on Population Dynamics')
plt.legend()
plt.grid(True)
plt.savefig('experiment_results.png')
plt.show()

# Find optimal speed ratio
for speed in speeds:
    ratio = speed / 3.0  # Prey speed is 3.0
    stability = results[speed]['ecosystem_stability']
    print(f"Predator speed {speed} (ratio {ratio:.2f}): Stability = {stability:.1f}")
```

**Step 4: Parameter Sweep (Advanced)**
```python
# sweep_parameters.py
import itertools

# Define parameter grid
param_grid = {
    'pred_speed': [2.0, 2.5, 3.0, 3.5],
    'mutation_rate': [0.05, 0.1, 0.2],
    'initial_prey': [100, 200, 300],
}

# Generate all combinations
experiments = []
for pred_speed, mut_rate, initial_prey in itertools.product(*param_grid.values()):
    config = create_config(
        pred_speed=pred_speed,
        mutation_rate=mut_rate,
        initial_prey_count=initial_prey,
    )

    experiments.append({
        'name': f'pred{pred_speed}_mut{mut_rate}_prey{initial_prey}',
        'config': config,
        'params': {
            'pred_speed': pred_speed,
            'mutation_rate': mut_rate,
            'initial_prey': initial_prey,
        },
    })

print(f"Total experiments: {len(experiments)}")  # 4 * 3 * 3 = 36 experiments

# Run in parallel
run_batch_experiments(experiments, parallel_jobs=8)
```

### What Files Changed?
- **1 new file**: `experiment_runner.py` (batch execution)
- **1 new file**: `analyze_experiments.py` (result analysis)
- **0 modified files** in core simulation (config-driven)

### Time Estimate
- **Before refactor**: Very difficult (global config blocks this)
- **After refactor**: 1-2 hours to set up, then automatic

---

## Common Patterns Summary

### Pattern 1: Extensible Species
```python
species_configs = [
    SpeciesConfig(name='prey', ...),
    SpeciesConfig(name='predator', ...),
    SpeciesConfig(name='scavenger', ...),  # Easy to add
    SpeciesConfig(name='parasite', ...),   # Easy to add
]
world = World.from_config(SimulationConfig(species=species_configs))
```

### Pattern 2: Composable Traits
```python
trait_definitions = {
    'speed': Trait(...),
    'vision': Trait(...),
    'camouflage': Trait(...),  # Add new traits easily
    'toxicity': Trait(...),    # Add more traits
}
agent = Agent(trait_definitions=trait_definitions, ...)
```

### Pattern 3: Composable Sensors
```python
sensors = [
    NearestAgentsSensor('predator', count=5),
    NearestAgentsSensor('prey', count=3),
    DistanceToIslandSensor(),      # Add new sensors
    PredatorScent Sensor(),         # Add more sensors
]
agent = Agent(sensors=sensors, ...)
```

### Pattern 4: Composable Environment
```python
environment = Environment(features=[
    River(width, height, ...),
    TemperatureZones(hot_zones, cold_zones),
    Obstacles(obstacle_list),       # Add new features
    Resources(food_patches),        # Add more features
])
world = World(environment=environment, ...)
```

### Pattern 5: Config-Driven Experiments
```python
for config_variant in config_variants:
    world = World.from_config(config_variant)
    world.run(steps=10000)
    world.save_stats(f'{config_variant.name}.npz')
```

---

## Testing Strategy

### Unit Tests
```python
# test_species.py
def test_add_scavenger():
    config = SimulationConfig(species=[prey_config, pred_config, scav_config])
    world = World.from_config(config)
    assert 'scavenger' in world.species_manager.species
    assert len(world.species_manager.get_species('scavenger')) == 30

# test_traits.py
def test_trait_inheritance():
    parent = Prey(trait_values={'speed': 5.0}, ...)
    child = parent.reproduce(mutation_rate=0.1)
    assert abs(child.traits['speed'] - parent.traits['speed']) < 1.0  # Mutation is small

# test_environment.py
def test_temperature_effect():
    temp = TemperatureZones(hot_zones=[(100, 100, 50)], cold_zones=[])
    effect = temp.get_effect(np.array([100, 100]))
    assert effect['energy_multiplier'] == 1.5
```

### Integration Tests
```python
# test_integration.py
def test_three_species_stable():
    """Test that 3-species ecosystem remains stable for 1000 steps."""
    config = SimulationConfig(species=[prey, predator, scavenger])
    world = World.from_config(config)

    for _ in range(1000):
        world.step()

    # All species should still exist
    assert len(world.species_manager.get_species('prey')) > 10
    assert len(world.species_manager.get_species('predator')) > 3
    assert len(world.species_manager.get_species('scavenger')) > 5
```

---

## Performance Considerations

### Scaling with N Species
- Observation computation: O(N × M) where N = agents, M = species
- Collision detection: O(N²) worst case, but spatial hashing can reduce
- Neural network forward pass: O(N) per species

**Recommendations**:
- Use spatial grid for collision detection when N > 1000
- Sample observations (like GPU version does) when many species
- Profile each new species to identify bottlenecks

### Memory Usage
- CPU version: Scales linearly with agent count
- GPU version: Memory for all agent tensors must fit in VRAM
- With N species: Roughly N × (base memory per species)

**Estimate**: 10K agents across 3 species ≈ 2GB VRAM

---

## Documentation Template

When adding new features, document using this template:

```markdown
# New Feature: [Name]

## Purpose
Brief description of what this adds and why.

## Usage Example
\`\`\`python
# Code showing how to use
\`\`\`

## Configuration
List all new config parameters.

## Performance Impact
Expected CPU/GPU/memory impact.

## Testing
How to verify it works correctly.

## Integration
What files changed, what interfaces affected.
```

---

## Conclusion

After refactoring, HUNT becomes a platform for experimenting with:
- **Multi-species ecosystems** (add species in hours)
- **Evolutionary dynamics** (add traits in minutes)
- **Environmental pressures** (add features in hours)
- **Scientific experiments** (run parameter sweeps automatically)

The key is **composition over hardcoding**: agents, traits, sensors, and environments are all composable building blocks that can be mixed and matched without modifying core simulation code.
