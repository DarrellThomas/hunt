# Adding a New Species to HUNT

This guide shows you how to add a third (or fourth, or fifth...) species to the HUNT simulation using the N-species architecture.

## Overview

Thanks to the architecture refactoring (Phase 2), adding new species is now a configuration task rather than a code modification task. You define species in JSON config files and the system handles the rest.

## Quick Example: Adding an Apex Predator

Here's how to add an "apex predator" species that hunts regular predators:

```python
from config_new import (
    SimulationConfig, SpeciesConfig, PhysicsConfig, LifecycleConfig,
    EnergyConfig, ObservationConfig, InteractionConfig, WorldConfig, RiverConfig
)

# Define the apex predator species
apex_config = SpeciesConfig(
    name="apex",
    physics=PhysicsConfig(
        max_speed=3.0,
        max_acceleration=0.5,
        swim_speed=2.5
    ),
    lifecycle=LifecycleConfig(
        max_lifespan=3000,
        lifespan_variance=300,
        reproduction_age=1000,
        reproduction_variance=100
    ),
    observation=ObservationConfig(
        observe_species={
            'predator': 5,  # Observe 5 nearest predators (prey for apex)
            'apex': 2        # Observe 2 nearest other apex predators
        },
        sense_hunger=True
    ),
    energy=EnergyConfig(
        max_energy=200.0,
        initial_energy=200.0,
        energy_cost_per_step=0.15,
        energy_gain_per_kill=80.0,
        reproduction_threshold=160.0,
        reproduction_cost=60.0
    ),
    initial_count=10,
    color=(180, 0, 180),  # Purple
    has_energy_system=True
)

# Add interaction: apex eats predators
apex_eats_predators = InteractionConfig(
    predator_species="apex",
    prey_species="predator",
    catch_radius=10.0
)

# Create simulation config with all three species
config = SimulationConfig(
    world=WorldConfig(width=800, height=600),
    river=RiverConfig(),
    species=[
        SimulationConfig.default_two_species().get_species('prey'),
        SimulationConfig.default_two_species().get_species('predator'),
        apex_config
    ],
    interactions=[
        InteractionConfig('predator', 'prey', 8.0),  # Predators eat prey
        apex_eats_predators                          # Apex eat predators
    ]
)

# Save to file
config.to_json_file('configs/three_species.json')
```

## Step-by-Step Guide

### 1. Define Physics Parameters

```python
physics = PhysicsConfig(
    max_speed=3.0,           # Maximum movement speed
    max_acceleration=0.5,     # How quickly it can change direction
    swim_speed=2.0            # Resistance to river current
)
```

### 2. Define Lifecycle Parameters

```python
lifecycle = LifecycleConfig(
    max_lifespan=2000,           # How many steps before dying of old age
    lifespan_variance=200,        # Natural variation in lifespan
    reproduction_age=500,         # How often it reproduces (for non-energy species)
    reproduction_variance=50      # Variation in reproduction timing
)
```

### 3. Define Observation (What it Sees)

```python
observation = ObservationConfig(
    observe_species={
        'prey': 3,        # Observe 3 nearest prey
        'predator': 2,    # Observe 2 nearest predators
        'my_species': 1   # Observe 1 nearest same-species agent
    },
    sense_hunger=True,    # Can sense own energy level (for energy-based species)
    sense_island=False    # Can sense if on river island
)
```

The observation config automatically calculates the correct neural network input size based on what sensors you enable.

### 4. Define Energy System (Optional)

If your species needs to hunt for energy:

```python
energy = EnergyConfig(
    max_energy=150.0,               # Maximum energy capacity
    initial_energy=150.0,            # Starting energy
    energy_cost_per_step=0.1,       # Energy lost each step (hunger)
    energy_gain_per_kill=60.0,      # Energy gained from eating
    reproduction_threshold=120.0,    # Must have this much to reproduce
    reproduction_cost=50.0          # Energy cost of reproduction
)

species_config = SpeciesConfig(
    # ... other params ...
    energy=energy,
    has_energy_system=True
)
```

If your species doesn't hunt (like herbivores):

```python
species_config = SpeciesConfig(
    # ... other params ...
    has_energy_system=False  # Age-based reproduction instead
)
```

### 5. Create the Complete Species Config

```python
my_species = SpeciesConfig(
    name="herbivore",
    physics=physics,
    lifecycle=lifecycle,
    observation=observation,
    energy=energy,  # Or None if has_energy_system=False
    initial_count=100,  # Starting population
    color=(100, 200, 100),  # RGB color for visualization
    has_energy_system=False  # True if needs energy to survive
)
```

### 6. Define Interactions

Specify which species can eat which:

```python
interactions = [
    InteractionConfig(
        predator_species="predator",
        prey_species="herbivore",
        catch_radius=8.0  # How close they must be to catch
    ),
    InteractionConfig(
        predator_species="apex",
        prey_species="predator",
        catch_radius=10.0
    )
]
```

### 7. Create Full Simulation Config

```python
config = SimulationConfig(
    world=WorldConfig(width=800, height=600),
    river=RiverConfig(),
    species=[species1, species2, species3],
    interactions=interactions
)

# Save to file
config.to_json_file('my_simulation.json')

# Or load from file
config = SimulationConfig.from_json_file('my_simulation.json')
```

## Common Patterns

### Herbivore (Eats Plants, Eaten by Predators)

```python
herbivore = SpeciesConfig(
    name="herbivore",
    physics=PhysicsConfig(max_speed=2.5, max_acceleration=0.4, swim_speed=1.5),
    lifecycle=LifecycleConfig(max_lifespan=1500, reproduction_age=400),
    observation=ObservationConfig(observe_species={'predator': 5, 'herbivore': 3}),
    has_energy_system=False,  # Doesn't need to hunt
    initial_count=200,
    color=(100, 200, 100)
)
```

### Scavenger (Eats Dead Bodies)

```python
# Note: Scavenger behavior requires implementing dead body tracking
# in the simulation. This is a future extension.

scavenger = SpeciesConfig(
    name="scavenger",
    # ... configure to seek out recently dead agents ...
)
```

### Pack Hunter (Hunts in Groups)

```python
# Pack hunting requires custom behavior implementation
# For now, create a species that observes many same-species agents

pack_hunter = SpeciesConfig(
    name="wolf",
    observation=ObservationConfig(
        observe_species={
            'prey': 3,
            'wolf': 5  # Observe many packmates
        }
    ),
    # Brain may learn cooperative strategies through evolution
)
```

## Neural Network Considerations

Each species gets its own neural network trained through neuroevolution. The network input size is automatically calculated from the observation config:

- Each observed agent contributes 4 features: (dx, dy, vx, vy)
- Hunger sensor adds 1 feature
- Island sensor adds 1 feature
- Wall proximity adds 4 features (for bounded mode)

**Example calculation:**
```python
observation = ObservationConfig(
    observe_species={'prey': 3, 'predator': 2},  # 5 agents * 4 = 20
    sense_hunger=True,                            # +1 = 21
    sense_island=False                            # +0 = 21 total
)
# Neural network input size: 21
```

## Validation

The config system automatically validates:
- All species names are unique
- Interaction species exist
- Energy values are positive
- Neural network input sizes are computable
- No circular dependencies

If validation fails, you'll get a clear error message:

```
ValueError: Interaction references unknown species 'dinosaur'
```

## Testing Your New Species

After creating a config:

```python
# 1. Validate the config
config = SimulationConfig.from_json_file('my_config.json')
print("Config valid!")

# 2. Check observation dimensions
for species in config.species:
    from sensors import SensorSuite
    suite = SensorSuite.from_config(species.observation)
    print(f"{species.name} observes {suite.total_dimension} values")

# 3. Run a short simulation
# (Implementation depends on simulation runner - see RUNNING_EXPERIMENTS.md)
```

## FAQ

**Q: Can I have a species that doesn't move?**
A: Yes, set `max_speed=0` and `max_acceleration=0` (like plants).

**Q: Can a species interact with itself?**
A: Not currently (no cannibalism), but you could add `InteractionConfig('species', 'species')`.

**Q: How many species can I have?**
A: No hard limit! Performance scales with population count, not species count.

**Q: Can species have different neural network architectures?**
A: Currently all use 32→32→2 architecture. Different sizes requires code changes.

**Q: Can I define custom sensors?**
A: Yes! Create a new `Sensor` subclass in `sensors.py`. See existing sensors for examples.

## Next Steps

- See `BOUNDARY_MODES.md` for toroidal vs bounded world configuration
- See `RUNNING_EXPERIMENTS.md` for batch experiment setup
- See `tests/test_n_species.py` for programmatic examples
- See `config_new.py` for all available configuration options

## Examples

See `configs/` directory for example configs:
- `two_species.json` - Classic predator-prey
- `three_species.json` - Herbivore, carnivore, apex predator
- `bounded_world.json` - Walled environment example
