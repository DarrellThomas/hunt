# HUNT Platform: Architecture Refactoring

**Prerequisites**: GPU neuroevolution fix must be complete (Prompt 1)
**Budget**: ~$50
**Scope**: Refactor architecture for N-species support and extensibility

## Overview

This prompt covers the remaining refactoring work after the GPU neuroevolution bug is fixed. The goal is to transform HUNT from a hardcoded 2-species simulation into an extensible platform.

## Implementation Order

1. **Phase 1: Foundation** - Utilities, config system, boundary modes
2. **Phase 2: Core Architecture** - N-species, dynamic sensors
3. **Phase 3: Optimization** - GPU river, trait system
4. **Phase 4: Quality** - Tests, documentation

---

## Phase 1: Foundation

### 1.1 Extract Shared Utilities

**Create**: `utils.py`
**Effort**: Small (1-2 hours)

Extract duplicated distance and spawn calculations:

```python
# utils.py
"""Shared utilities for HUNT simulation."""

import numpy as np
import torch
from typing import Tuple, Union, Literal

def toroidal_distance_numpy(
    pos1: np.ndarray, 
    pos2: np.ndarray,
    world_width: float, 
    world_height: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute toroidal (wrap-around) distances using NumPy.
    
    Args:
        pos1: Reference position (2,) or batch (N, 2)
        pos2: Target positions (M, 2)
        world_width: World width for wrapping
        world_height: World height for wrapping
    
    Returns:
        distances: (M,) or (N, M) array of distances
        vectors: Direction vectors from pos1 to pos2
    """
    if pos1.ndim == 1:
        pos1 = pos1.reshape(1, 2)
        squeeze = True
    else:
        squeeze = False
    
    # Compute raw differences
    dx = pos2[:, 0] - pos1[:, 0:1]  # (N, M) or (1, M)
    dy = pos2[:, 1] - pos1[:, 1:2]
    
    # Apply toroidal wrapping
    dx = np.where(np.abs(dx) > world_width / 2,
                  dx - np.sign(dx) * world_width, dx)
    dy = np.where(np.abs(dy) > world_height / 2,
                  dy - np.sign(dy) * world_height, dy)
    
    distances = np.sqrt(dx**2 + dy**2)
    vectors = np.stack([dx, dy], axis=-1)
    
    if squeeze:
        distances = distances.squeeze(0)
        vectors = vectors.squeeze(0)
    
    return distances, vectors


def bounded_distance_numpy(
    pos1: np.ndarray,
    pos2: np.ndarray,
    world_width: float,
    world_height: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute distances in bounded (walled) world using NumPy.
    
    Same interface as toroidal_distance_numpy but without wrapping.
    """
    if pos1.ndim == 1:
        pos1 = pos1.reshape(1, 2)
        squeeze = True
    else:
        squeeze = False
    
    dx = pos2[:, 0] - pos1[:, 0:1]
    dy = pos2[:, 1] - pos1[:, 1:2]
    
    distances = np.sqrt(dx**2 + dy**2)
    vectors = np.stack([dx, dy], axis=-1)
    
    if squeeze:
        distances = distances.squeeze(0)
        vectors = vectors.squeeze(0)
    
    return distances, vectors


def toroidal_distance_torch(
    pos1: torch.Tensor, 
    pos2: torch.Tensor,
    world_width: float, 
    world_height: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute toroidal distances using PyTorch (GPU-compatible).
    
    Args:
        pos1: Reference positions (N, 2)
        pos2: Target positions (M, 2)
        
    Returns:
        distances: (N, M) tensor of distances
        vectors: (N, M, 2) direction vectors
    """
    # Broadcast to (N, M, 2)
    diff = pos1.unsqueeze(1) - pos2.unsqueeze(0)
    
    # Apply toroidal wrapping
    diff[..., 0] = torch.where(
        torch.abs(diff[..., 0]) > world_width / 2,
        diff[..., 0] - torch.sign(diff[..., 0]) * world_width,
        diff[..., 0]
    )
    diff[..., 1] = torch.where(
        torch.abs(diff[..., 1]) > world_height / 2,
        diff[..., 1] - torch.sign(diff[..., 1]) * world_height,
        diff[..., 1]
    )
    
    distances = torch.norm(diff, dim=2)
    return distances, diff


def bounded_distance_torch(
    pos1: torch.Tensor,
    pos2: torch.Tensor,
    world_width: float,
    world_height: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute distances in bounded world using PyTorch."""
    diff = pos1.unsqueeze(1) - pos2.unsqueeze(0)
    distances = torch.norm(diff, dim=2)
    return distances, diff


def spawn_offset(
    count: int,
    min_distance: float = 20.0,
    max_distance: float = 150.0,
    framework: Literal['numpy', 'torch'] = 'numpy',
    device: str = 'cpu'
) -> Union[np.ndarray, torch.Tensor]:
    """Generate random spawn offsets for reproduction.
    
    Args:
        count: Number of offsets to generate
        min_distance: Minimum spawn distance from parent
        max_distance: Maximum spawn distance from parent
        framework: 'numpy' or 'torch'
        device: PyTorch device (only used if framework='torch')
    
    Returns:
        Array/Tensor of shape (count, 2) with x,y offsets
    """
    if framework == 'numpy':
        distance = np.random.uniform(min_distance, max_distance, size=count)
        angle = np.random.uniform(0, 2 * np.pi, size=count)
        offset_x = distance * np.cos(angle)
        offset_y = distance * np.sin(angle)
        return np.column_stack([offset_x, offset_y])
    else:
        distance = torch.rand(count, device=device) * (max_distance - min_distance) + min_distance
        angle = torch.rand(count, device=device) * 2 * torch.pi
        offset_x = distance * torch.cos(angle)
        offset_y = distance * torch.sin(angle)
        return torch.stack([offset_x, offset_y], dim=1)


def wrap_position_numpy(
    pos: np.ndarray,
    world_width: float,
    world_height: float
) -> np.ndarray:
    """Wrap positions to stay within toroidal world bounds."""
    pos = pos.copy()
    pos[..., 0] = pos[..., 0] % world_width
    pos[..., 1] = pos[..., 1] % world_height
    return pos


def clamp_position_numpy(
    pos: np.ndarray,
    world_width: float,
    world_height: float,
    margin: float = 0.0
) -> np.ndarray:
    """Clamp positions to stay within bounded world."""
    pos = pos.copy()
    pos[..., 0] = np.clip(pos[..., 0], margin, world_width - margin)
    pos[..., 1] = np.clip(pos[..., 1], margin, world_height - margin)
    return pos


def wrap_position_torch(
    pos: torch.Tensor,
    world_width: float,
    world_height: float
) -> torch.Tensor:
    """Wrap positions for toroidal world (GPU)."""
    pos = pos.clone()
    pos[..., 0] = pos[..., 0] % world_width
    pos[..., 1] = pos[..., 1] % world_height
    return pos


def clamp_position_torch(
    pos: torch.Tensor,
    world_width: float,
    world_height: float,
    margin: float = 0.0
) -> torch.Tensor:
    """Clamp positions for bounded world (GPU)."""
    pos = pos.clone()
    pos[..., 0] = pos[..., 0].clamp(margin, world_width - margin)
    pos[..., 1] = pos[..., 1].clamp(margin, world_height - margin)
    return pos
```

**Migration**:
1. Create `utils.py` with above functions
2. Update `agent.py`: replace `vectorized_distances()` with import from utils
3. Update `world.py`: use utils for distance calculations
4. Update `simulation_gpu.py`: use torch versions from utils
5. Write unit tests in `tests/test_utils.py`

---

### 1.2 Refactor Configuration System

**Create**: `config_new.py`
**Effort**: Medium (half day)

Create dataclass-based configuration with **boundary mode support**:

```python
# config_new.py
"""Configuration system for HUNT simulation."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Literal
from enum import Enum
import json

class BoundaryMode(Enum):
    """World boundary behavior."""
    TOROIDAL = "toroidal"  # Wrap-around (current default)
    BOUNDED = "bounded"    # Walls at edges


@dataclass
class PhysicsConfig:
    """Physics parameters for a species."""
    max_speed: float
    max_acceleration: float
    swim_speed: float = 2.0
    
    def validate(self):
        assert self.max_speed > 0, "max_speed must be positive"
        assert self.max_acceleration > 0, "max_acceleration must be positive"
        assert self.swim_speed >= 0, "swim_speed must be non-negative"


@dataclass  
class LifecycleConfig:
    """Lifecycle parameters for age-based reproduction (prey-style)."""
    max_lifespan: int
    lifespan_variance: int
    reproduction_age: int
    reproduction_variance: int
    
    def validate(self):
        assert self.max_lifespan > 0, "max_lifespan must be positive"
        assert self.reproduction_age > 0, "reproduction_age must be positive"
        assert self.reproduction_age < self.max_lifespan, "reproduction_age must be less than max_lifespan"


@dataclass
class EnergyConfig:
    """Energy parameters for predator-style species."""
    max_energy: float
    initial_energy: float
    energy_cost_per_step: float
    energy_gain_per_kill: float
    reproduction_threshold: float
    reproduction_cost: float
    reproduction_cooldown: int
    cooldown_variance: int
    
    def validate(self):
        assert self.max_energy > 0, "max_energy must be positive"
        assert self.initial_energy > 0, "initial_energy must be positive"
        assert self.energy_cost_per_step >= 0, "energy_cost must be non-negative"
        assert self.reproduction_threshold <= self.max_energy, "reproduction_threshold must be <= max_energy"


@dataclass
class ObservationConfig:
    """What a species can perceive."""
    # species_name -> count of nearest agents to observe
    observe_species: Dict[str, int] = field(default_factory=dict)
    # Additional sensors
    sense_hunger: bool = False
    sense_island: bool = False
    
    @property
    def base_dimension(self) -> int:
        """Calculate observation vector size (before custom sensors)."""
        # 4 values per observed agent: dx, dy, vx, vy (normalized)
        dim = sum(count * 4 for count in self.observe_species.values())
        if self.sense_hunger:
            dim += 1
        if self.sense_island:
            dim += 1
        return dim


@dataclass
class SpeciesConfig:
    """Complete configuration for a species."""
    name: str
    physics: PhysicsConfig
    lifecycle: LifecycleConfig
    observation: ObservationConfig
    initial_count: int
    color: Tuple[int, int, int]
    energy: Optional[EnergyConfig] = None  # None for prey-style, set for predator-style
    
    @property
    def has_energy_system(self) -> bool:
        return self.energy is not None
    
    @property
    def input_size(self) -> int:
        """Neural network input size for this species."""
        return self.observation.base_dimension
    
    def validate(self):
        self.physics.validate()
        self.lifecycle.validate()
        if self.energy:
            self.energy.validate()
        assert self.initial_count >= 0, "initial_count must be non-negative"
        assert len(self.color) == 3, "color must be RGB tuple"


@dataclass
class RiverConfig:
    """River and island configuration."""
    enabled: bool = True
    width: float = 80.0
    flow_speed: float = 0.5
    curviness: float = 0.3
    num_path_points: int = 50
    
    # Island (split river) settings
    split: bool = True
    split_start: float = 0.3  # Where split begins (0-1 along river)
    split_end: float = 0.7    # Where split ends
    island_width: float = 100.0
    
    # Island behavior modifiers
    island_speed_multiplier: float = 0.5
    island_hunger_multiplier: float = 0.5
    island_reproduction_multiplier: float = 2.0


@dataclass
class ExtinctionPreventionConfig:
    """Settings for preventing species extinction."""
    enabled: bool = True
    emergency_respawn_count: int = 5  # Respawn this many if population hits 0
    minimum_population: int = 10
    scale_with_world_size: bool = True  # If true: min = max(minimum_population, area/24000)


@dataclass
class WorldConfig:
    """World parameters."""
    width: int = 1600
    height: int = 1200
    boundary_mode: BoundaryMode = BoundaryMode.TOROIDAL
    catch_radius: float = 8.0  # Default predation radius


@dataclass
class InteractionConfig:
    """Defines predator-prey relationship between two species."""
    predator_species: str
    prey_species: str
    catch_radius: float = 8.0
    energy_gain: float = 50.0  # Energy predator gains from catching prey


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    world: WorldConfig
    river: RiverConfig
    species: List[SpeciesConfig]
    interactions: List[InteractionConfig] = field(default_factory=list)
    extinction_prevention: ExtinctionPreventionConfig = field(default_factory=ExtinctionPreventionConfig)
    mutation_rate: float = 0.1
    
    def validate(self):
        """Validate entire configuration."""
        for sp in self.species:
            sp.validate()
        
        # Validate interactions reference valid species
        species_names = {sp.name for sp in self.species}
        for interaction in self.interactions:
            assert interaction.predator_species in species_names, \
                f"Unknown predator species: {interaction.predator_species}"
            assert interaction.prey_species in species_names, \
                f"Unknown prey species: {interaction.prey_species}"
    
    def get_species(self, name: str) -> Optional[SpeciesConfig]:
        """Get species config by name."""
        for sp in self.species:
            if sp.name == name:
                return sp
        return None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        import dataclasses
        def convert(obj):
            if dataclasses.is_dataclass(obj):
                return {k: convert(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, (list, tuple)):
                return [convert(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        return convert(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'SimulationConfig':
        """Create from dictionary."""
        # Convert boundary_mode string back to enum
        if 'world' in data and 'boundary_mode' in data['world']:
            data['world']['boundary_mode'] = BoundaryMode(data['world']['boundary_mode'])
        
        # Reconstruct nested dataclasses
        world = WorldConfig(**data['world'])
        river = RiverConfig(**data['river'])
        
        species = []
        for sp_data in data['species']:
            physics = PhysicsConfig(**sp_data['physics'])
            lifecycle = LifecycleConfig(**sp_data['lifecycle'])
            observation = ObservationConfig(**sp_data['observation'])
            energy = EnergyConfig(**sp_data['energy']) if sp_data.get('energy') else None
            species.append(SpeciesConfig(
                name=sp_data['name'],
                physics=physics,
                lifecycle=lifecycle,
                observation=observation,
                initial_count=sp_data['initial_count'],
                color=tuple(sp_data['color']),
                energy=energy
            ))
        
        interactions = [InteractionConfig(**i) for i in data.get('interactions', [])]
        extinction = ExtinctionPreventionConfig(**data.get('extinction_prevention', {}))
        
        return cls(
            world=world,
            river=river,
            species=species,
            interactions=interactions,
            extinction_prevention=extinction,
            mutation_rate=data.get('mutation_rate', 0.1)
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'SimulationConfig':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    @classmethod
    def default_two_species(cls) -> 'SimulationConfig':
        """Create default config matching original HUNT behavior."""
        prey_config = SpeciesConfig(
            name='prey',
            physics=PhysicsConfig(max_speed=3.0, max_acceleration=0.5, swim_speed=2.0),
            lifecycle=LifecycleConfig(
                max_lifespan=500,
                lifespan_variance=100,
                reproduction_age=100,
                reproduction_variance=20
            ),
            observation=ObservationConfig(
                observe_species={'predator': 5, 'prey': 3},
                sense_hunger=False,
                sense_island=False
            ),
            initial_count=200,
            color=(50, 255, 50),  # Green
            energy=None
        )
        
        predator_config = SpeciesConfig(
            name='predator',
            physics=PhysicsConfig(max_speed=2.5, max_acceleration=0.4, swim_speed=1.8),
            lifecycle=LifecycleConfig(
                max_lifespan=800,
                lifespan_variance=150,
                reproduction_age=150,  # Not used directly for predators
                reproduction_variance=30
            ),
            observation=ObservationConfig(
                observe_species={'prey': 5},
                sense_hunger=True,
                sense_island=False
            ),
            initial_count=40,
            color=(255, 50, 50),  # Red
            energy=EnergyConfig(
                max_energy=150.0,
                initial_energy=100.0,
                energy_cost_per_step=0.3,
                energy_gain_per_kill=50.0,
                reproduction_threshold=120.0,
                reproduction_cost=60.0,
                reproduction_cooldown=100,
                cooldown_variance=20
            )
        )
        
        return cls(
            world=WorldConfig(
                width=1600,
                height=1200,
                boundary_mode=BoundaryMode.TOROIDAL,
                catch_radius=8.0
            ),
            river=RiverConfig(
                enabled=True,
                width=80.0,
                flow_speed=0.5,
                curviness=0.3,
                split=True
            ),
            species=[prey_config, predator_config],
            interactions=[
                InteractionConfig(
                    predator_species='predator',
                    prey_species='prey',
                    catch_radius=8.0,
                    energy_gain=50.0
                )
            ],
            extinction_prevention=ExtinctionPreventionConfig(
                enabled=True,
                emergency_respawn_count=5,
                minimum_population=10,
                scale_with_world_size=True
            ),
            mutation_rate=0.1
        )
    
    @classmethod
    def default_bounded(cls) -> 'SimulationConfig':
        """Create config with bounded (walled) world instead of toroidal."""
        config = cls.default_two_species()
        config.world.boundary_mode = BoundaryMode.BOUNDED
        return config
```

**Migration**:
1. Create `config_new.py`
2. Add `from_config()` class methods to `World` and `GPUEcosystem`
3. Update `world.py` to support `BoundaryMode`:
   ```python
   def update_physics(self):
       # ... velocity update ...
       self.pos += self.vel * dt
       
       # Apply boundary based on mode
       if self.config.world.boundary_mode == BoundaryMode.TOROIDAL:
           self.pos = wrap_position_numpy(self.pos, self.width, self.height)
       else:
           self.pos = clamp_position_numpy(self.pos, self.width, self.height)
           # Optionally: bounce off walls
           # if self.pos[0] <= 0 or self.pos[0] >= self.width:
           #     self.vel[0] *= -1
   ```
4. Update `simulation_gpu.py` similarly for GPU version
5. Update distance calculations to use appropriate function based on boundary mode
6. Test both modes work correctly

---

### 1.3 Unify Extinction Prevention

**Modify**: `world.py`, `simulation_gpu.py`
**Effort**: Small (2 hours)

Use the `ExtinctionPreventionConfig` from above to unify both versions:

```python
# In World.step()
def _handle_extinction_prevention(self):
    config = self.config.extinction_prevention
    if not config.enabled:
        return
    
    min_pop = config.minimum_population
    if config.scale_with_world_size:
        min_pop = max(min_pop, int(self.width * self.height / 24000))
    
    for species_name, agents in self.species_manager.populations.items():
        if len(agents) < 1:
            # Emergency respawn
            self._respawn_random(species_name, config.emergency_respawn_count)
        elif len(agents) < min_pop:
            # Boost population
            deficit = min_pop - len(agents)
            self._respawn_random(species_name, min(deficit, config.emergency_respawn_count))
```

---

## Phase 2: Core Architecture

### 2.1 Implement N-Species Architecture

**Create**: `species.py`
**Modify**: `world.py`, `simulation_gpu.py`, visualizers
**Effort**: Large (1-2 days)

```python
# species.py
"""Species management for HUNT simulation."""

from dataclasses import dataclass
from typing import Dict, List, Type, Optional
from enum import Enum, auto
import numpy as np

class AgentRole(Enum):
    """Predefined agent roles (for common patterns)."""
    PREY = auto()      # Hunted, age-based reproduction
    PREDATOR = auto()  # Hunter, energy-based reproduction
    SCAVENGER = auto() # Eats dead agents (future)
    PRODUCER = auto()  # Creates resources (future, e.g., plants)


@dataclass
class InteractionResult:
    """Result of an interaction between agents."""
    prey_killed: List[int]  # Indices of killed prey
    predator_fed: List[int]  # Indices of predators that ate
    energy_gained: List[float]  # Energy gained by each fed predator


class SpeciesManager:
    """Manages multiple species in the simulation."""
    
    def __init__(self, config: 'SimulationConfig'):
        self.config = config
        self.species_configs = {sp.name: sp for sp in config.species}
        self.populations: Dict[str, List] = {}  # species_name -> list of agents
        
    def initialize_populations(self, agent_factory):
        """Create initial populations using provided factory."""
        for sp_config in self.config.species:
            agents = []
            for _ in range(sp_config.initial_count):
                agent = agent_factory(sp_config)
                agents.append(agent)
            self.populations[sp_config.name] = agents
    
    def get_all_positions(self) -> Dict[str, np.ndarray]:
        """Get positions of all species for observation."""
        return {
            name: np.array([a.pos for a in agents]) if agents else np.empty((0, 2))
            for name, agents in self.populations.items()
        }
    
    def get_all_velocities(self) -> Dict[str, np.ndarray]:
        """Get velocities of all species."""
        return {
            name: np.array([a.vel for a in agents]) if agents else np.empty((0, 2))
            for name, agents in self.populations.items()
        }
    
    def get_species(self, name: str) -> List:
        return self.populations.get(name, [])
    
    def get_config(self, name: str) -> 'SpeciesConfig':
        return self.species_configs[name]
    
    def total_population(self) -> int:
        return sum(len(agents) for agents in self.populations.values())
    
    def remove_dead(self):
        """Remove dead agents from all populations."""
        for name in self.populations:
            self.populations[name] = [a for a in self.populations[name] if not a.should_die()]
    
    def add_agent(self, species_name: str, agent):
        """Add a new agent to a species."""
        self.populations[species_name].append(agent)
```

**Update World.py for N-species**:

```python
class World:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.width = config.world.width
        self.height = config.world.height
        self.boundary_mode = config.world.boundary_mode
        
        self.species_manager = SpeciesManager(config)
        self.species_manager.initialize_populations(self._create_agent)
        
        self.river = River(config.river, self.width, self.height) if config.river.enabled else None
        
    def _create_agent(self, sp_config: SpeciesConfig):
        """Factory method to create agents from config."""
        x = np.random.uniform(0, self.width)
        y = np.random.uniform(0, self.height)
        
        # Create appropriate agent type based on config
        if sp_config.has_energy_system:
            return Predator.from_config(x, y, sp_config, self.width, self.height)
        else:
            return Prey.from_config(x, y, sp_config, self.width, self.height)
    
    def step(self, mutation_rate: float = None):
        if mutation_rate is None:
            mutation_rate = self.config.mutation_rate
        
        # Get all positions/velocities for observations
        positions = self.species_manager.get_all_positions()
        velocities = self.species_manager.get_all_velocities()
        
        # Observation and action phase
        for species_name, agents in self.species_manager.populations.items():
            for i, agent in enumerate(agents):
                observation = agent.observe(self, positions, velocities, species_name, i)
                agent.act(observation)
        
        # Physics update
        for agents in self.species_manager.populations.values():
            for agent in agents:
                agent.update_physics(self.boundary_mode, self.width, self.height)
        
        # River effects
        if self.river:
            self._apply_river_effects()
        
        # Process predator-prey interactions
        self._process_interactions(positions)
        
        # Life cycle
        self._process_deaths()
        self._process_reproduction(mutation_rate)
        self._handle_extinction_prevention()
        
        self.timestep += 1
    
    def _process_interactions(self, positions: Dict[str, np.ndarray]):
        """Handle predator-prey interactions based on config."""
        for interaction in self.config.interactions:
            predators = self.species_manager.get_species(interaction.predator_species)
            prey_list = self.species_manager.get_species(interaction.prey_species)
            prey_positions = positions[interaction.prey_species]
            
            if len(predators) == 0 or len(prey_list) == 0:
                continue
            
            prey_to_remove = set()
            
            for pred in predators:
                if len(prey_positions) == 0:
                    break
                
                # Calculate distances using appropriate mode
                if self.boundary_mode == BoundaryMode.TOROIDAL:
                    distances, _ = toroidal_distance_numpy(
                        pred.pos, prey_positions, self.width, self.height
                    )
                else:
                    distances, _ = bounded_distance_numpy(
                        pred.pos, prey_positions, self.width, self.height
                    )
                
                # Find prey within catch radius
                in_range = np.where(distances < interaction.catch_radius)[0]
                
                for prey_idx in in_range:
                    if prey_idx not in prey_to_remove:
                        # Catch this prey
                        prey_to_remove.add(prey_idx)
                        if hasattr(pred, 'eat'):
                            pred.eat(interaction.energy_gain)
                        break  # One catch per predator per step
            
            # Remove caught prey (in reverse order to preserve indices)
            for idx in sorted(prey_to_remove, reverse=True):
                prey_list.pop(idx)
```

---

### 2.2 Dynamic Observation System

**Create**: `sensors.py`
**Effort**: Medium (half day)

```python
# sensors.py
"""Sensor system for agent observations."""

from abc import ABC, abstractmethod
from typing import List, Dict, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from world import World


class Sensor(ABC):
    """Base class for agent sensors."""
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of values this sensor produces."""
        pass
    
    @abstractmethod
    def observe(
        self, 
        agent, 
        world: 'World',
        positions: Dict[str, np.ndarray],
        velocities: Dict[str, np.ndarray],
        my_species: str,
        my_index: int
    ) -> np.ndarray:
        """Generate observation values.
        
        Returns:
            Flattened numpy array of sensor readings
        """
        pass


class NearestAgentsSensor(Sensor):
    """Sense nearest N agents of a target species."""
    
    def __init__(self, target_species: str, count: int, include_velocity: bool = True):
        self.target_species = target_species
        self.count = count
        self.include_velocity = include_velocity
        self._features_per_agent = 4 if include_velocity else 2
    
    @property
    def dimension(self) -> int:
        return self.count * self._features_per_agent
    
    def observe(self, agent, world, positions, velocities, my_species, my_index):
        from utils import toroidal_distance_numpy, bounded_distance_numpy
        from config_new import BoundaryMode
        
        target_pos = positions.get(self.target_species, np.empty((0, 2)))
        target_vel = velocities.get(self.target_species, np.empty((0, 2)))
        
        # Exclude self if observing own species
        if self.target_species == my_species and len(target_pos) > my_index:
            mask = np.ones(len(target_pos), dtype=bool)
            mask[my_index] = False
            target_pos = target_pos[mask]
            target_vel = target_vel[mask]
        
        if len(target_pos) == 0:
            return np.zeros(self.dimension, dtype=np.float32)
        
        # Calculate distances
        if world.boundary_mode == BoundaryMode.TOROIDAL:
            distances, vectors = toroidal_distance_numpy(
                agent.pos, target_pos, world.width, world.height
            )
        else:
            distances, vectors = bounded_distance_numpy(
                agent.pos, target_pos, world.width, world.height
            )
        
        # Get nearest
        nearest_idx = np.argsort(distances)[:self.count]
        
        observation = []
        for idx in nearest_idx:
            # Normalized direction
            dx = vectors[idx, 0] / world.width
            dy = vectors[idx, 1] / world.height
            observation.extend([dx, dy])
            
            if self.include_velocity:
                vx = target_vel[idx, 0] / 10.0  # Normalize velocity
                vy = target_vel[idx, 1] / 10.0
                observation.extend([vx, vy])
        
        # Pad if fewer than count targets
        while len(observation) < self.dimension:
            observation.append(0.0)
        
        return np.array(observation[:self.dimension], dtype=np.float32)


class HungerSensor(Sensor):
    """Sense own energy/hunger level."""
    
    @property
    def dimension(self) -> int:
        return 1
    
    def observe(self, agent, world, positions, velocities, my_species, my_index):
        if hasattr(agent, 'energy') and hasattr(agent, 'max_energy'):
            hunger = 1.0 - (agent.energy / agent.max_energy)
            return np.array([hunger], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)


class IslandProximitySensor(Sensor):
    """Sense whether on island (binary)."""
    
    @property
    def dimension(self) -> int:
        return 1
    
    def observe(self, agent, world, positions, velocities, my_species, my_index):
        if world.river and world.river.split:
            on_island = world.river.is_on_island(agent.pos[0], agent.pos[1])
            return np.array([1.0 if on_island else 0.0], dtype=np.float32)
        return np.array([0.0], dtype=np.float32)


class WallProximitySensor(Sensor):
    """Sense distance to nearest wall (for bounded mode)."""
    
    @property
    def dimension(self) -> int:
        return 4  # Distance to: left, right, top, bottom walls
    
    def observe(self, agent, world, positions, velocities, my_species, my_index):
        from config_new import BoundaryMode
        
        if world.boundary_mode != BoundaryMode.BOUNDED:
            return np.zeros(4, dtype=np.float32)
        
        x, y = agent.pos
        return np.array([
            x / world.width,                    # Distance to left
            (world.width - x) / world.width,    # Distance to right
            y / world.height,                   # Distance to top
            (world.height - y) / world.height   # Distance to bottom
        ], dtype=np.float32)


class SensorSuite:
    """Collection of sensors for an agent type."""
    
    def __init__(self, sensors: List[Sensor]):
        self.sensors = sensors
    
    @property
    def total_dimension(self) -> int:
        return sum(s.dimension for s in self.sensors)
    
    def observe(self, agent, world, positions, velocities, my_species, my_index) -> np.ndarray:
        if not self.sensors:
            return np.array([], dtype=np.float32)
        
        observations = [
            s.observe(agent, world, positions, velocities, my_species, my_index)
            for s in self.sensors
        ]
        return np.concatenate(observations)
    
    @classmethod
    def from_config(cls, obs_config: 'ObservationConfig') -> 'SensorSuite':
        """Create sensor suite from ObservationConfig."""
        sensors = []
        
        # Add nearest agents sensors
        for species_name, count in obs_config.observe_species.items():
            sensors.append(NearestAgentsSensor(species_name, count))
        
        # Add optional sensors
        if obs_config.sense_hunger:
            sensors.append(HungerSensor())
        if obs_config.sense_island:
            sensors.append(IslandProximitySensor())
        
        return cls(sensors)
```

---

## Phase 3: Optimization

### 3.1 Port River to GPU

**Create**: `river_gpu.py`
**Effort**: Medium (half day)

Implement fully GPU-resident river calculations to eliminate CPU transfers. See REFACTOR_PLAN.md for detailed implementation.

### 3.2 Add Trait System

**Create**: `traits.py`
**Effort**: Medium (half day)

Implement generic evolvable traits. See REFACTOR_PLAN.md for detailed implementation.

---

## Phase 4: Quality

### 4.1 Create Test Suite

**Create**: `tests/` directory with:
- `test_utils.py` - Distance and spawn utilities
- `test_config.py` - Configuration validation and serialization
- `test_sensors.py` - Sensor behavior
- `test_species.py` - Multi-species management
- `test_boundary_modes.py` - Toroidal vs bounded behavior
- `test_integration.py` - Full simulation runs

### 4.2 Update Documentation

- Update README with new architecture
- Create `docs/ADDING_SPECIES.md` - How to add a new species
- Create `docs/BOUNDARY_MODES.md` - Toroidal vs bounded explanation
- Create `docs/RUNNING_EXPERIMENTS.md` - Batch experiment guide

---

## Validation Checklist

### Phase 1 Complete:
- [ ] `utils.py` exists with tested functions
- [ ] `config_new.py` exists with `BoundaryMode` enum
- [ ] `default_two_species()` produces identical behavior to old config
- [ ] `default_bounded()` creates walled world
- [ ] Toroidal and bounded modes both work correctly
- [ ] Agents bounce/stop at walls in bounded mode
- [ ] Distance calculations correct for both modes

### Phase 2 Complete:
- [ ] `species.py` exists with `SpeciesManager`
- [ ] `sensors.py` exists with dynamic observation system
- [ ] Can configure species entirely via config (no hardcoding)
- [ ] Interactions defined in config work correctly

### Phase 3 Complete:
- [ ] `river_gpu.py` exists (optional - can defer)
- [ ] `traits.py` exists with `TraitDefinition` and `TraitSet`
- [ ] Can add new trait without code changes

### Phase 4 Complete:
- [ ] Test suite passes
- [ ] Documentation updated
- [ ] Can add 3rd species in <1 day

---

## Git Workflow

```bash
# Phase 1
git checkout -b refactor/foundation
git commit -m "feat: add utils.py with shared distance calculations"
git commit -m "feat: add config_new.py with BoundaryMode support"
git commit -m "feat: implement bounded world mode"
git checkout main && git merge refactor/foundation

# Phase 2
git checkout -b refactor/n-species
git commit -m "feat: add species.py with SpeciesManager"
git commit -m "feat: add sensors.py with dynamic observation system"
git commit -m "refactor: update World to use SpeciesManager"
git checkout main && git merge refactor/n-species

# etc.
```

## Begin

Start with Phase 1.1 - create `utils.py` and migrate distance calculations.
