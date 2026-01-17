"""Configuration system for HUNT simulation.

This module provides a dataclass-based configuration system that supports:
- Multiple species with individual parameters
- Boundary modes (toroidal vs bounded worlds)
- JSON serialization for experiment saving
- Validation of configuration values
- Extensible species interactions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
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
