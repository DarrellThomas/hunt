"""Sensor system for agent observations.

This module provides a flexible sensor system that allows agents to have
dynamic observation configurations based on their needs.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from world import World
    from config_new import ObservationConfig


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

        Args:
            agent: The agent that is observing
            world: The world instance
            positions: Dictionary mapping species name to position array
            velocities: Dictionary mapping species name to velocity array
            my_species: Name of the agent's species
            my_index: Index of this agent in its species list

        Returns:
            Flattened numpy array of sensor readings
        """
        pass


class NearestAgentsSensor(Sensor):
    """Sense nearest N agents of a target species."""

    def __init__(self, target_species: str, count: int, include_velocity: bool = True):
        """Initialize nearest agents sensor.

        Args:
            target_species: Name of species to observe
            count: Number of nearest agents to sense
            include_velocity: Whether to include velocity in observations
        """
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

        # Calculate distances using appropriate boundary mode
        if world.boundary_mode == BoundaryMode.TOROIDAL:
            distances, vectors = toroidal_distance_numpy(
                agent.pos, target_pos, world.width, world.height
            )
        else:
            distances, vectors = bounded_distance_numpy(
                agent.pos, target_pos, world.width, world.height
            )

        # Get nearest agents
        num_to_observe = min(self.count, len(distances))
        if num_to_observe == 0:
            return np.zeros(self.dimension, dtype=np.float32)

        nearest_idx = np.argsort(distances)[:num_to_observe]

        observation = []
        for idx in nearest_idx:
            # Normalized direction
            dx = vectors[idx, 0] / world.width
            dy = vectors[idx, 1] / world.height
            observation.extend([dx, dy])

            if self.include_velocity:
                # Normalize velocity
                vx = target_vel[idx, 0] / 10.0
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
    """Collection of sensors for an agent type.

    This combines multiple sensors into a single observation vector.
    """

    def __init__(self, sensors: List[Sensor]):
        """Initialize sensor suite.

        Args:
            sensors: List of Sensor instances
        """
        self.sensors = sensors

    @property
    def total_dimension(self) -> int:
        """Get total observation dimension across all sensors.

        Returns:
            Sum of dimensions from all sensors
        """
        return sum(s.dimension for s in self.sensors)

    def observe(
        self,
        agent,
        world: 'World',
        positions: Dict[str, np.ndarray],
        velocities: Dict[str, np.ndarray],
        my_species: str,
        my_index: int
    ) -> np.ndarray:
        """Generate complete observation by combining all sensors.

        Args:
            agent: The agent that is observing
            world: The world instance
            positions: Dictionary mapping species name to position array
            velocities: Dictionary mapping species name to velocity array
            my_species: Name of the agent's species
            my_index: Index of this agent in its species list

        Returns:
            Concatenated observations from all sensors
        """
        if not self.sensors:
            return np.array([], dtype=np.float32)

        observations = [
            s.observe(agent, world, positions, velocities, my_species, my_index)
            for s in self.sensors
        ]
        return np.concatenate(observations)

    @classmethod
    def from_config(cls, obs_config: 'ObservationConfig') -> 'SensorSuite':
        """Create sensor suite from ObservationConfig.

        Args:
            obs_config: ObservationConfig with sensor specifications

        Returns:
            SensorSuite instance configured from obs_config
        """
        sensors = []

        # Add nearest agents sensors for each species
        for species_name, count in obs_config.observe_species.items():
            sensors.append(NearestAgentsSensor(species_name, count))

        # Add optional sensors
        if obs_config.sense_hunger:
            sensors.append(HungerSensor())
        if obs_config.sense_island:
            sensors.append(IslandProximitySensor())

        return cls(sensors)
