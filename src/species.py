"""Species management for HUNT simulation.

This module provides N-species support, allowing arbitrary numbers of species
to coexist and interact in the simulation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
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
    """Manages multiple species in the simulation.

    This class replaces hardcoded prey/predator lists with a flexible
    dictionary-based system that can handle arbitrary numbers of species.
    """

    def __init__(self, config: 'SimulationConfig'):
        """Initialize species manager from config.

        Args:
            config: SimulationConfig with species definitions
        """
        self.config = config
        self.species_configs = {sp.name: sp for sp in config.species}
        self.populations: Dict[str, List] = {}  # species_name -> list of agents
        self.timestep = 0

    def initialize_populations(self, agent_factory):
        """Create initial populations using provided factory.

        Args:
            agent_factory: Callable that takes (SpeciesConfig) and returns an agent
        """
        for sp_config in self.config.species:
            agents = []
            for _ in range(sp_config.initial_count):
                agent = agent_factory(sp_config)
                agents.append(agent)
            self.populations[sp_config.name] = agents

    def get_all_positions(self) -> Dict[str, np.ndarray]:
        """Get positions of all species for observation.

        Returns:
            Dictionary mapping species name to (N, 2) position array
        """
        return {
            name: np.array([a.pos for a in agents]) if agents else np.empty((0, 2))
            for name, agents in self.populations.items()
        }

    def get_all_velocities(self) -> Dict[str, np.ndarray]:
        """Get velocities of all species.

        Returns:
            Dictionary mapping species name to (N, 2) velocity array
        """
        return {
            name: np.array([a.vel for a in agents]) if agents else np.empty((0, 2))
            for name, agents in self.populations.items()
        }

    def get_species(self, name: str) -> List:
        """Get list of agents for a species.

        Args:
            name: Species name

        Returns:
            List of agents (empty list if species doesn't exist)
        """
        return self.populations.get(name, [])

    def get_config(self, name: str) -> Optional['SpeciesConfig']:
        """Get configuration for a species.

        Args:
            name: Species name

        Returns:
            SpeciesConfig or None if not found
        """
        return self.species_configs.get(name)

    def total_population(self) -> int:
        """Get total number of agents across all species.

        Returns:
            Sum of all population sizes
        """
        return sum(len(agents) for agents in self.populations.values())

    def get_population_count(self, name: str) -> int:
        """Get population count for a species.

        Args:
            name: Species name

        Returns:
            Number of agents in that species
        """
        return len(self.populations.get(name, []))

    def remove_dead(self):
        """Remove dead agents from all populations.

        Agents with should_die() returning True are removed.
        """
        for name in self.populations:
            self.populations[name] = [
                a for a in self.populations[name]
                if not (hasattr(a, 'should_die') and a.should_die())
            ]

    def add_agent(self, species_name: str, agent):
        """Add a new agent to a species.

        Args:
            species_name: Name of species to add to
            agent: Agent instance to add
        """
        if species_name not in self.populations:
            self.populations[species_name] = []
        self.populations[species_name].append(agent)

    def remove_agents_by_index(self, species_name: str, indices: List[int]):
        """Remove agents at specific indices from a species.

        Args:
            species_name: Name of species
            indices: List of indices to remove (must be sorted in descending order)
        """
        if species_name not in self.populations:
            return

        agents = self.populations[species_name]
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(agents):
                agents.pop(idx)

    def get_species_list(self) -> List[str]:
        """Get list of all species names.

        Returns:
            List of species names
        """
        return list(self.species_configs.keys())

    def stats_summary(self) -> Dict[str, int]:
        """Get population counts for all species.

        Returns:
            Dictionary mapping species name to population count
        """
        return {name: len(agents) for name, agents in self.populations.items()}
