"""
Generic trait system for evolvable agent properties.

This module provides a flexible trait definition system that allows any agent
property to be evolved through genetic algorithms with consistent mutation and
inheritance logic.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class Trait:
    """
    Definition for an evolvable trait.

    A trait represents any numeric property that can evolve through
    mutation and inheritance (e.g., speed, vision range, energy capacity).
    """

    name: str
    """Descriptive name of the trait (e.g., 'speed', 'vision_range')"""

    initial_value: float
    """Mean value for initial population"""

    initial_std: float
    """Standard deviation for initial population sampling"""

    mutation_std: float
    """Standard deviation for mutations (applied per mutation_rate)"""

    min_value: float
    """Minimum allowed value (hard constraint)"""

    max_value: float
    """Maximum allowed value (hard constraint)"""

    def sample_initial(self) -> float:
        """
        Sample an initial trait value for a new agent.

        Uses normal distribution centered on initial_value with initial_std,
        then clamps to [min_value, max_value].

        Returns:
            Random trait value for initial population
        """
        value = np.random.normal(self.initial_value, self.initial_std)
        return np.clip(value, self.min_value, self.max_value)

    def mutate(self, parent_value: float, mutation_rate: float) -> float:
        """
        Apply mutation to a parent's trait value.

        Mutation is Gaussian noise scaled by mutation_std and mutation_rate.
        Result is clamped to [min_value, max_value].

        Args:
            parent_value: The parent's trait value
            mutation_rate: Global mutation rate (typically 0.05 - 0.2)

        Returns:
            Mutated trait value for offspring
        """
        noise = np.random.randn() * self.mutation_std * mutation_rate
        value = parent_value + noise
        return np.clip(value, self.min_value, self.max_value)

    def __repr__(self):
        return (f"Trait(name={self.name!r}, "
                f"initial={self.initial_value:.2f}±{self.initial_std:.2f}, "
                f"mutation_std={self.mutation_std:.2f}, "
                f"range=[{self.min_value:.2f}, {self.max_value:.2f}])")


class TraitCollection:
    """
    Collection of traits for an agent type.

    This class manages multiple traits and provides convenient methods
    for initialization and mutation of entire trait sets.
    """

    def __init__(self, trait_definitions: Dict[str, Trait]):
        """
        Initialize trait collection.

        Args:
            trait_definitions: Dictionary mapping trait names to Trait objects
        """
        self.definitions = trait_definitions

    def sample_initial_values(self) -> Dict[str, float]:
        """
        Sample initial values for all traits.

        Returns:
            Dictionary mapping trait names to initial values
        """
        return {name: trait.sample_initial()
                for name, trait in self.definitions.items()}

    def mutate_values(self, parent_values: Dict[str, float],
                     mutation_rate: float) -> Dict[str, float]:
        """
        Mutate all trait values from parent.

        Args:
            parent_values: Parent's trait values
            mutation_rate: Global mutation rate

        Returns:
            Dictionary mapping trait names to mutated values
        """
        return {name: self.definitions[name].mutate(parent_values[name], mutation_rate)
                for name in self.definitions}

    def get_trait_names(self):
        """Get list of trait names."""
        return list(self.definitions.keys())

    def __repr__(self):
        return f"TraitCollection({len(self.definitions)} traits: {list(self.definitions.keys())})"


# Common trait definitions for easy reuse

COMMON_PREY_TRAITS = {
    'max_speed': Trait(
        name='max_speed',
        initial_value=3.0,
        initial_std=0.3,
        mutation_std=0.2,
        min_value=0.5,
        max_value=10.0
    ),
    'swim_speed': Trait(
        name='swim_speed',
        initial_value=2.0,
        initial_std=0.2,
        mutation_std=0.3,
        min_value=0.1,
        max_value=5.0
    ),
    'max_acceleration': Trait(
        name='max_acceleration',
        initial_value=0.5,
        initial_std=0.05,
        mutation_std=0.05,
        min_value=0.1,
        max_value=2.0
    ),
}

COMMON_PREDATOR_TRAITS = {
    'max_speed': Trait(
        name='max_speed',
        initial_value=2.5,
        initial_std=0.25,
        mutation_std=0.15,
        min_value=0.5,
        max_value=8.0
    ),
    'swim_speed': Trait(
        name='swim_speed',
        initial_value=1.5,
        initial_std=0.15,
        mutation_std=0.2,
        min_value=0.1,
        max_value=4.0
    ),
    'max_acceleration': Trait(
        name='max_acceleration',
        initial_value=0.4,
        initial_std=0.04,
        mutation_std=0.04,
        min_value=0.1,
        max_value=1.5
    ),
    'max_energy': Trait(
        name='max_energy',
        initial_value=150.0,
        initial_std=15.0,
        mutation_std=10.0,
        min_value=50.0,
        max_value=500.0
    ),
}


# Example usage and integration patterns

def example_usage():
    """Example of how to use the trait system."""

    # Define custom traits for a new agent type
    herbivore_traits = TraitCollection({
        'speed': Trait('speed', 2.5, 0.2, 0.15, 0.5, 8.0),
        'vision_range': Trait('vision_range', 100.0, 10.0, 15.0, 20.0, 300.0),
        'stamina': Trait('stamina', 100.0, 10.0, 8.0, 20.0, 200.0),
    })

    # Create initial agent
    initial_values = herbivore_traits.sample_initial_values()
    print(f"Initial traits: {initial_values}")

    # Simulate reproduction with mutation
    child_values = herbivore_traits.mutate_values(initial_values, mutation_rate=0.1)
    print(f"Child traits: {child_values}")

    # Access individual traits
    agent_speed = initial_values['speed']
    agent_vision = initial_values['vision_range']
    print(f"Agent has speed={agent_speed:.2f}, vision={agent_vision:.2f}")


if __name__ == "__main__":
    print("=== Trait System Example ===\n")
    example_usage()

    print("\n=== Common Prey Traits ===")
    for name, trait in COMMON_PREY_TRAITS.items():
        print(f"  {trait}")

    print("\n=== Common Predator Traits ===")
    for name, trait in COMMON_PREDATOR_TRAITS.items():
        print(f"  {trait}")
