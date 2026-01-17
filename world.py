"""
World simulation that manages the ecosystem.
Handles physics, collisions, reproduction, and death.
"""

import numpy as np
from agent import Prey, Predator


class World:
    """The ecosystem where predators and prey co-evolve."""

    def __init__(self, width=800, height=600, initial_prey=50, initial_predators=10):
        """
        Initialize the world.

        Args:
            width, height: World dimensions
            initial_prey: Starting number of prey
            initial_predators: Starting number of predators
        """
        self.width = width
        self.height = height
        self.timestep = 0

        # Create initial populations
        self.prey = []
        self.predators = []

        for _ in range(initial_prey):
            x = np.random.uniform(0, width)
            y = np.random.uniform(0, height)
            self.prey.append(Prey(x, y, width, height))

        for _ in range(initial_predators):
            x = np.random.uniform(0, width)
            y = np.random.uniform(0, height)
            self.predators.append(Predator(x, y, width, height))

        # Statistics
        self.stats = {
            'prey_count': [],
            'predator_count': [],
            'prey_avg_age': [],
            'predator_avg_age': [],
            'prey_avg_fitness': [],
            'predator_avg_fitness': []
        }

    def step(self, mutation_rate=0.1):
        """
        Simulate one timestep of the ecosystem.

        Args:
            mutation_rate: Mutation rate for offspring
        """
        self.timestep += 1

        # 1. Agents observe and act
        for prey in self.prey:
            observation = prey.observe(self)
            prey.act(observation)

        for predator in self.predators:
            observation = predator.observe(self)
            predator.act(observation)

        # 2. Update physics
        for prey in self.prey:
            prey.update_physics()
            prey.time_since_reproduction += 1

        for predator in self.predators:
            predator.update_physics()
            predator.update_energy()

        # 3. Check collisions (predators catching prey)
        prey_to_remove = set()
        for predator in self.predators:
            for i, prey in enumerate(self.prey):
                if i not in prey_to_remove:
                    if predator.distance_to(prey) < predator.catch_radius:
                        # Predator catches prey
                        predator.eat()
                        prey_to_remove.add(i)
                        break  # Each predator can only catch one prey per timestep

        # Remove caught prey
        self.prey = [p for i, p in enumerate(self.prey) if i not in prey_to_remove]

        # 4. Update fitness
        for prey in self.prey:
            prey.update_fitness()

        for predator in self.predators:
            predator.update_fitness()

        # 5. Handle deaths (old age, starvation)
        self.prey = [p for p in self.prey if not p.should_die()]
        self.predators = [p for p in self.predators if not p.should_die()]

        # 6. Handle reproduction
        new_prey = []
        for prey in self.prey:
            if prey.can_reproduce():
                child = prey.reproduce(mutation_rate)
                new_prey.append(child)
                prey.time_since_reproduction = 0  # Reset reproduction timer

        new_predators = []
        for predator in self.predators:
            if predator.can_reproduce():
                child = predator.reproduce(mutation_rate)
                new_predators.append(child)
                predator.pay_reproduction_cost()

        self.prey.extend(new_prey)
        self.predators.extend(new_predators)

        # 7. Prevent extinction - spawn random agents if population too low
        if len(self.prey) < 10:
            for _ in range(10 - len(self.prey)):
                x = np.random.uniform(0, self.width)
                y = np.random.uniform(0, self.height)
                self.prey.append(Prey(x, y, self.width, self.height))

        if len(self.predators) < 3:
            for _ in range(3 - len(self.predators)):
                x = np.random.uniform(0, self.width)
                y = np.random.uniform(0, self.height)
                self.predators.append(Predator(x, y, self.width, self.height))

        # 8. Record statistics
        self.record_stats()

    def record_stats(self):
        """Record statistics about the current state."""
        self.stats['prey_count'].append(len(self.prey))
        self.stats['predator_count'].append(len(self.predators))

        if len(self.prey) > 0:
            self.stats['prey_avg_age'].append(np.mean([p.age for p in self.prey]))
            self.stats['prey_avg_fitness'].append(np.mean([p.fitness for p in self.prey]))
        else:
            self.stats['prey_avg_age'].append(0)
            self.stats['prey_avg_fitness'].append(0)

        if len(self.predators) > 0:
            self.stats['predator_avg_age'].append(np.mean([p.age for p in self.predators]))
            self.stats['predator_avg_fitness'].append(np.mean([p.fitness for p in self.predators]))
        else:
            self.stats['predator_avg_age'].append(0)
            self.stats['predator_avg_fitness'].append(0)

    def get_state(self):
        """Get current state for visualization."""
        return {
            'timestep': self.timestep,
            'prey_positions': np.array([p.pos for p in self.prey]) if self.prey else np.array([]),
            'predator_positions': np.array([p.pos for p in self.predators]) if self.predators else np.array([]),
            'prey_count': len(self.prey),
            'predator_count': len(self.predators),
        }

    def save_stats(self, filename='stats.npz'):
        """Save statistics to file."""
        np.savez(filename, **self.stats)
        print(f"Statistics saved to {filename}")

    def print_stats(self):
        """Print current statistics."""
        print(f"\n=== Timestep {self.timestep} ===")
        print(f"Prey: {len(self.prey)} | Predators: {len(self.predators)}")
        if len(self.prey) > 0:
            print(f"Prey avg age: {np.mean([p.age for p in self.prey]):.1f} | avg fitness: {np.mean([p.fitness for p in self.prey]):.1f}")
        if len(self.predators) > 0:
            print(f"Predator avg age: {np.mean([p.age for p in self.predators]):.1f} | avg fitness: {np.mean([p.fitness for p in self.predators]):.1f}")
