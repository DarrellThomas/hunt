"""
Optimized world simulation with vectorized operations.
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
        Simulate one timestep of the ecosystem with vectorized operations.

        Args:
            mutation_rate: Mutation rate for offspring
        """
        self.timestep += 1

        # Pre-compute position and velocity arrays (vectorized!)
        prey_positions = np.array([p.pos for p in self.prey]) if self.prey else np.array([]).reshape(0, 2)
        prey_velocities = np.array([p.vel for p in self.prey]) if self.prey else np.array([]).reshape(0, 2)
        predator_positions = np.array([p.pos for p in self.predators]) if self.predators else np.array([]).reshape(0, 2)
        predator_velocities = np.array([p.vel for p in self.predators]) if self.predators else np.array([]).reshape(0, 2)

        # 1. Agents observe and act (using pre-computed arrays)
        for i, prey in enumerate(self.prey):
            observation = prey.observe(predator_positions, predator_velocities,
                                      prey_positions, prey_velocities, i)
            prey.act(observation)

        for predator in self.predators:
            observation = predator.observe(prey_positions, prey_velocities)
            predator.act(observation)

        # 2. Update physics
        for prey in self.prey:
            prey.update_physics()
            prey.time_since_reproduction += 1

        for predator in self.predators:
            predator.update_physics()
            predator.update_energy()
            predator.time_since_reproduction += 1

        # 3. Check collisions (vectorized!)
        prey_to_remove = set()
        if len(self.predators) > 0 and len(self.prey) > 0:
            # Update positions after physics
            prey_positions = np.array([p.pos for p in self.prey])
            predator_positions = np.array([p.pos for p in self.predators])

            for pred_idx, predator in enumerate(self.predators):
                if len(prey_to_remove) >= len(self.prey):
                    break

                # Vectorized distance calculation for this predator to all remaining prey
                dx = prey_positions[:, 0] - predator.pos[0]
                dy = prey_positions[:, 1] - predator.pos[1]

                # Toroidal wrapping
                dx = np.where(np.abs(dx) > self.width / 2, dx - np.sign(dx) * self.width, dx)
                dy = np.where(np.abs(dy) > self.height / 2, dy - np.sign(dy) * self.height, dy)

                distances = np.sqrt(dx**2 + dy**2)

                # Find prey within catch radius
                caught_mask = distances < predator.catch_radius
                caught_indices = np.where(caught_mask)[0]

                # Remove already-caught prey from consideration
                caught_indices = [idx for idx in caught_indices if idx not in prey_to_remove]

                if len(caught_indices) > 0:
                    # Catch the nearest one
                    nearest_idx = caught_indices[np.argmin(distances[caught_indices])]
                    predator.eat()
                    prey_to_remove.add(nearest_idx)

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
                predator.time_since_reproduction = 0  # Reset reproduction timer

        self.prey.extend(new_prey)
        self.predators.extend(new_predators)

        # 7. Emergency extinction prevention - respawn 5 if population hits 0
        if len(self.prey) < 1:
            print(f"\n⚠️  PREY EXTINCTION at timestep {self.timestep}! Respawning 5 random prey...")
            for _ in range(5):
                x = np.random.uniform(0, self.width)
                y = np.random.uniform(0, self.height)
                self.prey.append(Prey(x, y, self.width, self.height))

        if len(self.predators) < 1:
            print(f"\n⚠️  PREDATOR EXTINCTION at timestep {self.timestep}! Respawning 5 random predators...")
            for _ in range(5):
                x = np.random.uniform(0, self.width)
                y = np.random.uniform(0, self.height)
                self.predators.append(Predator(x, y, self.width, self.height))

        # 8. Prevent extinction - spawn random agents if population too low
        # Minimum thresholds scale with world size
        min_prey = max(10, int(self.width * self.height / 24000))  # ~40 for 1600x1200
        min_predators = max(3, int(self.width * self.height / 160000))  # ~12 for 1600x1200

        if len(self.prey) < min_prey:
            for _ in range(min_prey - len(self.prey)):
                x = np.random.uniform(0, self.width)
                y = np.random.uniform(0, self.height)
                self.prey.append(Prey(x, y, self.width, self.height))

        if len(self.predators) < min_predators:
            for _ in range(min_predators - len(self.predators)):
                x = np.random.uniform(0, self.width)
                y = np.random.uniform(0, self.height)
                self.predators.append(Predator(x, y, self.width, self.height))

        # 9. Record statistics
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
