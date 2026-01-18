"""
Optimized world simulation with vectorized operations.
"""

import numpy as np
from agent import Prey, Predator
from river import River


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

        # Create river
        self.river = River(width, height)

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
            # Apply river flow only in river direction (preserves perpendicular motion)
            if self.river.enabled:
                flow_x, flow_y = self.river.get_flow_at(prey.pos[0], prey.pos[1])
                flow_mag = np.sqrt(flow_x**2 + flow_y**2)

                if flow_mag > 0:
                    # Get river direction (unit vector)
                    river_dir_x = flow_x / flow_mag
                    river_dir_y = flow_y / flow_mag

                    # Decompose agent velocity into parallel and perpendicular components
                    vel_parallel_mag = prey.vel[0] * river_dir_x + prey.vel[1] * river_dir_y  # dot product
                    vel_parallel_x = vel_parallel_mag * river_dir_x
                    vel_parallel_y = vel_parallel_mag * river_dir_y
                    vel_perp_x = prey.vel[0] - vel_parallel_x
                    vel_perp_y = prey.vel[1] - vel_parallel_y

                    # Add river flow to parallel component only (reduced by swim speed)
                    flow_factor = max(0, 1.0 - prey.swim_speed / 5.0)
                    vel_parallel_x += flow_x * flow_factor
                    vel_parallel_y += flow_y * flow_factor

                    # Reconstruct velocity (perpendicular component unchanged)
                    prey.vel[0] = vel_parallel_x + vel_perp_x
                    prey.vel[1] = vel_parallel_y + vel_perp_y
            prey.time_since_reproduction += 1

        for predator in self.predators:
            predator.update_physics()
            # Apply river flow only in river direction (preserves perpendicular motion)
            if self.river.enabled:
                flow_x, flow_y = self.river.get_flow_at(predator.pos[0], predator.pos[1])
                flow_mag = np.sqrt(flow_x**2 + flow_y**2)

                if flow_mag > 0:
                    # Get river direction (unit vector)
                    river_dir_x = flow_x / flow_mag
                    river_dir_y = flow_y / flow_mag

                    # Decompose agent velocity into parallel and perpendicular components
                    vel_parallel_mag = predator.vel[0] * river_dir_x + predator.vel[1] * river_dir_y  # dot product
                    vel_parallel_x = vel_parallel_mag * river_dir_x
                    vel_parallel_y = vel_parallel_mag * river_dir_y
                    vel_perp_x = predator.vel[0] - vel_parallel_x
                    vel_perp_y = predator.vel[1] - vel_parallel_y

                    # Add river flow to parallel component only (reduced by swim speed)
                    flow_factor = max(0, 1.0 - predator.swim_speed / 5.0)
                    vel_parallel_x += flow_x * flow_factor
                    vel_parallel_y += flow_y * flow_factor

                    # Reconstruct velocity (perpendicular component unchanged)
                    predator.vel[0] = vel_parallel_x + vel_perp_x
                    predator.vel[1] = vel_parallel_y + vel_perp_y
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

        # 7. Unified extinction prevention
        self._handle_extinction_prevention()

        # 9. Record statistics
        self.record_stats()

    def _handle_extinction_prevention(
        self,
        enabled: bool = True,
        emergency_respawn_count: int = 5,
        minimum_population: int = 10,
        scale_with_world_size: bool = True
    ):
        """Unified extinction prevention for both emergency and minimum population.

        Args:
            enabled: Whether extinction prevention is enabled
            emergency_respawn_count: Number of agents to respawn on complete extinction
            minimum_population: Baseline minimum population (per species)
            scale_with_world_size: If True, scale minimum with world area
        """
        if not enabled:
            return

        # Calculate minimum populations (scale with world size if requested)
        if scale_with_world_size:
            min_prey = max(minimum_population, int(self.width * self.height / 24000))
            min_predators = max(minimum_population // 3, int(self.width * self.height / 160000))
        else:
            min_prey = minimum_population
            min_predators = minimum_population // 3

        # Handle prey
        prey_count = len(self.prey)
        if prey_count < 1:
            # Emergency: complete extinction
            print(f"\n⚠️  PREY EXTINCTION at timestep {self.timestep}! Respawning {emergency_respawn_count} random prey...")
            for _ in range(emergency_respawn_count):
                x = np.random.uniform(0, self.width)
                y = np.random.uniform(0, self.height)
                self.prey.append(Prey(x, y, self.width, self.height))
        elif prey_count < min_prey:
            # Below minimum threshold: gradually repopulate
            spawn_count = min(min_prey - prey_count, emergency_respawn_count)  # Cap spawn rate
            for _ in range(spawn_count):
                x = np.random.uniform(0, self.width)
                y = np.random.uniform(0, self.height)
                self.prey.append(Prey(x, y, self.width, self.height))

        # Handle predators
        pred_count = len(self.predators)
        if pred_count < 1:
            # Emergency: complete extinction
            print(f"\n⚠️  PREDATOR EXTINCTION at timestep {self.timestep}! Respawning {emergency_respawn_count} random predators...")
            for _ in range(emergency_respawn_count):
                x = np.random.uniform(0, self.width)
                y = np.random.uniform(0, self.height)
                self.predators.append(Predator(x, y, self.width, self.height))
        elif pred_count < min_predators:
            # Below minimum threshold: gradually repopulate
            spawn_count = min(min_predators - pred_count, emergency_respawn_count)  # Cap spawn rate
            for _ in range(spawn_count):
                x = np.random.uniform(0, self.width)
                y = np.random.uniform(0, self.height)
                self.predators.append(Predator(x, y, self.width, self.height))

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
