"""
Optimized agent classes with vectorized distance calculations.
"""

import numpy as np
from brain import Brain


class Agent:
    """Base class for all agents in the ecosystem."""

    def __init__(self, x, y, world_width, world_height, brain=None, input_size=10):
        """
        Initialize an agent.

        Args:
            x, y: Initial position
            world_width, world_height: World boundaries
            brain: Optional pre-existing brain (for offspring)
            input_size: Size of sensory input
        """
        self.pos = np.array([x, y], dtype=np.float64)
        self.vel = np.random.randn(2) * 0.1  # Small random initial velocity
        self.acc = np.zeros(2, dtype=np.float64)

        self.world_width = world_width
        self.world_height = world_height

        self.age = 0
        self.fitness = 0

        # Create or copy brain
        if brain is None:
            self.brain = Brain(input_size=input_size, hidden_size=32, output_size=2)
        else:
            self.brain = brain.copy()

    def observe(self, world):
        """
        Observe the environment and return sensory input.
        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def act(self, observation):
        """
        Use brain to decide action based on observation.

        Args:
            observation: Sensory input array

        Returns:
            Action (acceleration vector)
        """
        output = self.brain.forward(observation)
        # Output is in range [-1, 1], scale to acceleration
        self.acc = output * self.max_acceleration
        return self.acc

    def update_physics(self, dt=1.0):
        """Update velocity and position based on acceleration."""
        # Update velocity with acceleration
        self.vel += self.acc * dt

        # Limit speed
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = (self.vel / speed) * self.max_speed

        # Update position
        self.pos += self.vel * dt

        # Wrap around edges (toroidal world)
        self.pos[0] = self.pos[0] % self.world_width
        self.pos[1] = self.pos[1] % self.world_height

        # Age the agent
        self.age += 1

    def reproduce(self, mutation_rate=0.1):
        """
        Create offspring with mutated brain.

        Args:
            mutation_rate: Amount of mutation to apply

        Returns:
            New agent (offspring)
        """
        # Spawn near parent
        offset = np.random.randn(2) * 20
        child_x = (self.pos[0] + offset[0]) % self.world_width
        child_y = (self.pos[1] + offset[1]) % self.world_height

        # Create child with copied brain
        child = self.__class__(child_x, child_y, self.world_width, self.world_height, brain=self.brain)

        # Mutate child's brain
        child.brain.mutate(mutation_rate)

        return child


def vectorized_distances(pos, other_positions, world_width, world_height):
    """
    Vectorized distance calculation with toroidal wrapping.

    Args:
        pos: Single position [x, y]
        other_positions: Array of positions (N, 2)
        world_width, world_height: World dimensions

    Returns:
        distances: Array of distances (N,)
        vectors: Array of direction vectors (N, 2)
    """
    if len(other_positions) == 0:
        return np.array([]), np.array([]).reshape(0, 2)

    # Calculate raw differences
    dx = other_positions[:, 0] - pos[0]
    dy = other_positions[:, 1] - pos[1]

    # Apply toroidal wrapping - take shortest path
    dx = np.where(np.abs(dx) > world_width / 2, dx - np.sign(dx) * world_width, dx)
    dy = np.where(np.abs(dy) > world_height / 2, dy - np.sign(dy) * world_height, dy)

    # Calculate distances
    distances = np.sqrt(dx**2 + dy**2)
    vectors = np.column_stack([dx, dy])

    return distances, vectors


class Prey(Agent):
    """Prey agents that try to survive."""

    def __init__(self, x, y, world_width, world_height, brain=None):
        # Prey observes: 5 nearest predators (x, y, vx, vy each) + 3 nearest prey = 5*4 + 3*4 = 32 inputs
        super().__init__(x, y, world_width, world_height, brain, input_size=32)

        self.max_speed = 3.0
        self.max_acceleration = 0.5

        # Add natural variation to lifespan and reproduction timing
        # Normal distribution prevents synchronized birth/death waves
        self.max_lifespan = max(100, int(np.random.normal(500, 50)))  # Mean 500, std 50
        self.reproduction_age = max(50, int(np.random.normal(200, 20)))  # Mean 200, std 20
        self.time_since_reproduction = 0

    def observe(self, predator_positions, predator_velocities, prey_positions, prey_velocities, my_index):
        """
        Observe nearby predators and prey using vectorized operations.

        Args:
            predator_positions: Array of predator positions (N, 2)
            predator_velocities: Array of predator velocities (N, 2)
            prey_positions: Array of prey positions (M, 2)
            prey_velocities: Array of prey velocities (M, 2)
            my_index: This prey's index in the arrays

        Returns:
            Observation vector (numpy array)
        """
        observation = []

        # Find 5 nearest predators
        if len(predator_positions) > 0:
            distances, vectors = vectorized_distances(
                self.pos, predator_positions, self.world_width, self.world_height
            )
            nearest_indices = np.argsort(distances)[:5]

            for i in range(5):
                if i < len(nearest_indices):
                    idx = nearest_indices[i]
                    vec_norm = vectors[idx] / np.array([self.world_width, self.world_height])
                    vel_norm = predator_velocities[idx] / self.max_speed
                    observation.extend([vec_norm[0], vec_norm[1], vel_norm[0], vel_norm[1]])
                else:
                    observation.extend([0, 0, 0, 0])
        else:
            observation.extend([0] * 20)

        # Find 3 nearest other prey (excluding self)
        if len(prey_positions) > 1:
            # Remove self from consideration
            mask = np.ones(len(prey_positions), dtype=bool)
            mask[my_index] = False
            other_positions = prey_positions[mask]
            other_velocities = prey_velocities[mask]

            distances, vectors = vectorized_distances(
                self.pos, other_positions, self.world_width, self.world_height
            )
            nearest_indices = np.argsort(distances)[:3]

            for i in range(3):
                if i < len(nearest_indices):
                    idx = nearest_indices[i]
                    vec_norm = vectors[idx] / np.array([self.world_width, self.world_height])
                    vel_norm = other_velocities[idx] / self.max_speed
                    observation.extend([vec_norm[0], vec_norm[1], vel_norm[0], vel_norm[1]])
                else:
                    observation.extend([0, 0, 0, 0])
        else:
            observation.extend([0] * 12)

        return np.array(observation, dtype=np.float32)

    def should_die(self):
        """Check if prey should die of old age."""
        return self.age >= self.max_lifespan

    def can_reproduce(self):
        """Check if prey can reproduce."""
        return self.time_since_reproduction >= self.reproduction_age

    def update_fitness(self):
        """Update fitness based on survival."""
        self.fitness = self.age  # Fitness = survival time


class Predator(Agent):
    """Predator agents that must hunt to survive."""

    def __init__(self, x, y, world_width, world_height, brain=None):
        # Predator observes: 5 nearest prey (x, y, vx, vy each) + own hunger = 5*4 + 1 = 21 inputs
        super().__init__(x, y, world_width, world_height, brain, input_size=21)

        self.max_speed = 2.5  # Slightly slower than prey
        self.max_acceleration = 0.4

        # Add natural variation to lifespan and reproduction timing
        # Normal distribution prevents synchronized birth/death waves
        self.max_lifespan = max(200, int(np.random.normal(800, 80)))  # Mean 800, std 80
        self.reproduction_cooldown = max(50, int(np.random.normal(150, 15)))  # Mean 150, std 15

        self.energy = 150
        self.max_energy = 150
        self.energy_cost_per_step = 0.3
        self.energy_gain_per_kill = 60
        self.reproduction_threshold = 120
        self.reproduction_cost = 40
        self.time_since_reproduction = 0
        self.catch_radius = 8.0  # Larger radius to help random movement succeed occasionally

    def observe(self, prey_positions, prey_velocities):
        """
        Observe nearby prey and own hunger using vectorized operations.

        Args:
            prey_positions: Array of prey positions (N, 2)
            prey_velocities: Array of prey velocities (N, 2)

        Returns:
            Observation vector (numpy array)
        """
        observation = []

        # Find 5 nearest prey
        if len(prey_positions) > 0:
            distances, vectors = vectorized_distances(
                self.pos, prey_positions, self.world_width, self.world_height
            )
            nearest_indices = np.argsort(distances)[:5]

            for i in range(5):
                if i < len(nearest_indices):
                    idx = nearest_indices[i]
                    vec_norm = vectors[idx] / np.array([self.world_width, self.world_height])
                    vel_norm = prey_velocities[idx] / self.max_speed
                    observation.extend([vec_norm[0], vec_norm[1], vel_norm[0], vel_norm[1]])
                else:
                    observation.extend([0, 0, 0, 0])
        else:
            observation.extend([0] * 20)

        # Add hunger level (normalized)
        hunger = 1.0 - (self.energy / self.max_energy)
        observation.append(hunger)

        return np.array(observation, dtype=np.float32)

    def update_energy(self):
        """Decrease energy each timestep."""
        self.energy -= self.energy_cost_per_step
        self.energy = max(0, self.energy)

    def eat(self):
        """Eat prey and restore energy."""
        self.energy = min(self.max_energy, self.energy + self.energy_gain_per_kill)

    def should_die(self):
        """Check if predator should die of starvation or old age."""
        return self.energy <= 0 or self.age >= self.max_lifespan

    def can_reproduce(self):
        """Check if predator can reproduce."""
        return self.energy >= self.reproduction_threshold and self.time_since_reproduction >= self.reproduction_cooldown

    def pay_reproduction_cost(self):
        """Pay energy cost for reproduction."""
        self.energy -= self.reproduction_cost

    def update_fitness(self):
        """Update fitness based on survival and hunting success."""
        # Fitness = age (survival) + energy (hunting success)
        self.fitness = self.age + self.energy
