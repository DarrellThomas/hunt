"""
Optimized agent classes with vectorized distance calculations.
"""

import numpy as np
from brain import Brain
from config import *  # Import all configuration parameters
from utils import toroidal_distance_numpy


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
        Create offspring with mutated brain and traits.

        Args:
            mutation_rate: Amount of mutation to apply

        Returns:
            New agent (offspring)
        """
        # Spawn near parent with variable distance to prevent blobs
        # Random distance: 20 to 150 pixels
        spawn_distance = np.random.uniform(20, 150)
        spawn_angle = np.random.uniform(0, 2 * np.pi)
        offset_x = spawn_distance * np.cos(spawn_angle)
        offset_y = spawn_distance * np.sin(spawn_angle)
        child_x = (self.pos[0] + offset_x) % self.world_width
        child_y = (self.pos[1] + offset_y) % self.world_height

        # Inherit and mutate swim_speed (if agent has it)
        if hasattr(self, 'swim_speed'):
            # Mutate swim speed slightly
            child_swim_speed = max(0.1, self.swim_speed + np.random.randn() * mutation_rate * 2.0)
        else:
            child_swim_speed = None

        # Create child with copied brain and mutated swim_speed
        child = self.__class__(child_x, child_y, self.world_width, self.world_height,
                              brain=self.brain, swim_speed=child_swim_speed)

        # Mutate child's brain
        child.brain.mutate(mutation_rate)

        return child


def vectorized_distances(pos, other_positions, world_width, world_height):
    """
    Vectorized distance calculation with toroidal wrapping.

    DEPRECATED: Wrapper for toroidal_distance_numpy from utils.py

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

    # Use shared utility function
    return toroidal_distance_numpy(pos, other_positions, world_width, world_height)


class Prey(Agent):
    """Prey agents that try to survive."""

    def __init__(self, x, y, world_width, world_height, brain=None, swim_speed=None):
        # Prey observes: 5 nearest predators (x, y, vx, vy each) + 3 nearest prey = 5*4 + 3*4 = 32 inputs
        super().__init__(x, y, world_width, world_height, brain, input_size=32)

        self.max_speed = PREY_MAX_SPEED
        self.max_acceleration = PREY_MAX_ACCELERATION

        # Evolvable swimming ability - resistance to current
        if swim_speed is None:
            self.swim_speed = max(0.1, PREY_SWIM_SPEED + np.random.randn() * 0.2)
        else:
            self.swim_speed = swim_speed

        # Add natural variation to lifespan and reproduction timing
        # Normal distribution prevents synchronized birth/death waves
        self.max_lifespan = max(100, int(np.random.normal(PREY_MAX_LIFESPAN, PREY_LIFESPAN_VARIANCE)))
        self.reproduction_age = max(50, int(np.random.normal(PREY_REPRODUCTION_AGE, PREY_REPRODUCTION_VARIANCE)))
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

    def __init__(self, x, y, world_width, world_height, brain=None, swim_speed=None):
        # Predator observes: 5 nearest prey (x, y, vx, vy each) + own hunger = 5*4 + 1 = 21 inputs
        super().__init__(x, y, world_width, world_height, brain, input_size=21)

        self.max_speed = PRED_MAX_SPEED
        self.max_acceleration = PRED_MAX_ACCELERATION

        # Evolvable swimming ability - resistance to current
        if swim_speed is None:
            self.swim_speed = max(0.1, PRED_SWIM_SPEED + np.random.randn() * 0.2)
        else:
            self.swim_speed = swim_speed

        # Add natural variation to lifespan and reproduction timing
        # Normal distribution prevents synchronized birth/death waves
        self.max_lifespan = max(200, int(np.random.normal(PRED_MAX_LIFESPAN, PRED_LIFESPAN_VARIANCE)))
        self.reproduction_cooldown = max(50, int(np.random.normal(PRED_REPRODUCTION_COOLDOWN, PRED_REPRODUCTION_VARIANCE)))

        self.energy = PRED_MAX_ENERGY
        self.max_energy = PRED_MAX_ENERGY
        self.energy_cost_per_step = PRED_ENERGY_COST
        self.energy_gain_per_kill = PRED_ENERGY_GAIN
        self.reproduction_threshold = PRED_REPRODUCTION_THRESHOLD
        self.reproduction_cost = PRED_REPRODUCTION_COST
        self.time_since_reproduction = 0
        self.catch_radius = CATCH_RADIUS

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
