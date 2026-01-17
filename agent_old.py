"""
Agent classes for predators and prey.
Each agent has a neural network brain and evolves through survival.
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

    def distance_to(self, other):
        """
        Calculate distance to another agent, accounting for toroidal wrap.
        """
        dx = abs(self.pos[0] - other.pos[0])
        dy = abs(self.pos[1] - other.pos[1])

        # Account for wrapping
        if dx > self.world_width / 2:
            dx = self.world_width - dx
        if dy > self.world_height / 2:
            dy = self.world_height - dy

        return np.sqrt(dx**2 + dy**2)

    def vector_to(self, other):
        """
        Calculate vector to another agent, accounting for toroidal wrap.
        Returns the shortest vector considering wrapping.
        """
        dx = other.pos[0] - self.pos[0]
        dy = other.pos[1] - self.pos[1]

        # Account for wrapping - take shortest path
        if abs(dx) > self.world_width / 2:
            dx = dx - np.sign(dx) * self.world_width
        if abs(dy) > self.world_height / 2:
            dy = dy - np.sign(dy) * self.world_height

        return np.array([dx, dy])

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


class Prey(Agent):
    """Prey agents that try to survive."""

    def __init__(self, x, y, world_width, world_height, brain=None):
        # Prey observes: 5 nearest predators (x, y, vx, vy each) + 3 nearest prey = 5*4 + 3*4 = 32 inputs
        super().__init__(x, y, world_width, world_height, brain, input_size=32)

        self.max_speed = 3.0
        self.max_acceleration = 0.5
        self.max_lifespan = 500
        self.reproduction_age = 200
        self.time_since_reproduction = 0

    def observe(self, world):
        """
        Observe nearby predators and prey.

        Returns:
            Observation vector (numpy array)
        """
        observation = []

        # Find 5 nearest predators
        predators = world.predators
        if len(predators) > 0:
            # Calculate distances to all predators
            distances = [self.distance_to(p) for p in predators]
            # Sort by distance and take nearest 5
            nearest_indices = np.argsort(distances)[:5]

            for i in range(5):
                if i < len(nearest_indices):
                    predator = predators[nearest_indices[i]]
                    vec = self.vector_to(predator)
                    # Normalize by world size
                    vec_norm = vec / np.array([self.world_width, self.world_height])
                    vel_norm = predator.vel / predator.max_speed
                    observation.extend([vec_norm[0], vec_norm[1], vel_norm[0], vel_norm[1]])
                else:
                    # No predator, pad with zeros
                    observation.extend([0, 0, 0, 0])
        else:
            observation.extend([0] * 20)  # 5 predators * 4 values

        # Find 3 nearest other prey (for flocking)
        other_prey = [p for p in world.prey if p is not self]
        if len(other_prey) > 0:
            distances = [self.distance_to(p) for p in other_prey]
            nearest_indices = np.argsort(distances)[:3]

            for i in range(3):
                if i < len(nearest_indices):
                    prey = other_prey[nearest_indices[i]]
                    vec = self.vector_to(prey)
                    vec_norm = vec / np.array([self.world_width, self.world_height])
                    vel_norm = prey.vel / prey.max_speed
                    observation.extend([vec_norm[0], vec_norm[1], vel_norm[0], vel_norm[1]])
                else:
                    observation.extend([0, 0, 0, 0])
        else:
            observation.extend([0] * 12)  # 3 prey * 4 values

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
        self.max_lifespan = 800
        self.energy = 150
        self.max_energy = 150
        self.energy_cost_per_step = 0.3
        self.energy_gain_per_kill = 60
        self.reproduction_threshold = 120
        self.reproduction_cost = 40
        self.reproduction_cooldown = 150  # Minimum time between reproductions
        self.time_since_reproduction = 0
        self.catch_radius = 8.0  # Larger radius to help random movement succeed occasionally

    def observe(self, world):
        """
        Observe nearby prey and own hunger.

        Returns:
            Observation vector (numpy array)
        """
        observation = []

        # Find 5 nearest prey
        prey_list = world.prey
        if len(prey_list) > 0:
            distances = [self.distance_to(p) for p in prey_list]
            nearest_indices = np.argsort(distances)[:5]

            for i in range(5):
                if i < len(nearest_indices):
                    prey = prey_list[nearest_indices[i]]
                    vec = self.vector_to(prey)
                    vec_norm = vec / np.array([self.world_width, self.world_height])
                    vel_norm = prey.vel / prey.max_speed
                    observation.extend([vec_norm[0], vec_norm[1], vel_norm[0], vel_norm[1]])
                else:
                    observation.extend([0, 0, 0, 0])
        else:
            observation.extend([0] * 20)  # 5 prey * 4 values

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
