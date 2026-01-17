"""
Fully GPU-accelerated simulation.
Everything stays on GPU - positions, velocities, neural networks, physics.
Only visualization data goes to CPU.
"""

import torch
import torch.nn as nn
import numpy as np
from config import *  # Import all configuration parameters
from river import River
from utils import toroidal_distance_torch


class NeuralNetBatch(nn.Module):
    """Batched neural networks for all agents of one type."""

    def __init__(self, num_agents, input_size, hidden_size=32, output_size=2, device='cuda'):
        super().__init__()
        self.num_agents = num_agents
        self.device = device

        # One set of networks, process all agents in parallel
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        # Xavier init
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

        self.to(device)

    def forward(self, x):
        """Batch forward pass for all agents at once."""
        with torch.no_grad():
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = torch.tanh(self.fc3(x))
        return x

    def mutate_random(self, indices, mutation_rate=0.1):
        """Mutate weights for specific agents."""
        with torch.no_grad():
            for param in self.parameters():
                noise = torch.randn_like(param) * mutation_rate
                param.add_(noise)


class GPUEcosystem:
    """Fully GPU-accelerated predator-prey ecosystem."""

    def __init__(self, width=3200, height=2400, num_prey=8000, num_predators=2000, device='cuda'):
        self.width = width
        self.height = height
        self.device = device
        self.timestep = 0

        # Prey parameters (using config constants)
        self.num_prey = num_prey
        self.prey_max_speed = PREY_MAX_SPEED
        self.prey_max_accel = PREY_MAX_ACCELERATION
        self.prey_max_age = PREY_MAX_LIFESPAN
        self.prey_repro_age = PREY_REPRODUCTION_AGE

        # Predator parameters (using config constants)
        self.num_predators = num_predators
        self.pred_max_speed = PRED_MAX_SPEED
        self.pred_max_accel = PRED_MAX_ACCELERATION
        self.pred_max_age = PRED_MAX_LIFESPAN
        self.pred_max_energy = PRED_MAX_ENERGY
        self.pred_energy_cost = PRED_ENERGY_COST
        self.pred_energy_gain = PRED_ENERGY_GAIN
        self.pred_repro_threshold = PRED_REPRODUCTION_THRESHOLD
        self.pred_repro_cost = PRED_REPRODUCTION_COST
        self.pred_repro_cooldown = PRED_REPRODUCTION_COOLDOWN
        self.catch_radius = CATCH_RADIUS

        # Create river
        self.river = River(width, height)

        # Initialize prey on GPU
        self.prey_pos = torch.rand(num_prey, 2, device=device) * torch.tensor([width, height], device=device)
        self.prey_vel = torch.randn(num_prey, 2, device=device) * 0.1
        self.prey_acc = torch.zeros(num_prey, 2, device=device)
        self.prey_age = torch.zeros(num_prey, device=device)
        self.prey_repro_timer = torch.zeros(num_prey, device=device)
        self.prey_alive = torch.ones(num_prey, dtype=torch.bool, device=device)
        # Evolvable swimming ability for each prey
        self.prey_swim_speed = torch.clamp(
            torch.normal(PREY_SWIM_SPEED, 0.2, size=(num_prey,), device=device),
            min=0.1
        )

        # Individual lifespan and reproduction timing for each prey (normal distribution)
        # Prevents synchronized birth/death waves
        self.prey_max_age_individual = torch.clamp(
            torch.normal(self.prey_max_age, PREY_LIFESPAN_VARIANCE, size=(num_prey,), device=device),
            min=100
        )
        self.prey_repro_age_individual = torch.clamp(
            torch.normal(self.prey_repro_age, PREY_REPRODUCTION_VARIANCE, size=(num_prey,), device=device),
            min=50
        )

        # Initialize predators on GPU
        self.pred_pos = torch.rand(num_predators, 2, device=device) * torch.tensor([width, height], device=device)
        self.pred_vel = torch.randn(num_predators, 2, device=device) * 0.1
        self.pred_acc = torch.zeros(num_predators, 2, device=device)
        self.pred_age = torch.zeros(num_predators, device=device)
        self.pred_energy = torch.full((num_predators,), float(self.pred_max_energy), dtype=torch.float32, device=device)
        self.pred_repro_timer = torch.zeros(num_predators, device=device)
        self.pred_alive = torch.ones(num_predators, dtype=torch.bool, device=device)
        # Evolvable swimming ability for each predator
        self.pred_swim_speed = torch.clamp(
            torch.normal(PRED_SWIM_SPEED, 0.2, size=(num_predators,), device=device),
            min=0.1
        )

        # Individual lifespan and reproduction timing for each predator (normal distribution)
        # Prevents synchronized birth/death waves
        self.pred_max_age_individual = torch.clamp(
            torch.normal(self.pred_max_age, PRED_LIFESPAN_VARIANCE, size=(num_predators,), device=device),
            min=200
        )
        self.pred_repro_cooldown_individual = torch.clamp(
            torch.normal(self.pred_repro_cooldown, PRED_REPRODUCTION_VARIANCE, size=(num_predators,), device=device),
            min=50
        )

        # Per-agent neural network weights (proper neuroevolution)
        # Each agent has individual weights that are inherited and mutated at reproduction
        self.prey_arch = {'input': 32, 'hidden': [32, 32], 'output': 2}
        self.pred_arch = {'input': 21, 'hidden': [32, 32], 'output': 2}

        self.prey_weight_count = self._calc_weight_count(self.prey_arch)
        self.pred_weight_count = self._calc_weight_count(self.pred_arch)

        # Initialize random weights for each agent (small initial values)
        self.prey_weights = torch.randn(num_prey, self.prey_weight_count, device=device) * 0.1
        self.pred_weights = torch.randn(num_predators, self.pred_weight_count, device=device) * 0.1

        print(f"GPU Ecosystem initialized on {device}")
        print(f"World: {width}x{height}")
        print(f"Prey: {num_prey}, Predators: {num_predators}")

    def compute_toroidal_distances(self, pos1, pos2):
        """Compute all pairwise distances with toroidal wrapping on GPU.

        DEPRECATED: Wrapper for toroidal_distance_torch from utils.py
        """
        # Use shared utility function
        return toroidal_distance_torch(pos1, pos2, self.width, self.height)

    def _calc_weight_count(self, arch):
        """Calculate total number of weights and biases for a network architecture."""
        sizes = [arch['input']] + arch['hidden'] + [arch['output']]
        count = 0
        for i in range(len(sizes) - 1):
            count += sizes[i] * sizes[i+1]  # weights
            count += sizes[i+1]              # biases
        return count

    def _batch_forward(self, inputs, weights, arch, alive_mask):
        """
        Batched forward pass with per-agent weights.

        Args:
            inputs: (num_alive, input_size) observations for ALIVE agents only
            weights: (num_total, weight_count) per-agent weights for ALL agents
            arch: Network architecture dict
            alive_mask: (num_total,) boolean mask of alive agents

        Returns:
            outputs: (num_alive, output_size) actions for alive agents
        """
        sizes = [arch['input']] + arch['hidden'] + [arch['output']]

        # inputs already only contains alive agents
        # Extract weights for alive agents
        alive_weights = weights[alive_mask]

        if len(alive_weights) == 0:
            return torch.zeros(0, arch['output'], device=self.device)

        # Forward pass through layers
        x = inputs  # inputs are already filtered to alive agents
        offset = 0

        for i in range(len(sizes) - 1):
            in_size, out_size = sizes[i], sizes[i+1]

            # Extract weights and biases for this layer
            w_count = in_size * out_size
            b_count = out_size

            # Reshape weights: (alive_count, in_size, out_size)
            W = alive_weights[:, offset:offset+w_count].view(-1, in_size, out_size)
            offset += w_count

            # Biases: (alive_count, out_size)
            b = alive_weights[:, offset:offset+b_count]
            offset += b_count

            # Batched matrix multiply: (alive, 1, in) @ (alive, in, out) → (alive, 1, out)
            x = torch.bmm(x.unsqueeze(1), W).squeeze(1) + b

            # Activation (tanh for all layers)
            x = torch.tanh(x)

        # Return outputs for alive agents only
        return x

    def observe_prey(self):
        """Compute observations for all prey in parallel (optimized with sampling)."""
        alive_prey_pos = self.prey_pos[self.prey_alive]
        alive_prey_vel = self.prey_vel[self.prey_alive]
        alive_pred_pos = self.pred_pos[self.pred_alive]
        alive_pred_vel = self.pred_vel[self.pred_alive]

        observations = torch.zeros(self.prey_alive.sum(), 32, device=self.device)

        # Sample predators to check (not all of them)
        max_pred_sample = min(100, len(alive_pred_pos))
        if len(alive_pred_pos) > max_pred_sample:
            sample_idx = torch.randperm(len(alive_pred_pos), device=self.device)[:max_pred_sample]
            sampled_pred_pos = alive_pred_pos[sample_idx]
            sampled_pred_vel = alive_pred_vel[sample_idx]
        else:
            sampled_pred_pos = alive_pred_pos
            sampled_pred_vel = alive_pred_vel

        # Distances to sampled predators
        if len(sampled_pred_pos) > 0:
            dists_to_pred, vecs_to_pred = self.compute_toroidal_distances(alive_prey_pos, sampled_pred_pos)

            # Get 5 nearest predators for each prey
            topk = min(5, dists_to_pred.shape[1])
            _, nearest_pred_idx = torch.topk(dists_to_pred, topk, largest=False, dim=1)

            for i in range(topk):
                if i < nearest_pred_idx.shape[1]:
                    idx = nearest_pred_idx[:, i]
                    vec_norm = vecs_to_pred[torch.arange(len(alive_prey_pos)), idx] / torch.tensor([self.width, self.height], device=self.device)
                    vel_norm = sampled_pred_vel[idx] / self.pred_max_speed
                    observations[:, i*4:(i+1)*4] = torch.cat([vec_norm, vel_norm], dim=1)

        # Sample other prey to check (not all of them)
        max_prey_sample = min(50, len(alive_prey_pos))
        if len(alive_prey_pos) > max_prey_sample:
            sample_idx = torch.randperm(len(alive_prey_pos), device=self.device)[:max_prey_sample]
            sampled_prey_pos = alive_prey_pos[sample_idx]
            sampled_prey_vel = alive_prey_vel[sample_idx]

            dists_to_prey, vecs_to_prey = self.compute_toroidal_distances(alive_prey_pos, sampled_prey_pos)

            # Get 3 nearest other prey
            topk = min(3, dists_to_prey.shape[1])
            _, nearest_prey_idx = torch.topk(dists_to_prey, topk, largest=False, dim=1)

            for i in range(topk):
                if i < nearest_prey_idx.shape[1]:
                    idx = nearest_prey_idx[:, i]
                    vec_norm = vecs_to_prey[torch.arange(len(alive_prey_pos)), idx] / torch.tensor([self.width, self.height], device=self.device)
                    vel_norm = sampled_prey_vel[idx] / self.prey_max_speed
                    observations[:, 20+i*4:20+(i+1)*4] = torch.cat([vec_norm, vel_norm], dim=1)

        return observations

    def observe_predators(self):
        """Compute observations for all predators in parallel (optimized with sampling)."""
        alive_pred_pos = self.pred_pos[self.pred_alive]
        alive_prey_pos = self.prey_pos[self.prey_alive]
        alive_prey_vel = self.prey_vel[self.prey_alive]
        alive_pred_energy = self.pred_energy[self.pred_alive]

        observations = torch.zeros(self.pred_alive.sum(), 21, device=self.device)

        # Sample prey to check (not all of them)
        max_prey_sample = min(200, len(alive_prey_pos))
        if len(alive_prey_pos) > max_prey_sample:
            sample_idx = torch.randperm(len(alive_prey_pos), device=self.device)[:max_prey_sample]
            sampled_prey_pos = alive_prey_pos[sample_idx]
            sampled_prey_vel = alive_prey_vel[sample_idx]
        else:
            sampled_prey_pos = alive_prey_pos
            sampled_prey_vel = alive_prey_vel

        # Distances to sampled prey
        if len(sampled_prey_pos) > 0:
            dists_to_prey, vecs_to_prey = self.compute_toroidal_distances(alive_pred_pos, sampled_prey_pos)

            # Get 5 nearest prey for each predator
            topk = min(5, dists_to_prey.shape[1])
            _, nearest_prey_idx = torch.topk(dists_to_prey, topk, largest=False, dim=1)

            for i in range(topk):
                if i < nearest_prey_idx.shape[1]:
                    idx = nearest_prey_idx[:, i]
                    vec_norm = vecs_to_prey[torch.arange(len(alive_pred_pos)), idx] / torch.tensor([self.width, self.height], device=self.device)
                    vel_norm = sampled_prey_vel[idx] / self.prey_max_speed
                    observations[:, i*4:(i+1)*4] = torch.cat([vec_norm, vel_norm], dim=1)

        # Hunger
        hunger = 1.0 - (alive_pred_energy / self.pred_max_energy)
        observations[:, 20] = hunger

        return observations

    def get_island_modifiers(self, positions, agent_type):
        """
        Get island behavior modifiers for a batch of positions.

        Args:
            positions: Tensor of positions (N, 2) on GPU
            agent_type: 'prey' or 'predator'

        Returns:
            Dictionary with modifier tensors on GPU
        """
        if not self.river.enabled or not self.river.split:
            return None

        # Transfer positions to CPU for island checking
        positions_np = positions.cpu().numpy()

        # Check which agents are on island
        on_island = np.array([
            self.river.is_on_island(pos[0], pos[1])
            for pos in positions_np
        ], dtype=bool)

        if not np.any(on_island):
            return None

        # Get modifiers from river config
        if agent_type == 'prey':
            speed_mult = ISLAND_PREY_SPEED_MULTIPLIER
            repro_mult = ISLAND_PREY_REPRODUCTION_MULTIPLIER

            # Create modifier tensors (1.0 for not on island, multiplier for on island)
            speed_modifiers = torch.ones(len(positions), device=self.device)
            speed_modifiers[on_island] = speed_mult

            repro_modifiers = torch.ones(len(positions), device=self.device)
            repro_modifiers[on_island] = repro_mult

            return {
                'speed': speed_modifiers,
                'reproduction': repro_modifiers,
                'on_island': torch.tensor(on_island, device=self.device)
            }

        elif agent_type == 'predator':
            speed_mult = ISLAND_PRED_SPEED_MULTIPLIER
            hunger_mult = ISLAND_PRED_HUNGER_MULTIPLIER
            repro_mult = ISLAND_PRED_REPRODUCTION_MULTIPLIER

            # Create modifier tensors
            speed_modifiers = torch.ones(len(positions), device=self.device)
            speed_modifiers[on_island] = speed_mult

            hunger_modifiers = torch.ones(len(positions), device=self.device)
            hunger_modifiers[on_island] = hunger_mult

            repro_modifiers = torch.ones(len(positions), device=self.device)
            repro_modifiers[on_island] = repro_mult

            return {
                'speed': speed_modifiers,
                'hunger': hunger_modifiers,
                'reproduction': repro_modifiers,
                'on_island': torch.tensor(on_island, device=self.device)
            }

        return None

    def step(self, mutation_rate=0.1):
        """Single simulation step - fully on GPU."""
        self.timestep += 1

        # 1. Observations and actions
        prey_obs = self.observe_prey()
        # Use per-agent weights for forward pass (returns actions for alive agents only)
        prey_actions = self._batch_forward(prey_obs, self.prey_weights, self.prey_arch, self.prey_alive)
        self.prey_acc[self.prey_alive] = prey_actions * self.prey_max_accel

        pred_obs = self.observe_predators()
        # Use per-agent weights for forward pass (returns actions for alive agents only)
        pred_actions = self._batch_forward(pred_obs, self.pred_weights, self.pred_arch, self.pred_alive)
        self.pred_acc[self.pred_alive] = pred_actions * self.pred_max_accel

        # 2. Physics update (vectorized)
        self.prey_vel[self.prey_alive] += self.prey_acc[self.prey_alive]

        # Apply island speed modifiers to prey
        prey_island_mods = self.get_island_modifiers(self.prey_pos[self.prey_alive], 'prey')
        if prey_island_mods is not None:
            prey_speed_limit = self.prey_max_speed * prey_island_mods['speed']
        else:
            prey_speed_limit = self.prey_max_speed

        speed = torch.norm(self.prey_vel[self.prey_alive], dim=1, keepdim=True)
        if prey_island_mods is not None:
            # Apply different speed limits based on island position
            speed_limit = prey_speed_limit.unsqueeze(1)
            self.prey_vel[self.prey_alive] = torch.where(
                speed > speed_limit,
                self.prey_vel[self.prey_alive] / speed * speed_limit,
                self.prey_vel[self.prey_alive]
            )
        else:
            self.prey_vel[self.prey_alive] = torch.where(
                speed > self.prey_max_speed,
                self.prey_vel[self.prey_alive] / speed * self.prey_max_speed,
                self.prey_vel[self.prey_alive]
            )
        self.prey_pos[self.prey_alive] = (self.prey_pos[self.prey_alive] + self.prey_vel[self.prey_alive]) % torch.tensor([self.width, self.height], device=self.device)

        # Apply river flow to prey
        if self.river.enabled:
            alive_prey_pos_np = self.prey_pos[self.prey_alive].cpu().numpy()
            flows = self.river.get_flow_at_batch(alive_prey_pos_np)
            if len(flows) > 0:
                # Better swimmers resist current more
                alive_swim_speeds = self.prey_swim_speed[self.prey_alive]
                flow_factors = torch.clamp(1.0 - alive_swim_speeds / 5.0, min=0.0, max=1.0)
                flow_tensor = torch.tensor(flows, device=self.device, dtype=torch.float32)
                self.prey_vel[self.prey_alive] += flow_tensor * flow_factors.unsqueeze(1)

        self.prey_age[self.prey_alive] += 1
        self.prey_repro_timer[self.prey_alive] += 1

        self.pred_vel[self.pred_alive] += self.pred_acc[self.pred_alive]

        # Apply island speed modifiers to predators
        pred_island_mods = self.get_island_modifiers(self.pred_pos[self.pred_alive], 'predator')
        if pred_island_mods is not None:
            pred_speed_limit = self.pred_max_speed * pred_island_mods['speed']
        else:
            pred_speed_limit = self.pred_max_speed

        speed = torch.norm(self.pred_vel[self.pred_alive], dim=1, keepdim=True)
        if pred_island_mods is not None:
            # Apply different speed limits based on island position
            speed_limit = pred_speed_limit.unsqueeze(1)
            self.pred_vel[self.pred_alive] = torch.where(
                speed > speed_limit,
                self.pred_vel[self.pred_alive] / speed * speed_limit,
                self.pred_vel[self.pred_alive]
            )
        else:
            self.pred_vel[self.pred_alive] = torch.where(
                speed > self.pred_max_speed,
                self.pred_vel[self.pred_alive] / speed * self.pred_max_speed,
                self.pred_vel[self.pred_alive]
            )
        self.pred_pos[self.pred_alive] = (self.pred_pos[self.pred_alive] + self.pred_vel[self.pred_alive]) % torch.tensor([self.width, self.height], device=self.device)

        # Apply river flow to predators
        if self.river.enabled:
            alive_pred_pos_np = self.pred_pos[self.pred_alive].cpu().numpy()
            flows = self.river.get_flow_at_batch(alive_pred_pos_np)
            if len(flows) > 0:
                # Better swimmers resist current more
                alive_swim_speeds = self.pred_swim_speed[self.pred_alive]
                flow_factors = torch.clamp(1.0 - alive_swim_speeds / 5.0, min=0.0, max=1.0)
                flow_tensor = torch.tensor(flows, device=self.device, dtype=torch.float32)
                self.pred_vel[self.pred_alive] += flow_tensor * flow_factors.unsqueeze(1)

        self.pred_age[self.pred_alive] += 1

        # Apply island hunger modifiers to energy cost
        if pred_island_mods is not None:
            energy_cost = self.pred_energy_cost * pred_island_mods['hunger']
            self.pred_energy[self.pred_alive] -= energy_cost
        else:
            self.pred_energy[self.pred_alive] -= self.pred_energy_cost

        self.pred_repro_timer[self.pred_alive] += 1

        # 3. Collision detection (GPU)
        if self.prey_alive.sum() > 0 and self.pred_alive.sum() > 0:
            alive_prey_pos = self.prey_pos[self.prey_alive]
            alive_pred_pos = self.pred_pos[self.pred_alive]

            dists, _ = self.compute_toroidal_distances(alive_pred_pos, alive_prey_pos)
            catches = dists < self.catch_radius

            # Each predator catches at most one prey
            prey_caught = torch.zeros(self.prey_alive.sum(), dtype=torch.bool, device=self.device)
            for pred_idx in range(catches.shape[0]):
                caught_prey = torch.where(catches[pred_idx] & ~prey_caught)[0]
                if len(caught_prey) > 0:
                    # Catch nearest
                    nearest = caught_prey[dists[pred_idx, caught_prey].argmin()]
                    prey_caught[nearest] = True
                    # Predator eats
                    alive_pred_idx = torch.where(self.pred_alive)[0][pred_idx]
                    self.pred_energy[alive_pred_idx] = torch.clamp(
                        self.pred_energy[alive_pred_idx] + float(self.pred_energy_gain),
                        max=float(self.pred_max_energy)
                    )

            # Remove caught prey
            alive_indices = torch.where(self.prey_alive)[0]
            self.prey_alive[alive_indices[prey_caught]] = False

        # 4. Deaths (using individual age limits - prevents synchronized deaths)
        self.prey_alive &= self.prey_age < self.prey_max_age_individual
        self.pred_alive &= (self.pred_age < self.pred_max_age_individual) & (self.pred_energy > 0)

        # 5. Reproduction (using individual timing - prevents synchronized births)
        # Apply island reproduction modifiers
        # Get modifiers for all alive agents
        prey_repro_mods = self.get_island_modifiers(self.prey_pos[self.prey_alive], 'prey')
        pred_repro_mods = self.get_island_modifiers(self.pred_pos[self.pred_alive], 'predator')

        # For prey: check if repro_timer >= (repro_age * multiplier)
        if prey_repro_mods is not None:
            prey_repro_threshold = self.prey_repro_age_individual[self.prey_alive] * prey_repro_mods['reproduction']
            can_repro_prey_subset = self.prey_repro_timer[self.prey_alive] >= prey_repro_threshold
            can_repro_prey = torch.zeros_like(self.prey_alive)
            can_repro_prey[self.prey_alive] = can_repro_prey_subset
        else:
            can_repro_prey = self.prey_alive & (self.prey_repro_timer >= self.prey_repro_age_individual)

        # For predators: check if repro_timer >= (repro_cooldown * multiplier)
        if pred_repro_mods is not None:
            pred_repro_cooldown_threshold = self.pred_repro_cooldown_individual[self.pred_alive] * pred_repro_mods['reproduction']
            can_repro_pred_subset = (self.pred_energy[self.pred_alive] >= self.pred_repro_threshold) & \
                                   (self.pred_repro_timer[self.pred_alive] >= pred_repro_cooldown_threshold)
            can_repro_pred = torch.zeros_like(self.pred_alive)
            can_repro_pred[self.pred_alive] = can_repro_pred_subset
        else:
            can_repro_pred = self.pred_alive & (self.pred_energy >= self.pred_repro_threshold) & (self.pred_repro_timer >= self.pred_repro_cooldown_individual)

        # Respawn dead agents as offspring of survivors
        dead_prey_idx = torch.where(~self.prey_alive)[0]
        alive_prey_idx = torch.where(can_repro_prey)[0]
        if len(dead_prey_idx) > 0 and len(alive_prey_idx) > 0:
            # Limit reproduction per step to prevent blob spawning
            num_to_spawn = min(len(dead_prey_idx), len(alive_prey_idx) * 2)  # Max 2 offspring per parent
            dead_prey_idx = dead_prey_idx[:num_to_spawn]

            # Random parents
            parents = alive_prey_idx[torch.randint(0, len(alive_prey_idx), (len(dead_prey_idx),), device=self.device)]
            # Variable spawn distance: some near (20-50), some far (50-150)
            spawn_distance = torch.rand(len(dead_prey_idx), device=self.device) * 130 + 20  # 20 to 150 pixels
            spawn_angle = torch.rand(len(dead_prey_idx), device=self.device) * 2 * 3.14159
            offset_x = spawn_distance * torch.cos(spawn_angle)
            offset_y = spawn_distance * torch.sin(spawn_angle)
            self.prey_pos[dead_prey_idx, 0] = self.prey_pos[parents, 0] + offset_x
            self.prey_pos[dead_prey_idx, 1] = self.prey_pos[parents, 1] + offset_y
            self.prey_pos[dead_prey_idx] %= torch.tensor([self.width, self.height], device=self.device)
            self.prey_vel[dead_prey_idx] = torch.randn(len(dead_prey_idx), 2, device=self.device) * 0.1
            self.prey_age[dead_prey_idx] = 0
            self.prey_repro_timer[dead_prey_idx] = 0
            self.prey_alive[dead_prey_idx] = True
            # Give offspring randomized individual lifespans and reproduction ages
            self.prey_max_age_individual[dead_prey_idx] = torch.clamp(
                torch.normal(self.prey_max_age, PREY_LIFESPAN_VARIANCE, size=(len(dead_prey_idx),), device=self.device),
                min=100
            )
            self.prey_repro_age_individual[dead_prey_idx] = torch.clamp(
                torch.normal(self.prey_repro_age, PREY_REPRODUCTION_VARIANCE, size=(len(dead_prey_idx),), device=self.device),
                min=50
            )
            # === CRITICAL: Inherit brain weights from parents ===
            self.prey_weights[dead_prey_idx] = self.prey_weights[parents].clone()

            # === CRITICAL: Mutate offspring brain weights ===
            mutation = torch.randn_like(self.prey_weights[dead_prey_idx]) * mutation_rate
            self.prey_weights[dead_prey_idx] += mutation

            # Inherit and mutate swim speed from parents
            parent_swim_speeds = self.prey_swim_speed[parents]
            self.prey_swim_speed[dead_prey_idx] = torch.clamp(
                parent_swim_speeds + torch.randn(len(dead_prey_idx), device=self.device) * mutation_rate * 2.0,
                min=0.1
            )
            self.prey_repro_timer[parents] = 0  # Reset parent timers

        dead_pred_idx = torch.where(~self.pred_alive)[0]
        alive_pred_idx = torch.where(can_repro_pred)[0]
        if len(dead_pred_idx) > 0 and len(alive_pred_idx) > 0:
            # Limit reproduction per step to prevent blob spawning
            num_to_spawn = min(len(dead_pred_idx), len(alive_pred_idx) * 2)  # Max 2 offspring per parent
            dead_pred_idx = dead_pred_idx[:num_to_spawn]

            parents = alive_pred_idx[torch.randint(0, len(alive_pred_idx), (len(dead_pred_idx),), device=self.device)]
            # Variable spawn distance: some near (20-50), some far (50-150)
            spawn_distance = torch.rand(len(dead_pred_idx), device=self.device) * 130 + 20  # 20 to 150 pixels
            spawn_angle = torch.rand(len(dead_pred_idx), device=self.device) * 2 * 3.14159
            offset_x = spawn_distance * torch.cos(spawn_angle)
            offset_y = spawn_distance * torch.sin(spawn_angle)
            self.pred_pos[dead_pred_idx, 0] = self.pred_pos[parents, 0] + offset_x
            self.pred_pos[dead_pred_idx, 1] = self.pred_pos[parents, 1] + offset_y
            self.pred_pos[dead_pred_idx] %= torch.tensor([self.width, self.height], device=self.device)
            self.pred_vel[dead_pred_idx] = torch.randn(len(dead_pred_idx), 2, device=self.device) * 0.1
            self.pred_age[dead_pred_idx] = 0
            self.pred_energy[dead_pred_idx] = float(self.pred_max_energy)
            self.pred_repro_timer[dead_pred_idx] = 0
            self.pred_alive[dead_pred_idx] = True
            # Give offspring randomized individual lifespans and reproduction cooldowns
            self.pred_max_age_individual[dead_pred_idx] = torch.clamp(
                torch.normal(self.pred_max_age, PRED_LIFESPAN_VARIANCE, size=(len(dead_pred_idx),), device=self.device),
                min=200
            )
            self.pred_repro_cooldown_individual[dead_pred_idx] = torch.clamp(
                torch.normal(self.pred_repro_cooldown, PRED_REPRODUCTION_VARIANCE, size=(len(dead_pred_idx),), device=self.device),
                min=50
            )
            # === CRITICAL: Inherit brain weights from parents ===
            self.pred_weights[dead_pred_idx] = self.pred_weights[parents].clone()

            # === CRITICAL: Mutate offspring brain weights ===
            mutation = torch.randn_like(self.pred_weights[dead_pred_idx]) * mutation_rate
            self.pred_weights[dead_pred_idx] += mutation

            # Inherit and mutate swim speed from parents
            parent_swim_speeds = self.pred_swim_speed[parents]
            self.pred_swim_speed[dead_pred_idx] = torch.clamp(
                parent_swim_speeds + torch.randn(len(dead_pred_idx), device=self.device) * mutation_rate * 2.0,
                min=0.1
            )
            self.pred_energy[parents] -= self.pred_repro_cost
            self.pred_repro_timer[parents] = 0

        # Unified extinction prevention
        self._handle_extinction_prevention()

        # === REMOVED: Global mutation code (incorrect neuroevolution) ===
        # Mutation now happens ONLY at reproduction with parent weight inheritance

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

        prey_alive_count = self.prey_alive.sum().item()
        pred_alive_count = self.pred_alive.sum().item()

        # Handle prey
        if prey_alive_count < 1:
            # Emergency: complete extinction
            print(f"\n⚠️  PREY EXTINCTION at timestep {self.timestep}! Respawning {emergency_respawn_count} random prey...")
            dead_prey = torch.where(~self.prey_alive)[0][:emergency_respawn_count]
            if len(dead_prey) > 0:
                self._respawn_prey(dead_prey)
        elif prey_alive_count < min_prey:
            # Below minimum threshold: gradually repopulate
            spawn_count = min(min_prey - prey_alive_count, emergency_respawn_count)  # Cap spawn rate
            dead_prey = torch.where(~self.prey_alive)[0][:spawn_count]
            if len(dead_prey) > 0:
                self._respawn_prey(dead_prey)

        # Handle predators
        if pred_alive_count < 1:
            # Emergency: complete extinction
            print(f"\n⚠️  PREDATOR EXTINCTION at timestep {self.timestep}! Respawning {emergency_respawn_count} random predators...")
            dead_pred = torch.where(~self.pred_alive)[0][:emergency_respawn_count]
            if len(dead_pred) > 0:
                self._respawn_predators(dead_pred)
        elif pred_alive_count < min_predators:
            # Below minimum threshold: gradually repopulate
            spawn_count = min(min_predators - pred_alive_count, emergency_respawn_count)  # Cap spawn rate
            dead_pred = torch.where(~self.pred_alive)[0][:spawn_count]
            if len(dead_pred) > 0:
                self._respawn_predators(dead_pred)

    def _respawn_prey(self, indices):
        """Respawn prey at given indices with random parameters."""
        # Random positions across the map
        self.prey_pos[indices] = torch.rand(len(indices), 2, device=self.device) * torch.tensor([self.width, self.height], device=self.device)
        self.prey_vel[indices] = torch.randn(len(indices), 2, device=self.device) * 0.1
        self.prey_age[indices] = 0
        self.prey_repro_timer[indices] = 0
        self.prey_alive[indices] = True
        # Give random individual parameters
        self.prey_max_age_individual[indices] = torch.clamp(
            torch.normal(self.prey_max_age, PREY_LIFESPAN_VARIANCE, size=(len(indices),), device=self.device),
            min=100
        )
        self.prey_repro_age_individual[indices] = torch.clamp(
            torch.normal(self.prey_repro_age, PREY_REPRODUCTION_VARIANCE, size=(len(indices),), device=self.device),
            min=50
        )
        # Give random swim speeds
        self.prey_swim_speed[indices] = torch.clamp(
            torch.normal(PREY_SWIM_SPEED, 0.2, size=(len(indices),), device=self.device),
            min=0.1
        )
        # Initialize random brain weights
        self.prey_weights[indices] = torch.randn(len(indices), self.prey_weight_count, device=self.device) * 0.1

    def _respawn_predators(self, indices):
        """Respawn predators at given indices with random parameters."""
        # Random positions across the map
        self.pred_pos[indices] = torch.rand(len(indices), 2, device=self.device) * torch.tensor([self.width, self.height], device=self.device)
        self.pred_vel[indices] = torch.randn(len(indices), 2, device=self.device) * 0.1
        self.pred_age[indices] = 0
        self.pred_energy[indices] = float(self.pred_max_energy)
        self.pred_repro_timer[indices] = 0
        self.pred_alive[indices] = True
        # Give random individual parameters
        self.pred_max_age_individual[indices] = torch.clamp(
            torch.normal(self.pred_max_age, PRED_LIFESPAN_VARIANCE, size=(len(indices),), device=self.device),
            min=200
        )
        self.pred_repro_cooldown_individual[indices] = torch.clamp(
            torch.normal(self.pred_repro_cooldown, PRED_REPRODUCTION_VARIANCE, size=(len(indices),), device=self.device),
            min=50
        )
        # Give random swim speeds
        self.pred_swim_speed[indices] = torch.clamp(
            torch.normal(PRED_SWIM_SPEED, 0.2, size=(len(indices),), device=self.device),
            min=0.1
        )
        # Initialize random brain weights
        self.pred_weights[indices] = torch.randn(len(indices), self.pred_weight_count, device=self.device) * 0.1

    def get_state_cpu(self):
        """Transfer current state to CPU for visualization and analysis."""
        state = {
            'prey_pos': self.prey_pos[self.prey_alive].cpu().numpy(),
            'pred_pos': self.pred_pos[self.pred_alive].cpu().numpy(),
            'prey_count': self.prey_alive.sum().item(),
            'pred_count': self.pred_alive.sum().item(),
            'prey_avg_age': self.prey_age[self.prey_alive].mean().item() if self.prey_alive.sum() > 0 else 0,
            'pred_avg_age': self.pred_age[self.pred_alive].mean().item() if self.pred_alive.sum() > 0 else 0,
            'pred_avg_energy': self.pred_energy[self.pred_alive].mean().item() if self.pred_alive.sum() > 0 else 0,
        }

        # Add swim speed statistics for evolutionary tracking
        if self.prey_alive.sum() > 0:
            prey_swim = self.prey_swim_speed[self.prey_alive]
            state['prey_avg_swim'] = prey_swim.mean().item()
            state['prey_std_swim'] = prey_swim.std().item()
            state['prey_min_swim'] = prey_swim.min().item()
            state['prey_max_swim'] = prey_swim.max().item()

            # Classify prey as land or water specialists
            # Check which prey are in the river vs on island vs on land
            prey_pos_np = self.prey_pos[self.prey_alive].cpu().numpy()
            prey_in_river = 0
            prey_on_island = 0
            for pos in prey_pos_np:
                if self.river.is_on_island(pos[0], pos[1]):
                    prey_on_island += 1
                elif self.river.is_in_river(pos[0], pos[1]):
                    prey_in_river += 1
            state['prey_in_river_pct'] = (prey_in_river / len(prey_pos_np)) * 100 if len(prey_pos_np) > 0 else 0
            state['prey_on_island_pct'] = (prey_on_island / len(prey_pos_np)) * 100 if len(prey_pos_np) > 0 else 0
        else:
            state['prey_avg_swim'] = 0
            state['prey_std_swim'] = 0
            state['prey_min_swim'] = 0
            state['prey_max_swim'] = 0
            state['prey_in_river_pct'] = 0
            state['prey_on_island_pct'] = 0

        if self.pred_alive.sum() > 0:
            pred_swim = self.pred_swim_speed[self.pred_alive]
            state['pred_avg_swim'] = pred_swim.mean().item()
            state['pred_std_swim'] = pred_swim.std().item()
            state['pred_min_swim'] = pred_swim.min().item()
            state['pred_max_swim'] = pred_swim.max().item()

            # Check which predators are in the river vs on island vs on land
            pred_pos_np = self.pred_pos[self.pred_alive].cpu().numpy()
            pred_in_river = 0
            pred_on_island = 0
            for pos in pred_pos_np:
                if self.river.is_on_island(pos[0], pos[1]):
                    pred_on_island += 1
                elif self.river.is_in_river(pos[0], pos[1]):
                    pred_in_river += 1
            state['pred_in_river_pct'] = (pred_in_river / len(pred_pos_np)) * 100 if len(pred_pos_np) > 0 else 0
            state['pred_on_island_pct'] = (pred_on_island / len(pred_pos_np)) * 100 if len(pred_pos_np) > 0 else 0
        else:
            state['pred_avg_swim'] = 0
            state['pred_std_swim'] = 0
            state['pred_min_swim'] = 0
            state['pred_max_swim'] = 0
            state['pred_in_river_pct'] = 0
            state['pred_on_island_pct'] = 0

        return state
