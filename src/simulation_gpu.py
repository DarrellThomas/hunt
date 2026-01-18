"""
Fully GPU-accelerated simulation.
Everything stays on GPU - positions, velocities, neural networks, physics.
Only visualization data goes to CPU.
"""

import torch
import torch.nn as nn
import numpy as np
from config import *  # Import all configuration parameters
from river_gpu import RiverGPU
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

    def __init__(self, width=3200, height=2400, num_prey=8000, num_predators=2000,
                 max_prey_capacity=None, max_pred_capacity=None, device='cuda'):
        """
        Initialize GPU ecosystem with population capacity for growth.

        Args:
            width, height: World dimensions
            num_prey, num_predators: Initial alive population
            max_prey_capacity: Maximum prey slots (default: 3x num_prey)
            max_pred_capacity: Maximum predator slots (default: 3x num_predators)
            device: GPU device
        """
        self.width = width
        self.height = height
        self.device = device
        self.timestep = 0

        # Set maximum capacities from config (default from POPULATION_CAPACITY_MULTIPLIER)
        if max_prey_capacity is None:
            max_prey_capacity = num_prey * POPULATION_CAPACITY_MULTIPLIER
        if max_pred_capacity is None:
            max_pred_capacity = num_predators * POPULATION_CAPACITY_MULTIPLIER

        self.max_prey_capacity = max_prey_capacity
        self.max_pred_capacity = max_pred_capacity
        self.initial_prey = num_prey
        self.initial_predators = num_predators

        # Prey parameters (using config constants)
        self.prey_max_speed = PREY_MAX_SPEED
        self.prey_max_accel = PREY_MAX_ACCELERATION
        self.prey_max_age = PREY_MAX_LIFESPAN
        self.prey_repro_age = PREY_REPRODUCTION_AGE

        # Predator parameters (using config constants)
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

        # Create GPU-resident river
        self.river = RiverGPU(width, height, device=device)

        # Allocate prey tensors at MAX CAPACITY (not initial population)
        self.prey_pos = torch.rand(max_prey_capacity, 2, device=device) * torch.tensor([width, height], device=device)
        self.prey_vel = torch.randn(max_prey_capacity, 2, device=device) * 0.1
        self.prey_acc = torch.zeros(max_prey_capacity, 2, device=device)
        self.prey_age = torch.zeros(max_prey_capacity, device=device)
        self.prey_repro_timer = torch.zeros(max_prey_capacity, device=device)
        # Only initial num_prey are alive, rest are dead (available for reproduction)
        self.prey_alive = torch.zeros(max_prey_capacity, dtype=torch.bool, device=device)
        self.prey_alive[:num_prey] = True

        # Evolvable swimming ability for each prey
        self.prey_swim_speed = torch.clamp(
            torch.normal(PREY_SWIM_SPEED, 0.2, size=(max_prey_capacity,), device=device),
            min=0.1
        )

        # Individual lifespan and reproduction timing for each prey (normal distribution)
        self.prey_max_age_individual = torch.clamp(
            torch.normal(self.prey_max_age, PREY_LIFESPAN_VARIANCE, size=(max_prey_capacity,), device=device),
            min=100
        )
        self.prey_repro_age_individual = torch.clamp(
            torch.normal(self.prey_repro_age, PREY_REPRODUCTION_VARIANCE, size=(max_prey_capacity,), device=device),
            min=50
        )

        # Allocate predator tensors at MAX CAPACITY (not initial population)
        self.pred_pos = torch.rand(max_pred_capacity, 2, device=device) * torch.tensor([width, height], device=device)
        self.pred_vel = torch.randn(max_pred_capacity, 2, device=device) * 0.1
        self.pred_acc = torch.zeros(max_pred_capacity, 2, device=device)
        self.pred_age = torch.zeros(max_pred_capacity, device=device)
        self.pred_energy = torch.full((max_pred_capacity,), float(self.pred_max_energy), dtype=torch.float32, device=device)
        self.pred_repro_timer = torch.zeros(max_pred_capacity, device=device)
        # Only initial num_predators are alive, rest are dead (available for reproduction)
        self.pred_alive = torch.zeros(max_pred_capacity, dtype=torch.bool, device=device)
        self.pred_alive[:num_predators] = True

        # Evolvable swimming ability for each predator
        self.pred_swim_speed = torch.clamp(
            torch.normal(PRED_SWIM_SPEED, 0.2, size=(max_pred_capacity,), device=device),
            min=0.1
        )

        # Individual lifespan and reproduction timing for each predator (normal distribution)
        self.pred_max_age_individual = torch.clamp(
            torch.normal(self.pred_max_age, PRED_LIFESPAN_VARIANCE, size=(max_pred_capacity,), device=device),
            min=200
        )
        self.pred_repro_cooldown_individual = torch.clamp(
            torch.normal(self.pred_repro_cooldown, PRED_REPRODUCTION_VARIANCE, size=(max_pred_capacity,), device=device),
            min=50
        )

        # Per-agent neural network weights (proper neuroevolution)
        # Each agent has individual weights that are inherited and mutated at reproduction
        self.prey_arch = {'input': 32, 'hidden': [32, 32], 'output': 2}
        self.pred_arch = {'input': 21, 'hidden': [32, 32], 'output': 2}

        self.prey_weight_count = self._calc_weight_count(self.prey_arch)
        self.pred_weight_count = self._calc_weight_count(self.pred_arch)

        # Initialize random weights at MAX CAPACITY
        self.prey_weights = torch.randn(max_prey_capacity, self.prey_weight_count, device=device) * 0.1
        self.pred_weights = torch.randn(max_pred_capacity, self.pred_weight_count, device=device) * 0.1

        # Statistics tracking for evolution analysis
        self.stats = {
            'timesteps': [],
            'prey_count': [],
            'pred_count': [],
            'prey_avg_age': [],
            'pred_avg_age': [],
            'pred_avg_energy': [],
            'prey_avg_swim': [],
            'prey_std_swim': [],
            'pred_avg_swim': [],
            'pred_std_swim': [],
            'prey_in_river_pct': [],
            'pred_in_river_pct': [],
        }

        # Extinction flag for stopping simulation
        self.extinct = False
        self.extinction_message = ""

        # Metadata for analysis and run tracking
        import datetime
        import platform
        self.metadata = {
            'run_title': 'Untitled Run',  # Can be set via CLI
            'start_time': datetime.datetime.now().isoformat(),
            'world_width': width,
            'world_height': height,
            'initial_prey': num_prey,
            'initial_predators': num_predators,
            'max_prey_capacity': max_prey_capacity,
            'max_pred_capacity': max_pred_capacity,
            'device': str(device),
            'platform': platform.system(),
            'python_version': platform.python_version(),
            # Config snapshot (key parameters)
            'config_friction': FRICTION,
            'config_prey_max_speed': PREY_MAX_SPEED,
            'config_pred_max_speed': PRED_MAX_SPEED,
            'config_river_enabled': RIVER_ENABLED,
            'config_river_flow_speed': RIVER_FLOW_SPEED if RIVER_ENABLED else 0,
            'config_extinction_threshold': EXTINCTION_THRESHOLD,
        }

        print(f"GPU Ecosystem initialized on {device}")
        print(f"World: {width}x{height}")
        print(f"Initial Population: {num_prey:,} prey, {num_predators:,} predators")
        print(f"Maximum Capacity: {max_prey_capacity:,} prey slots, {max_pred_capacity:,} predator slots")
        print(f"Growth headroom: {max_prey_capacity/num_prey:.1f}x prey, {max_pred_capacity/num_predators:.1f}x predators")

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

        # Check which agents are on island (GPU-resident)
        on_island = self.river.is_on_island_batch_gpu(positions)

        if not torch.any(on_island):
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
                'on_island': on_island
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
                'on_island': on_island
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

        # Apply friction (allows agents to stay still)
        self.prey_vel[self.prey_alive] *= (1.0 - FRICTION)

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

        # Apply river flow to prey only in river direction (preserves perpendicular motion)
        if self.river.enabled:
            flows = self.river.get_flow_at_batch_gpu(self.prey_pos[self.prey_alive])
            if len(flows) > 0:
                # Get river direction (unit vector)
                flow_mag = torch.norm(flows, dim=1, keepdim=True)
                river_dir = torch.where(
                    flow_mag > 0,
                    flows / flow_mag,
                    torch.zeros_like(flows)
                )

                # Decompose velocity into parallel and perpendicular components
                vel_parallel_mag = torch.sum(self.prey_vel[self.prey_alive] * river_dir, dim=1, keepdim=True)  # dot product
                vel_parallel = vel_parallel_mag * river_dir
                vel_perp = self.prey_vel[self.prey_alive] - vel_parallel

                # Add river flow to parallel component only (reduced by swim speed)
                alive_swim_speeds = self.prey_swim_speed[self.prey_alive]
                flow_factors = torch.clamp(1.0 - alive_swim_speeds / 5.0, min=0.0, max=1.0)
                vel_parallel += flows * flow_factors.unsqueeze(1)

                # Reconstruct velocity (perpendicular component unchanged)
                self.prey_vel[self.prey_alive] = vel_parallel + vel_perp

        self.prey_age[self.prey_alive] += 1
        self.prey_repro_timer[self.prey_alive] += 1

        self.pred_vel[self.pred_alive] += self.pred_acc[self.pred_alive]

        # Apply friction (allows agents to stay still)
        self.pred_vel[self.pred_alive] *= (1.0 - FRICTION)

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

        # Apply river flow to predators only in river direction (preserves perpendicular motion)
        if self.river.enabled:
            flows = self.river.get_flow_at_batch_gpu(self.pred_pos[self.pred_alive])
            if len(flows) > 0:
                # Get river direction (unit vector)
                flow_mag = torch.norm(flows, dim=1, keepdim=True)
                river_dir = torch.where(
                    flow_mag > 0,
                    flows / flow_mag,
                    torch.zeros_like(flows)
                )

                # Decompose velocity into parallel and perpendicular components
                vel_parallel_mag = torch.sum(self.pred_vel[self.pred_alive] * river_dir, dim=1, keepdim=True)  # dot product
                vel_parallel = vel_parallel_mag * river_dir
                vel_perp = self.pred_vel[self.pred_alive] - vel_parallel

                # Add river flow to parallel component only (reduced by swim speed)
                alive_swim_speeds = self.pred_swim_speed[self.pred_alive]
                flow_factors = torch.clamp(1.0 - alive_swim_speeds / 5.0, min=0.0, max=1.0)
                vel_parallel += flows * flow_factors.unsqueeze(1)

                # Reconstruct velocity (perpendicular component unchanged)
                self.pred_vel[self.pred_alive] = vel_parallel + vel_perp

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

        # Extinction prevention disabled - allow natural extinction
        # (Experiment ends when either species goes extinct)
        # self._handle_extinction_prevention()

        # Record statistics for evolution analysis
        self.record_stats()

        # Check for extinction - stop simulation if either species below threshold
        prey_count = self.prey_alive.sum().item()
        pred_count = self.pred_alive.sum().item()

        if prey_count <= EXTINCTION_THRESHOLD:
            print(f"\n💀 PREY EXTINCTION at timestep {self.timestep}!")
            print(f"   Population fell to {prey_count} (threshold: {EXTINCTION_THRESHOLD})")
            print("   Experiment over - predators won!")
            self.extinct = True
            self.extinction_message = f"Prey extinct at timestep {self.timestep}"

        if pred_count <= EXTINCTION_THRESHOLD:
            print(f"\n💀 PREDATOR EXTINCTION at timestep {self.timestep}!")
            print(f"   Population fell to {pred_count} (threshold: {EXTINCTION_THRESHOLD})")
            print("   Experiment over - prey won!")
            self.extinct = True
            self.extinction_message = f"Predators extinct at timestep {self.timestep}"

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

            # Classify prey as land or water specialists (GPU-resident)
            # Check which prey are in the river vs on island vs on land
            prey_pos = self.prey_pos[self.prey_alive]
            prey_on_island_mask = self.river.is_on_island_batch_gpu(prey_pos)
            prey_in_river_mask = self.river.is_in_river_batch_gpu(prey_pos)
            prey_on_island = torch.sum(prey_on_island_mask).item()
            prey_in_river = torch.sum(prey_in_river_mask).item()
            num_prey = len(prey_pos)
            state['prey_in_river_pct'] = (prey_in_river / num_prey) * 100 if num_prey > 0 else 0
            state['prey_on_island_pct'] = (prey_on_island / num_prey) * 100 if num_prey > 0 else 0
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

            # Check which predators are in the river vs on island vs on land (GPU-resident)
            pred_pos = self.pred_pos[self.pred_alive]
            pred_on_island_mask = self.river.is_on_island_batch_gpu(pred_pos)
            pred_in_river_mask = self.river.is_in_river_batch_gpu(pred_pos)
            pred_on_island = torch.sum(pred_on_island_mask).item()
            pred_in_river = torch.sum(pred_in_river_mask).item()
            num_pred = len(pred_pos)
            state['pred_in_river_pct'] = (pred_in_river / num_pred) * 100 if num_pred > 0 else 0
            state['pred_on_island_pct'] = (pred_on_island / num_pred) * 100 if num_pred > 0 else 0
        else:
            state['pred_avg_swim'] = 0
            state['pred_std_swim'] = 0
            state['pred_min_swim'] = 0
            state['pred_max_swim'] = 0
            state['pred_in_river_pct'] = 0
            state['pred_on_island_pct'] = 0

        return state

    def record_stats(self):
        """Record current state statistics for evolution tracking."""
        # Use get_state_cpu to get all necessary stats
        state = self.get_state_cpu()

        self.stats['timesteps'].append(self.timestep)
        self.stats['prey_count'].append(state['prey_count'])
        self.stats['pred_count'].append(state['pred_count'])
        self.stats['prey_avg_age'].append(state['prey_avg_age'])
        self.stats['pred_avg_age'].append(state['pred_avg_age'])
        self.stats['pred_avg_energy'].append(state['pred_avg_energy'])
        self.stats['prey_avg_swim'].append(state['prey_avg_swim'])
        self.stats['prey_std_swim'].append(state['prey_std_swim'])
        self.stats['pred_avg_swim'].append(state['pred_avg_swim'])
        self.stats['pred_std_swim'].append(state['pred_std_swim'])
        self.stats['prey_in_river_pct'].append(state['prey_in_river_pct'])
        self.stats['pred_in_river_pct'].append(state['pred_in_river_pct'])

    def save_stats(self, filename='stats_autosave.npz'):
        """Save statistics to file for evolution analysis."""
        # Convert lists to numpy arrays for saving
        save_dict = {key: np.array(val) for key, val in self.stats.items()}
        # Include metadata
        save_dict['metadata'] = np.array([self.metadata], dtype=object)
        np.savez(filename, **save_dict)
        print(f"Statistics saved to {filename}")
        print(f"  Data points: {len(self.stats['timesteps'])}")
        print(f"  Timesteps: {self.stats['timesteps'][0]} to {self.stats['timesteps'][-1]}")

    def save_brain_weights(self, name=None):
        """Save all agent brain weights and state to disk.

        Args:
            name: Optional name for the save file. If None, uses timestep.
        """
        import datetime

        if name is None:
            filename = f'brains_step_{self.timestep}.npz'
        else:
            # Clean the name, ensure .npz extension
            name = name.replace(' ', '_')
            if not name.endswith('.npz'):
                filename = f'brains_{name}.npz'
            else:
                filename = name

        # Update metadata with save time
        save_metadata = self.metadata.copy()
        save_metadata['save_time'] = datetime.datetime.now().isoformat()
        save_metadata['save_timestep'] = self.timestep

        np.savez(filename,
            # Brain weights (critical for persistence)
            prey_weights=self.prey_weights.cpu().numpy(),
            pred_weights=self.pred_weights.cpu().numpy(),

            # Agent state
            timestep=self.timestep,
            prey_alive=self.prey_alive.cpu().numpy(),
            pred_alive=self.pred_alive.cpu().numpy(),

            # Evolvable traits
            prey_swim_speed=self.prey_swim_speed.cpu().numpy(),
            pred_swim_speed=self.pred_swim_speed.cpu().numpy(),
            prey_max_age_individual=self.prey_max_age_individual.cpu().numpy(),
            prey_repro_age_individual=self.prey_repro_age_individual.cpu().numpy(),
            pred_max_age_individual=self.pred_max_age_individual.cpu().numpy(),
            pred_repro_cooldown_individual=self.pred_repro_cooldown_individual.cpu().numpy(),

            # Positions and velocities (for visualization/continuation)
            prey_pos=self.prey_pos.cpu().numpy(),
            pred_pos=self.pred_pos.cpu().numpy(),
            prey_vel=self.prey_vel.cpu().numpy(),
            pred_vel=self.pred_vel.cpu().numpy(),

            # Age and energy state
            prey_age=self.prey_age.cpu().numpy(),
            prey_repro_timer=self.prey_repro_timer.cpu().numpy(),
            pred_age=self.pred_age.cpu().numpy(),
            pred_energy=self.pred_energy.cpu().numpy(),
            pred_repro_timer=self.pred_repro_timer.cpu().numpy(),

            # Metadata
            metadata=np.array([save_metadata], dtype=object),
        )

        print(f"Brain weights saved to {filename}")
        print(f"  Timestep: {self.timestep}")
        print(f"  Prey: {self.prey_alive.sum().item()} alive / {len(self.prey_alive)} capacity")
        print(f"  Predators: {self.pred_alive.sum().item()} alive / {len(self.pred_alive)} capacity")

    def load_brain_weights(self, filename):
        """Load agent brain weights and state from disk.

        Args:
            filename: Path to the .npz file
        """
        print(f"Loading brain weights from {filename}...")
        data = np.load(filename, allow_pickle=True)

        # Restore brain weights
        self.prey_weights = torch.tensor(data['prey_weights'], device=self.device)
        self.pred_weights = torch.tensor(data['pred_weights'], device=self.device)

        # Restore alive masks
        self.prey_alive = torch.tensor(data['prey_alive'], device=self.device)
        self.pred_alive = torch.tensor(data['pred_alive'], device=self.device)

        # Restore evolvable traits
        self.prey_swim_speed = torch.tensor(data['prey_swim_speed'], device=self.device)
        self.pred_swim_speed = torch.tensor(data['pred_swim_speed'], device=self.device)
        self.prey_max_age_individual = torch.tensor(data['prey_max_age_individual'], device=self.device)
        self.prey_repro_age_individual = torch.tensor(data['prey_repro_age_individual'], device=self.device)
        self.pred_max_age_individual = torch.tensor(data['pred_max_age_individual'], device=self.device)
        self.pred_repro_cooldown_individual = torch.tensor(data['pred_repro_cooldown_individual'], device=self.device)

        # Restore positions and velocities
        self.prey_pos = torch.tensor(data['prey_pos'], device=self.device)
        self.pred_pos = torch.tensor(data['pred_pos'], device=self.device)
        self.prey_vel = torch.tensor(data['prey_vel'], device=self.device)
        self.pred_vel = torch.tensor(data['pred_vel'], device=self.device)

        # Restore age and energy state
        self.prey_age = torch.tensor(data['prey_age'], device=self.device)
        self.prey_repro_timer = torch.tensor(data['prey_repro_timer'], device=self.device)
        self.pred_age = torch.tensor(data['pred_age'], device=self.device)
        self.pred_energy = torch.tensor(data['pred_energy'], device=self.device)
        self.pred_repro_timer = torch.tensor(data['pred_repro_timer'], device=self.device)

        # Restore timestep
        self.timestep = int(data['timestep'])

        # Load metadata if present
        if 'metadata' in data:
            loaded_metadata = data['metadata'].item()
            print(f"  Run title: {loaded_metadata.get('run_title', 'Unknown')}")
            print(f"  Original start: {loaded_metadata.get('start_time', 'Unknown')}")

        print(f"Brain weights loaded successfully!")
        print(f"  Resumed at timestep: {self.timestep}")
        print(f"  Prey: {self.prey_alive.sum().item()} alive")
        print(f"  Predators: {self.pred_alive.sum().item()} alive")
