"""
Fully GPU-accelerated simulation.
Everything stays on GPU - positions, velocities, neural networks, physics.
Only visualization data goes to CPU.
"""

import torch
import torch.nn as nn
import numpy as np


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

        # Prey parameters
        self.num_prey = num_prey
        self.prey_max_speed = 3.0
        self.prey_max_accel = 0.5
        self.prey_max_age = 500
        self.prey_repro_age = 200

        # Predator parameters
        self.num_predators = num_predators
        self.pred_max_speed = 2.5
        self.pred_max_accel = 0.4
        self.pred_max_age = 800
        self.pred_max_energy = 150.0
        self.pred_energy_cost = 0.3
        self.pred_energy_gain = 60.0
        self.pred_repro_threshold = 120.0
        self.pred_repro_cost = 40.0
        self.pred_repro_cooldown = 150
        self.catch_radius = 8.0

        # Initialize prey on GPU
        self.prey_pos = torch.rand(num_prey, 2, device=device) * torch.tensor([width, height], device=device)
        self.prey_vel = torch.randn(num_prey, 2, device=device) * 0.1
        self.prey_acc = torch.zeros(num_prey, 2, device=device)
        self.prey_age = torch.zeros(num_prey, device=device)
        self.prey_repro_timer = torch.zeros(num_prey, device=device)
        self.prey_alive = torch.ones(num_prey, dtype=torch.bool, device=device)

        # Individual lifespan and reproduction timing for each prey (normal distribution)
        # Prevents synchronized birth/death waves
        self.prey_max_age_individual = torch.clamp(
            torch.normal(self.prey_max_age, 50, size=(num_prey,), device=device),
            min=100
        )
        self.prey_repro_age_individual = torch.clamp(
            torch.normal(self.prey_repro_age, 20, size=(num_prey,), device=device),
            min=50
        )

        # Initialize predators on GPU
        self.pred_pos = torch.rand(num_predators, 2, device=device) * torch.tensor([width, height], device=device)
        self.pred_vel = torch.randn(num_predators, 2, device=device) * 0.1
        self.pred_acc = torch.zeros(num_predators, 2, device=device)
        self.pred_age = torch.zeros(num_predators, device=device)
        self.pred_energy = torch.full((num_predators,), self.pred_max_energy, device=device)
        self.pred_repro_timer = torch.zeros(num_predators, device=device)
        self.pred_alive = torch.ones(num_predators, dtype=torch.bool, device=device)

        # Individual lifespan and reproduction timing for each predator (normal distribution)
        # Prevents synchronized birth/death waves
        self.pred_max_age_individual = torch.clamp(
            torch.normal(self.pred_max_age, 80, size=(num_predators,), device=device),
            min=200
        )
        self.pred_repro_cooldown_individual = torch.clamp(
            torch.normal(self.pred_repro_cooldown, 15, size=(num_predators,), device=device),
            min=50
        )

        # Neural networks
        self.prey_brain = NeuralNetBatch(num_prey, input_size=32, hidden_size=32, output_size=2, device=device)
        self.pred_brain = NeuralNetBatch(num_predators, input_size=21, hidden_size=32, output_size=2, device=device)

        print(f"GPU Ecosystem initialized on {device}")
        print(f"World: {width}x{height}")
        print(f"Prey: {num_prey}, Predators: {num_predators}")

    def compute_toroidal_distances(self, pos1, pos2):
        """Compute all pairwise distances with toroidal wrapping on GPU."""
        # pos1: (N, 2), pos2: (M, 2)
        # Returns: (N, M) distances, (N, M, 2) direction vectors

        # Broadcast to compute all pairs: (N, 1, 2) - (1, M, 2) = (N, M, 2)
        diff = pos1.unsqueeze(1) - pos2.unsqueeze(0)

        # Toroidal wrapping
        diff[:, :, 0] = torch.where(
            torch.abs(diff[:, :, 0]) > self.width / 2,
            diff[:, :, 0] - torch.sign(diff[:, :, 0]) * self.width,
            diff[:, :, 0]
        )
        diff[:, :, 1] = torch.where(
            torch.abs(diff[:, :, 1]) > self.height / 2,
            diff[:, :, 1] - torch.sign(diff[:, :, 1]) * self.height,
            diff[:, :, 1]
        )

        distances = torch.norm(diff, dim=2)
        return distances, diff

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

    def step(self, mutation_rate=0.1):
        """Single simulation step - fully on GPU."""
        self.timestep += 1

        # 1. Observations and actions
        prey_obs = self.observe_prey()
        prey_actions = self.prey_brain(prey_obs)  # Batched forward pass!
        self.prey_acc[self.prey_alive] = prey_actions * self.prey_max_accel

        pred_obs = self.observe_predators()
        pred_actions = self.pred_brain(pred_obs)  # Batched forward pass!
        self.pred_acc[self.pred_alive] = pred_actions * self.pred_max_accel

        # 2. Physics update (vectorized)
        self.prey_vel[self.prey_alive] += self.prey_acc[self.prey_alive]
        speed = torch.norm(self.prey_vel[self.prey_alive], dim=1, keepdim=True)
        self.prey_vel[self.prey_alive] = torch.where(
            speed > self.prey_max_speed,
            self.prey_vel[self.prey_alive] / speed * self.prey_max_speed,
            self.prey_vel[self.prey_alive]
        )
        self.prey_pos[self.prey_alive] = (self.prey_pos[self.prey_alive] + self.prey_vel[self.prey_alive]) % torch.tensor([self.width, self.height], device=self.device)
        self.prey_age[self.prey_alive] += 1
        self.prey_repro_timer[self.prey_alive] += 1

        self.pred_vel[self.pred_alive] += self.pred_acc[self.pred_alive]
        speed = torch.norm(self.pred_vel[self.pred_alive], dim=1, keepdim=True)
        self.pred_vel[self.pred_alive] = torch.where(
            speed > self.pred_max_speed,
            self.pred_vel[self.pred_alive] / speed * self.pred_max_speed,
            self.pred_vel[self.pred_alive]
        )
        self.pred_pos[self.pred_alive] = (self.pred_pos[self.pred_alive] + self.pred_vel[self.pred_alive]) % torch.tensor([self.width, self.height], device=self.device)
        self.pred_age[self.pred_alive] += 1
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
                    self.pred_energy[alive_pred_idx] = min(self.pred_max_energy,
                                                           self.pred_energy[alive_pred_idx] + self.pred_energy_gain)

            # Remove caught prey
            alive_indices = torch.where(self.prey_alive)[0]
            self.prey_alive[alive_indices[prey_caught]] = False

        # 4. Deaths (using individual age limits - prevents synchronized deaths)
        self.prey_alive &= self.prey_age < self.prey_max_age_individual
        self.pred_alive &= (self.pred_age < self.pred_max_age_individual) & (self.pred_energy > 0)

        # 5. Reproduction (using individual timing - prevents synchronized births)
        can_repro_prey = self.prey_alive & (self.prey_repro_timer >= self.prey_repro_age_individual)
        can_repro_pred = self.pred_alive & (self.pred_energy >= self.pred_repro_threshold) & (self.pred_repro_timer >= self.pred_repro_cooldown_individual)

        # Respawn dead agents as offspring of survivors
        dead_prey_idx = torch.where(~self.prey_alive)[0]
        alive_prey_idx = torch.where(can_repro_prey)[0]
        if len(dead_prey_idx) > 0 and len(alive_prey_idx) > 0:
            # Random parents
            parents = alive_prey_idx[torch.randint(0, len(alive_prey_idx), (len(dead_prey_idx),), device=self.device)]
            self.prey_pos[dead_prey_idx] = self.prey_pos[parents] + torch.randn(len(dead_prey_idx), 2, device=self.device) * 20
            self.prey_pos[dead_prey_idx] %= torch.tensor([self.width, self.height], device=self.device)
            self.prey_vel[dead_prey_idx] = torch.randn(len(dead_prey_idx), 2, device=self.device) * 0.1
            self.prey_age[dead_prey_idx] = 0
            self.prey_repro_timer[dead_prey_idx] = 0
            self.prey_alive[dead_prey_idx] = True
            # Give offspring randomized individual lifespans and reproduction ages
            self.prey_max_age_individual[dead_prey_idx] = torch.clamp(
                torch.normal(self.prey_max_age, 50, size=(len(dead_prey_idx),), device=self.device),
                min=100
            )
            self.prey_repro_age_individual[dead_prey_idx] = torch.clamp(
                torch.normal(self.prey_repro_age, 20, size=(len(dead_prey_idx),), device=self.device),
                min=50
            )
            self.prey_repro_timer[parents] = 0  # Reset parent timers

        dead_pred_idx = torch.where(~self.pred_alive)[0]
        alive_pred_idx = torch.where(can_repro_pred)[0]
        if len(dead_pred_idx) > 0 and len(alive_pred_idx) > 0:
            parents = alive_pred_idx[torch.randint(0, len(alive_pred_idx), (len(dead_pred_idx),), device=self.device)]
            self.pred_pos[dead_pred_idx] = self.pred_pos[parents] + torch.randn(len(dead_pred_idx), 2, device=self.device) * 20
            self.pred_pos[dead_pred_idx] %= torch.tensor([self.width, self.height], device=self.device)
            self.pred_vel[dead_pred_idx] = torch.randn(len(dead_pred_idx), 2, device=self.device) * 0.1
            self.pred_age[dead_pred_idx] = 0
            self.pred_energy[dead_pred_idx] = self.pred_max_energy
            self.pred_repro_timer[dead_pred_idx] = 0
            self.pred_alive[dead_pred_idx] = True
            # Give offspring randomized individual lifespans and reproduction cooldowns
            self.pred_max_age_individual[dead_pred_idx] = torch.clamp(
                torch.normal(self.pred_max_age, 80, size=(len(dead_pred_idx),), device=self.device),
                min=200
            )
            self.pred_repro_cooldown_individual[dead_pred_idx] = torch.clamp(
                torch.normal(self.pred_repro_cooldown, 15, size=(len(dead_pred_idx),), device=self.device),
                min=50
            )
            self.pred_energy[parents] -= self.pred_repro_cost
            self.pred_repro_timer[parents] = 0

        # Emergency extinction prevention - respawn 5 if population hits 0
        prey_alive_count = self.prey_alive.sum().item()
        pred_alive_count = self.pred_alive.sum().item()

        if prey_alive_count < 1:
            print(f"\n⚠️  PREY EXTINCTION at timestep {self.timestep}! Respawning 5 random prey...")
            # Find 5 dead prey slots to revive
            dead_prey = torch.where(~self.prey_alive)[0][:5]
            if len(dead_prey) > 0:
                # Random positions across the map
                self.prey_pos[dead_prey] = torch.rand(len(dead_prey), 2, device=self.device) * torch.tensor([self.width, self.height], device=self.device)
                self.prey_vel[dead_prey] = torch.randn(len(dead_prey), 2, device=self.device) * 0.1
                self.prey_age[dead_prey] = 0
                self.prey_repro_timer[dead_prey] = 0
                self.prey_alive[dead_prey] = True
                # Give random individual parameters
                self.prey_max_age_individual[dead_prey] = torch.clamp(
                    torch.normal(self.prey_max_age, 50, size=(len(dead_prey),), device=self.device),
                    min=100
                )
                self.prey_repro_age_individual[dead_prey] = torch.clamp(
                    torch.normal(self.prey_repro_age, 20, size=(len(dead_prey),), device=self.device),
                    min=50
                )

        if pred_alive_count < 1:
            print(f"\n⚠️  PREDATOR EXTINCTION at timestep {self.timestep}! Respawning 5 random predators...")
            # Find 5 dead predator slots to revive
            dead_pred = torch.where(~self.pred_alive)[0][:5]
            if len(dead_pred) > 0:
                # Random positions across the map
                self.pred_pos[dead_pred] = torch.rand(len(dead_pred), 2, device=self.device) * torch.tensor([self.width, self.height], device=self.device)
                self.pred_vel[dead_pred] = torch.randn(len(dead_pred), 2, device=self.device) * 0.1
                self.pred_age[dead_pred] = 0
                self.pred_energy[dead_pred] = self.pred_max_energy
                self.pred_repro_timer[dead_pred] = 0
                self.pred_alive[dead_pred] = True
                # Give random individual parameters
                self.pred_max_age_individual[dead_pred] = torch.clamp(
                    torch.normal(self.pred_max_age, 80, size=(len(dead_pred),), device=self.device),
                    min=200
                )
                self.pred_repro_cooldown_individual[dead_pred] = torch.clamp(
                    torch.normal(self.pred_repro_cooldown, 15, size=(len(dead_pred),), device=self.device),
                    min=50
                )

        # Mutate brains occasionally
        if self.timestep % 50 == 0:
            self.prey_brain.mutate_random(None, mutation_rate * 0.01)
            self.pred_brain.mutate_random(None, mutation_rate * 0.01)

    def get_state_cpu(self):
        """Transfer current state to CPU for visualization."""
        return {
            'prey_pos': self.prey_pos[self.prey_alive].cpu().numpy(),
            'pred_pos': self.pred_pos[self.pred_alive].cpu().numpy(),
            'prey_count': self.prey_alive.sum().item(),
            'pred_count': self.pred_alive.sum().item(),
            'prey_avg_age': self.prey_age[self.prey_alive].mean().item() if self.prey_alive.sum() > 0 else 0,
            'pred_avg_age': self.pred_age[self.pred_alive].mean().item() if self.pred_alive.sum() > 0 else 0,
            'pred_avg_energy': self.pred_energy[self.pred_alive].mean().item() if self.pred_alive.sum() > 0 else 0,
        }
