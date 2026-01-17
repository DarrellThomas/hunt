# HUNT Platform: Fix GPU Neuroevolution Bug

**Priority**: CRITICAL
**Budget**: ~$30
**Scope**: Fix the fundamental neuroevolution algorithm in GPU version

## Context

The audit identified a **critical bug** in `simulation_gpu.py`: the GPU version does not implement proper neuroevolution.

### Current (Broken) Behavior

```python
# simulation_gpu.py - Current implementation
class GPUEcosystem:
    def __init__(self, ...):
        # ONE shared brain for ALL prey
        self.prey_brain = NeuralNetBatch(num_prey, input_size=32, ...)
        # ONE shared brain for ALL predators  
        self.pred_brain = NeuralNetBatch(num_predators, input_size=21, ...)
    
    def step(self, ...):
        # Every 50 steps, mutate ALL agents simultaneously
        if self.timestep % 50 == 0:
            self.prey_brain.mutate_random(None, mutation_rate * 0.01)
            self.pred_brain.mutate_random(None, mutation_rate * 0.01)
```

**Problems**:
1. All agents share one global network — no individual variation
2. Mutation happens globally every 50 steps — not at reproduction
3. Offspring don't inherit parent weights — violates genetic algorithm
4. `NeuralNetBatch.mutate_random()` takes `indices` parameter but it's never used properly

### Correct Behavior (Like CPU Version)

```python
# CPU version - brain.py / agent.py
class Agent:
    def __init__(self, ...):
        self.brain = Brain(...)  # Each agent has INDIVIDUAL brain
    
    def reproduce(self, mutation_rate):
        child = self.__class__(...)
        child.brain = self.brain.copy()  # Offspring INHERITS parent weights
        child.brain.mutate(mutation_rate)  # Mutation at REPRODUCTION only
        return child
```

**Correct evolutionary dynamics**:
- Each agent has unique neural network weights
- Successful agents reproduce → offspring inherit their weights
- Mutation introduces variation at reproduction time
- Selection pressure drives evolution of better behaviors

## Your Task

Fix the GPU neuroevolution to match CPU behavior while maintaining GPU performance.

### Implementation Approach: Per-Agent Weight Storage

Store individual weights for each agent as GPU tensors, then use batched operations for forward passes.

#### Step 1: Calculate Weight Counts

```python
# Network architecture for prey: 32 → 32 → 32 → 2
# Layer 1: 32×32 weights + 32 biases = 1056
# Layer 2: 32×32 weights + 32 biases = 1056  
# Layer 3: 32×2 weights + 2 biases = 66
# Total: 2178 parameters per prey agent

# Network architecture for predators: 21 → 32 → 32 → 2
# Layer 1: 21×32 weights + 32 biases = 704
# Layer 2: 32×32 weights + 32 biases = 1056
# Layer 3: 32×2 weights + 2 biases = 66
# Total: 1826 parameters per predator agent
```

#### Step 2: Store Per-Agent Weights

```python
class GPUEcosystem:
    def __init__(self, num_prey, num_predators, ...):
        # Network architecture
        self.prey_arch = {'input': 32, 'hidden': [32, 32], 'output': 2}
        self.pred_arch = {'input': 21, 'hidden': [32, 32], 'output': 2}
        
        # Calculate weight counts
        self.prey_weight_count = self._calc_weight_count(self.prey_arch)
        self.pred_weight_count = self._calc_weight_count(self.pred_arch)
        
        # Store weights per agent (initialized randomly)
        # Shape: (max_agents, weight_count)
        self.prey_weights = torch.randn(
            num_prey, self.prey_weight_count, 
            device=self.device
        ) * 0.1  # Small initial weights
        
        self.pred_weights = torch.randn(
            num_predators, self.pred_weight_count,
            device=self.device
        ) * 0.1
    
    def _calc_weight_count(self, arch):
        """Calculate total parameters for network architecture."""
        sizes = [arch['input']] + arch['hidden'] + [arch['output']]
        count = 0
        for i in range(len(sizes) - 1):
            count += sizes[i] * sizes[i+1]  # weights
            count += sizes[i+1]  # biases
        return count
```

#### Step 3: Implement Batched Forward Pass

This is the performance-critical part. You need to compute forward passes for all agents in parallel using their individual weights.

```python
def _batch_forward(self, inputs: torch.Tensor, weights: torch.Tensor, 
                   arch: dict, alive_mask: torch.Tensor) -> torch.Tensor:
    """Batched forward pass with per-agent weights.
    
    Args:
        inputs: (N, input_size) observations for all agents
        weights: (N, weight_count) per-agent weights
        arch: Network architecture dict
        alive_mask: (N,) boolean mask of alive agents
        
    Returns:
        outputs: (N, output_size) actions for all agents
    """
    batch_size = inputs.shape[0]
    sizes = [arch['input']] + arch['hidden'] + [arch['output']]
    
    # Only process alive agents for efficiency
    alive_idx = torch.where(alive_mask)[0]
    if len(alive_idx) == 0:
        return torch.zeros(batch_size, arch['output'], device=self.device)
    
    alive_inputs = inputs[alive_idx]
    alive_weights = weights[alive_idx]
    
    # Unpack weights and compute forward pass
    x = alive_inputs
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
        
        # Activation (tanh for all layers, matching CPU version)
        x = torch.tanh(x)
    
    # Place results back into full tensor
    outputs = torch.zeros(batch_size, arch['output'], device=self.device)
    outputs[alive_idx] = x
    
    return outputs
```

#### Step 4: Modify Reproduction for Weight Inheritance

```python
def _reproduce_prey(self, mutation_rate):
    """Handle prey reproduction with weight inheritance."""
    # Find who can reproduce
    can_repro = self.prey_alive & (self.prey_repro_timer >= self.prey_repro_age_individual)
    
    # Find dead slots for offspring
    dead_idx = torch.where(~self.prey_alive)[0]
    parent_idx = torch.where(can_repro)[0]
    
    if len(dead_idx) == 0 or len(parent_idx) == 0:
        return
    
    # Limit births to available slots
    num_births = min(len(dead_idx), len(parent_idx))
    
    # Randomly select parents and slots
    selected_parents = parent_idx[torch.randperm(len(parent_idx))[:num_births]]
    selected_slots = dead_idx[:num_births]
    
    # === CRITICAL: Inherit weights from parents ===
    self.prey_weights[selected_slots] = self.prey_weights[selected_parents].clone()
    
    # === CRITICAL: Mutate offspring weights ===
    mutation = torch.randn_like(self.prey_weights[selected_slots]) * mutation_rate
    self.prey_weights[selected_slots] += mutation
    
    # Inherit and mutate swim_speed (existing trait)
    self.prey_swim_speed[selected_slots] = self.prey_swim_speed[selected_parents]
    self.prey_swim_speed[selected_slots] += torch.randn(num_births, device=self.device) * mutation_rate * 0.3
    self.prey_swim_speed[selected_slots].clamp_(0.1, 5.0)
    
    # Set offspring position (near parent)
    spawn_dist = torch.rand(num_births, device=self.device) * 130 + 20
    spawn_angle = torch.rand(num_births, device=self.device) * 2 * torch.pi
    offset_x = spawn_dist * torch.cos(spawn_angle)
    offset_y = spawn_dist * torch.sin(spawn_angle)
    
    self.prey_pos[selected_slots, 0] = self.prey_pos[selected_parents, 0] + offset_x
    self.prey_pos[selected_slots, 1] = self.prey_pos[selected_parents, 1] + offset_y
    
    # Wrap positions (toroidal)
    self.prey_pos[selected_slots, 0] %= self.width
    self.prey_pos[selected_slots, 1] %= self.height
    
    # Reset offspring state
    self.prey_vel[selected_slots] = 0
    self.prey_acc[selected_slots] = 0
    self.prey_age[selected_slots] = 0
    self.prey_repro_timer[selected_slots] = 0
    self.prey_alive[selected_slots] = True
    
    # Reset parent reproduction timer
    self.prey_repro_timer[selected_parents[:num_births]] = 0

# Similar implementation for _reproduce_predators()
```

#### Step 5: Remove Global Mutation

**DELETE** this code from `step()`:

```python
# REMOVE THIS:
if self.timestep % 50 == 0:
    self.prey_brain.mutate_random(None, mutation_rate * 0.01)
    self.pred_brain.mutate_random(None, mutation_rate * 0.01)
```

#### Step 6: Update Forward Pass Calls

Replace calls to `self.prey_brain(observations)` with the new batched forward:

```python
# Old:
# prey_actions = self.prey_brain(prey_observations)

# New:
prey_actions = self._batch_forward(
    prey_observations, 
    self.prey_weights,
    self.prey_arch,
    self.prey_alive
)
```

### Testing Requirements

#### Test 1: Weight Inheritance Verification

```python
def test_weight_inheritance():
    """Verify offspring inherit parent weights."""
    eco = GPUEcosystem(num_prey=100, num_predators=20, ...)
    
    # Get a parent's weights before reproduction
    parent_idx = 0
    parent_weights_before = eco.prey_weights[parent_idx].clone()
    
    # Force this parent to reproduce
    eco.prey_repro_timer[parent_idx] = 9999
    eco.prey_alive[parent_idx] = True
    
    # Find a dead slot
    eco.prey_alive[50] = False
    
    # Step to trigger reproduction
    eco.step(mutation_rate=0.0)  # Zero mutation for exact copy test
    
    # Check offspring (slot 50) has parent's weights
    offspring_weights = eco.prey_weights[50]
    assert torch.allclose(offspring_weights, parent_weights_before, atol=1e-6)
```

#### Test 2: Mutation Creates Diversity

```python
def test_mutation_creates_diversity():
    """Verify mutation introduces variation."""
    eco = GPUEcosystem(num_prey=100, ...)
    
    # All prey start with similar weights
    initial_diversity = eco.prey_weights.std(dim=0).mean()
    
    # Run for many generations with reproduction
    for _ in range(1000):
        eco.step(mutation_rate=0.1)
    
    # Diversity should increase
    final_diversity = eco.prey_weights[eco.prey_alive].std(dim=0).mean()
    assert final_diversity > initial_diversity * 1.5
```

#### Test 3: CPU/GPU Evolutionary Parity

```python
def test_cpu_gpu_evolution_similarity():
    """Verify CPU and GPU produce similar evolutionary dynamics."""
    import numpy as np
    
    # Run CPU version
    np.random.seed(42)
    torch.manual_seed(42)
    cpu_world = World(width=800, height=600, ...)
    cpu_fitness_history = []
    for _ in range(500):
        cpu_world.step(mutation_rate=0.1)
        avg_fitness = np.mean([p.fitness for p in cpu_world.prey])
        cpu_fitness_history.append(avg_fitness)
    
    # Run GPU version
    np.random.seed(42)
    torch.manual_seed(42)
    gpu_eco = GPUEcosystem(...)
    gpu_fitness_history = []
    for _ in range(500):
        gpu_eco.step(mutation_rate=0.1)
        # Calculate comparable fitness metric
        ...
    
    # Correlation should be positive (similar trends)
    correlation = np.corrcoef(cpu_fitness_history, gpu_fitness_history)[0,1]
    assert correlation > 0.5, f"CPU/GPU evolution too different: {correlation}"
```

#### Test 4: Performance Benchmark

```python
def test_performance_not_regressed():
    """Ensure per-agent weights don't kill performance."""
    import time
    
    eco = GPUEcosystem(num_prey=5000, num_predators=1000, ...)
    
    # Warmup
    for _ in range(10):
        eco.step(mutation_rate=0.1)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        eco.step(mutation_rate=0.1)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    steps_per_second = 100 / elapsed
    print(f"Performance: {steps_per_second:.1f} steps/sec")
    
    # Should still be reasonable (adjust threshold based on your hardware)
    assert steps_per_second > 5, f"Too slow: {steps_per_second} steps/sec"
```

### Deliverables

1. **Modified `simulation_gpu.py`** with:
   - Per-agent weight storage (`prey_weights`, `pred_weights` tensors)
   - `_calc_weight_count()` method
   - `_batch_forward()` method for efficient batched inference
   - `_reproduce_prey()` and `_reproduce_predators()` with weight inheritance
   - Removed global mutation code

2. **Test file `test_gpu_neuroevolution.py`** with:
   - Weight inheritance test
   - Mutation diversity test
   - CPU/GPU parity test
   - Performance benchmark

3. **Updated CHANGELOG.md** documenting the fix

### Validation Checklist

- [ ] Each agent has individual weights stored in GPU tensor
- [ ] Forward pass uses per-agent weights (verify with debugger/print)
- [ ] Offspring weights are copied from parent at reproduction
- [ ] Offspring weights are mutated at reproduction
- [ ] NO global mutation every 50 steps (code removed)
- [ ] Tests pass
- [ ] Performance acceptable (>5 steps/sec with 6000 agents)
- [ ] Visual inspection: behaviors should evolve over time (not static)

### Git Workflow

```bash
git checkout -b fix/gpu-neuroevolution
# ... implement fix ...
git add simulation_gpu.py test_gpu_neuroevolution.py CHANGELOG.md
git commit -m "fix: implement proper per-agent neuroevolution in GPU version

- Add per-agent weight storage as GPU tensors
- Implement batched forward pass with individual weights  
- Add weight inheritance at reproduction
- Add mutation at reproduction time
- Remove incorrect global mutation every 50 steps
- Add comprehensive tests for evolutionary dynamics"
```

## Begin

1. Read `simulation_gpu.py` carefully to understand current structure
2. Read `brain.py` to understand CPU brain architecture
3. Implement the fix following the steps above
4. Write and run tests
5. Verify visual behavior (run simulation, watch evolution happen)
