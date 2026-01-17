# HUNT Technical Debt

This document catalogs hardcoded assumptions, code smells, performance bottlenecks, and potential bugs that should be addressed before extending the platform.

## Critical Issues (Must Fix Before Extension)

### 1. Hardcoded Two-Species Architecture

**Location**: `world.py`, `simulation_gpu.py`, visualizers

**Problem**:
```python
# world.py
def __init__(self, ...):
    self.prey = []
    self.predators = []

def step(self):
    prey_positions = np.array([p.pos for p in self.prey])
    predator_positions = np.array([p.pos for p in self.predators])

    for i, prey in enumerate(self.prey):
        observation = prey.observe(predator_positions, ...)
```

**Impact**: Cannot add a 3rd species without rewriting entire simulation loop

**Scope**: ~400 lines across world.py, simulation_gpu.py, both visualizers

**Fix Difficulty**: Large refactor needed

---

### 2. Fixed Observation Dimensions

**Location**: `agent.py` lines 152-228, 273-308

**Problem**:
```python
class Prey:
    def __init__(self, ...):
        # Hardcoded: 5 predators * 4 values + 3 prey * 4 values = 32 inputs
        super().__init__(x, y, world_width, world_height, brain, input_size=32)

    def observe(self, ...):
        observation = []
        # Find 5 nearest predators
        # Find 3 nearest prey
        return np.array(observation, dtype=np.float32)  # Always 32 dimensions
```

**Impact**:
- Cannot add new sensors (e.g., "sense energy", "see island")
- Cannot change number of sensed neighbors
- Brain architecture must match exactly

**Examples of desired extensions blocked**:
- "Prey can sense predator energy levels" → Need 5*5 = 25 dims instead of 5*4 = 20
- "Add 'distance to island' sensor" → Need +1 dim = 33 total
- "Sense 10 predators instead of 5" → Need 40 dims for predators alone

**Fix Difficulty**: Medium - need dynamic observation system

---

###3. Species-Specific Reproduction Logic

**Location**: `agent.py` lines 234-236, 324-330

**Problem**:
```python
# Prey
def can_reproduce(self):
    return self.time_since_reproduction >= self.reproduction_age

# Predator
def can_reproduce(self):
    return (self.energy >= self.reproduction_threshold and
            self.time_since_reproduction >= self.reproduction_cooldown)
```

**Impact**:
- Cannot define reproduction rules generically
- Adding new species requires custom logic each time
- No way to mix strategies (e.g., "herbivore with energy system")

**Fix Difficulty**: Medium - need reproduction strategy pattern

---

### 4. GPU-CPU Transfer Bottleneck

**Location**: `simulation_gpu.py` lines 282-290, 305-313, 343-347, 383-387, 419-424, 462-482

**Problem**:
```python
# EVERY STEP:
# Transfer prey positions GPU → CPU
alive_prey_pos_np = self.prey_pos[self.prey_alive].cpu().numpy()

# Compute flows on CPU
flows = self.river.get_flow_at_batch(alive_prey_pos_np)

# Transfer flows CPU → GPU
flow_tensor = torch.tensor(flows, device=self.device, dtype=torch.float32)
```

**Impact**:
- 4× transfers per step (prey flow, pred flow, prey island, pred island)
- With 10K agents: ~40K position transfers + ~40K flow/modifier transfers per step
- Accounts for ~30% of step time

**Measurements**: Step time 58ms, estimated 15-20ms in transfers

**Fix Options**:
1. Implement river flow on GPU (requires porting river.py to PyTorch)
2. Cache island regions (only check when agents move significant distance)
3. Spatial grid for faster island lookups

**Fix Difficulty**: Medium-Large depending on approach

---

### 5. Configuration System Not Extensible

**Location**: `config.py`, imported everywhere with `from config import *`

**Problems**:
```python
# config.py
PREY_MAX_SPEED = 3.0
PREY_MAX_ACCELERATION = 0.5
PRED_MAX_SPEED = 2.5
PRED_MAX_ACCELERATION = 0.4
# ... 50+ more constants
```

**Issues**:
1. **No multi-species support**: `PREY_*` and `PRED_*` hardcoded
2. **No validation**: Can set `PREY_MAX_SPEED = -5` with no error
3. **No experiment variants**: Cannot easily run "what if predators were faster?"
4. **Global state**: Changing config requires editing file + restart
5. **No inheritance**: Defining species C that's "like prey but faster" requires copy-paste

**Impact**: Blocks batch experiments and N-species support

**Fix Difficulty**: Medium - need config class + per-species configs

---

### 6. Unclear Brain-Agent Coupling in GPU Version

**Location**: `simulation_gpu.py` lines 132-133, 489-492

**Problem**:
```python
# Brains are global for all agents
self.prey_brain = NeuralNetBatch(num_prey, input_size=32, ...)
self.pred_brain = NeuralNetBatch(num_predators, input_size=21, ...)

# But mutation is global too:
if self.timestep % 50 == 0:
    self.prey_brain.mutate_random(None, mutation_rate * 0.01)
    self.pred_brain.mutate_random(None, mutation_rate * 0.01)
```

**Issues**:
1. **Periodic global mutation** (every 50 steps) mutates ALL agents simultaneously
2. **Contradicts neuroevolution**: Offspring should inherit parent's weights
3. **Not matching CPU version**: CPU has individual brains per agent
4. **NeuralNetBatch.mutate_random()**: Takes `indices` parameter but it's never used

**Current behavior**: Agents share one global network that mutates occasionally

**CPU behavior**: Each agent has individual brain, mutation on reproduction

**Impact**: GPU version is not actually doing neuroevolution correctly!

**Fix Difficulty**: Large - need per-agent weight storage or proper genetic algorithm

---

## Important Issues (Will Cause Pain Later)

### 7. Copy-Pasted Distance Calculations

**Locations**: `agent.py` lines 117-145, `world.py` lines 114-121, `simulation_gpu.py` lines 139-160

**Problem**: Three different implementations of toroidal distance:
1. `agent.py`: `vectorized_distances()` function
2. `world.py`: Inline vectorized calculation in collision detection
3. `simulation_gpu.py`: `compute_toroidal_distances()` method

**Impact**: Bug in distance calculation requires fixing in 3 places

**Fix**: Extract to shared utility module

---

### 8. Reproduction Position Calculation Duplicated

**Locations**: `agent.py` lines 91-98, `world.py` (implicitly via agent.reproduce()), `simulation_gpu.py` lines 365-370, 402-407

**Problem**:
```python
# agent.py
spawn_distance = np.random.uniform(20, 150)
spawn_angle = np.random.uniform(0, 2 * np.pi)
offset_x = spawn_distance * np.cos(spawn_angle)
offset_y = spawn_distance * np.sin(spawn_angle)

# simulation_gpu.py (slightly different)
spawn_distance = torch.rand(len(dead_prey_idx), device=self.device) * 130 + 20  # 20 to 150
spawn_angle = torch.rand(len(dead_prey_idx), device=self.device) * 2 * 3.14159
```

**Issues**:
1. Magic numbers (20, 150) repeated
2. CPU uses `np.random`, GPU uses `torch.rand`
3. GPU hardcodes π as `3.14159` instead of using `torch.pi`

**Fix**: Shared spawn offset function with configurable distance range

---

### 9. Inconsistent Extinction Prevention

**Location**: `world.py` lines 169-199, `simulation_gpu.py` lines 432-487

**Problem**:
```python
# world.py:Respawns 5 agents if population hits 0
if len(self.prey) < 1:
    for _ in range(5):
        # Spawn random prey

# Also has minimum threshold prevention (scales with world size)
min_prey = max(10, int(self.width * self.height / 24000))
if len(self.prey) < min_prey:
    # Spawn more

# simulation_gpu.py:
# Only respawns 5 if hits exactly 0, no minimum threshold
if prey_alive_count < 1:
    # Respawn 5
```

**Issues**:
1. CPU has two systems (emergency + minimum), GPU has one
2. Magic numbers (5, 10, 24000) not in config
3. Different strategies could lead to different evolution
4. Minimum threshold in CPU prevents true population collapse

**Impact**: CPU and GPU versions have different evolutionary dynamics

**Fix**: Unify extinction prevention strategy, move thresholds to config

---

### 10. Statistics Tracking Inconsistency

**Location**: `world.py` lines 44-51, `main_gpu.py` lines 40-55

**Problem**:
- CPU tracks 6 metrics: counts, avg_age, avg_fitness
- GPU tracks 12 metrics: + swim speeds, habitat preferences
- Different save locations: `stats.npz` vs `stats_autosave.npz`
- GPU autosaves, CPU only saves on demand

**Impact**: Analysis scripts need to handle two formats

**Fix**: Unified stats system with versioning

---

### 11. River Path Generation Has Magic Numbers

**Location**: `river.py` lines 41-60

**Problem**:
```python
num_points = 50  # Why 50?
curve = np.sin(t * np.pi * 4) * self.curviness * self.height * 0.2
curve += np.sin(t * np.pi * 2.3) * self.curviness * self.height * 0.15
#                              ^^^                                 ^^^^
#                           Why 2.3?                            Why 0.15?
```

**Issues**:
- Magic numbers not documented
- `num_points = 50` might be too few for large worlds
- Sine wave parameters (4, 2.3, 0.2, 0.15) seem arbitrary
- No validation that path stays in world bounds except clamping

**Impact**: Difficult to customize river generation

**Fix**: Document parameters, make configurable, validate bounds properly

---

### 12. Island Behavior System Half-Implemented

**Location**: `river.py` lines 195-223, `simulation_gpu.py` lines 258-324, 342-482

**Problem**: Island behavior modifiers exist but are not used in CPU version

```python
# river.py defines island_behavior()
def island_behavior(self, agent_type, x, y):
    # Returns modifiers for speed, hunger, reproduction

# simulation_gpu.py uses it (recently added)
island_mods = self.get_island_modifiers(...)

# world.py DOES NOT use it
# Agents on island still behave normally
```

**Impact**:
- CPU and GPU have different island mechanics
- CPU version doesn't benefit from configurable island effects
- Inconsistent experiment results

**Fix**: Integrate island modifiers into world.py

---

### 13. No Validation of Agent State

**Location**: Throughout `agent.py`, `world.py`, `simulation_gpu.py`

**Problem**: No checks for invalid states:
- Energy can go negative (checked only in death condition)
- Position can NaN if physics glitches
- Velocity can exceed speed limits between steps
- Age can overflow (though unlikely with reasonable lifespans)

**Example failure case**:
```python
# If acceleration is very large due to bug:
vel += acc * dt  # vel could explode
# Then:
pos += vel * dt  # pos could become NaN
# Then:
distances = sqrt((pos1 - pos2)**2)  # NaN propagates
```

**Impact**: Silent failures that are hard to debug

**Fix**: Add assertions or validation in debug mode

---

### 14. Observation Sampling In GPU Inconsistent

**Location**: `simulation_gpu.py` lines 162-216, 218-256

**Problem**:
```python
# For prey observing predators:
max_pred_sample = min(100, len(alive_pred_pos))  # Sample up to 100

# For predators observing prey:
max_prey_sample = min(200, len(alive_prey_pos))  # Sample up to 200

# Why 100 vs 200? Asymmetry not documented.
```

**Issues**:
- Asymmetric sampling (100 pred, 200 prey) not justified
- Hard-coded limits not in config
- If population drops below sample size, uses all agents (changes observation quality)

**Impact**: Predators and prey have different "vision quality"

**Fix**: Make sample sizes configurable, document reasoning

---

## Performance Issues

### 15. CPU Version Uses Lists for Agents

**Location**: `world.py` lines 30-31

**Problem**:
```python
self.prey = []  # List of agent objects
self.predators = []  # List of agent objects

# Every step:
for prey in self.prey:  # Python loop
    observation = prey.observe(...)  # Python method call
```

**Impact**: Python loops are slow, limits scalability to ~500 agents

**Not critical because**: GPU version exists for large scale

**Optimization**: Use NumPy structured arrays or agent pools

---

### 16. Unnecessary Array Copies in CPU Version

**Location**: `world.py` lines 63-66, 106-107

**Problem**:
```python
# Create arrays every step
prey_positions = np.array([p.pos for p in self.prey])  # Copy
prey_velocities = np.array([p.vel for p in self.prey])  # Copy

# Later, recreate them again
prey_positions = np.array([p.pos for p in self.prey])  # Duplicate copy
```

**Impact**: Extra memory allocation + copying

**Fix**: Cache position/velocity arrays, update in-place

---

### 17. GPU Extinction Prevention Inefficient

**Location**: `simulation_gpu.py` lines 432-487

**Problem**:
```python
# Check EVERY STEP even when population is healthy
prey_alive_count = self.prey_alive.sum().item()  # GPU→CPU transfer
pred_alive_count = self.pred_alive.sum().item()  # GPU→CPU transfer

if prey_alive_count < 1:  # Almost never true
    # ... respawn logic
```

**Impact**: 2 extra GPU→CPU transfers per step even when not needed

**Optimization**: Check only every N steps or when death events occur

---

## Code Quality Issues

### 18. Inconsistent Naming Conventions

**Examples**:
- `prey_count` vs `num_prey` vs `initial_prey`
- `predator_count` vs `num_predators` vs `initial_predators`
- `max_speed` vs `prey_max_speed` vs `PREY_MAX_SPEED`
- `pred_*` vs `predator_*` (abbreviated inconsistently)

**Impact**: Harder to search, cognitive load

---

### 19. Poor Function Naming

**Location**: `simulation_gpu.py` line 258

**Problem**:
```python
def get_island_modifiers(self, positions, agent_type):
    # Actually checks if on island AND returns modifiers
    # Name suggests it always returns modifiers
```

Better name: `compute_island_effects()` or `apply_island_modifiers()`

---

### 20. Unclear Variable Scope in GPU Step

**Location**: `simulation_gpu.py` lines 326-482

**Problem**: The `step()` method is 156 lines with variables used across large spans:
```python
def step(self, mutation_rate=0.1):
    # Line 342: define prey_island_mods
    prey_island_mods = self.get_island_modifiers(...)

    # Lines 343-363: use prey_island_mods

    # Lines 364-378: unrelated code

    # Line 383: define pred_island_mods (70 lines after prey_island_mods)
    pred_island_mods = self.get_island_modifiers(...)

    # Line 419: use pred_island_mods again
    # Line 462: use prey_repro_mods (new variable)
    # Line 475: use pred_repro_mods
```

**Impact**: Hard to track variable lifespan, easy to reuse stale values

**Fix**: Break into smaller methods: `_update_physics()`, `_handle_reproduction()`, etc.

---

### 21. Magic String Literals

**Location**: `river.py` lines 211-221, `simulation_gpu.py` lines 285-322

**Problem**:
```python
if agent_type == 'prey':  # String literal
    # ...
elif agent_type == 'predator':  # String literal
```

**Issues**:
- Typo risk: `'predetor'` would silently fail
- No type safety
- Difficult to refactor

**Fix**: Use Enum or constants

---

### 22. Commented-Out Code and TODOs

**None found**: Code is clean in this regard

---

### 23. Inconsistent Error Handling

**Problem**: Almost no error handling anywhere

**Examples**:
- No check if GPU available before using CUDA
- No validation of config values
- No handling of empty populations in analysis scripts
- River assumes positions are in bounds

**Impact**: Cryptic errors when things go wrong

---

## Potential Bugs

### 24. Race Condition in Reproduction

**Location**: `world.py` lines 151-167

**Problem**:
```python
new_prey = []
for prey in self.prey:
    if prey.can_reproduce():
        child = prey.reproduce(mutation_rate)
        new_prey.append(child)
        prey.time_since_reproduction = 0  # Modify during iteration

# Later:
self.prey.extend(new_prey)
```

**Issues**:
1. Modifying agent state during iteration over same list
2. If agent dies between can_reproduce() check and reproduction, still reproduces
3. If agent reproduces twice in same step (shouldn't happen but no guard), could double-spawn

**Severity**: Low (unlikely to trigger)

**Fix**: Separate reproduction checks from state modification

---

### 25. GPU Reproduction May Overwrite Living Agents

**Location**: `simulation_gpu.py` lines 484-492

**Problem**:
```python
dead_prey_idx = torch.where(~self.prey_alive)[0]
alive_prey_idx = torch.where(can_repro_prey)[0]

if len(dead_prey_idx) > 0 and len(alive_prey_idx) > 0:
    # ... spawn offspring in dead_prey_idx slots

    self.prey_alive[dead_prey_idx] = True  # Mark as alive
```

**Issue**: If more agents die after `dead_prey_idx` is computed (e.g., in collision detection), those indices might not be dead anymore

**Severity**: Low (deaths happen before reproduction in step order)

**Actual Risk**: Safe due to step ordering, but fragile if step order changes

---

### 26. Toroidal Distance Edge Case

**Location**: `agent.py` lines 137-139

**Problem**:
```python
dx = np.where(np.abs(dx) > world_width / 2,
              dx - np.sign(dx) * world_width, dx)
```

**Edge case**: If agent is exactly at `world_width / 2` distance:
- `np.abs(dx) == world_width / 2`, condition is False
- Correct distance chosen by luck (both paths give same answer)
- But `>` vs `>=` ambiguity could cause issue if world size is small

**Severity**: Very low (unlikely to matter)

---

### 27. Island Detection Near Path Endpoints

**Location**: `river.py` lines 85-89, 131-143

**Problem**:
```python
nearest_idx = np.argmin(distances)

# Check if in split region
t = nearest_idx / len(self.path_x)
if self.split_start <= t <= self.split_end:
    # Check if on island
```

**Issue**: Near the endpoints (t ≈ 0 or t ≈ 1), `nearest_idx` could be edge point, but island logic assumes it's in the middle of a segment

**Edge case**: Agents exactly at world boundary (0, 0) or (width, height)

**Severity**: Low (island is in middle 0.01-0.99 of river)

---

### 28. Possible NaN in Swim Speed

**Location**: `agent.py` lines 101-103, `simulation_gpu.py` lines 386-389

**Problem**:
```python
# CPU version:
child_swim_speed = max(0.1, self.swim_speed + np.random.randn() * mutation_rate * 2.0)

# What if self.swim_speed is NaN? max(0.1, NaN) = NaN
# What if mutation is very negative? max(0.1, X) should protect, but...
```

**Severity**: Low (requires earlier NaN propagation)

**Fix**: Add validation or clipping

---

## Documentation Gaps

### 29. No Docstrings for Key Algorithms

**Missing explanations**:
- Why use toroidal topology? (prevents edge effects, but not documented)
- Why sample observations in GPU version? (O(n²) → O(n), should be explained)
- Why mutate globally every 50 steps in GPU? (unclear if intentional)
- Why different speeds for prey/predators? (currently hardcoded with no rationale)

---

### 30. No Type Hints

**Example**: Every function lacks type hints
```python
def observe(self, predator_positions, predator_velocities, prey_positions, prey_velocities, my_index):
    # What types? What shapes? What happens if None?
```

**Impact**: IDE support reduced, unclear contracts

---

## Configuration Debt

### 31. Config Values Not Justified

Many constants lack justification:
- `PREY_MAX_SPEED = 3.0` - Why 3.0? What happens at 2.0 or 5.0?
- `CATCH_RADIUS = 8.0` - Why 8.0?
- `PRED_ENERGY_COST = 0.3` - Tuned empirically? Random guess?

**Fix**: Add comments or separate tuning guide

---

## Summary by Severity

### Critical (Must fix before extending):
- #1: Hardcoded two-species architecture
- #2: Fixed observation dimensions
- #3: Species-specific reproduction logic
- #4: GPU-CPU transfer bottleneck
- #5: Config system not extensible
- #6: Unclear brain-agent coupling in GPU

### Important (Will cause pain):
- #7-14: Code duplication, inconsistencies, half-implementations

### Performance (Optimization opportunities):
- #15-17: CPU list performance, unnecessary copies, inefficient checks

### Quality (Technical debt):
- #18-23: Naming, code organization, error handling

### Potential Bugs (Low severity):
- #24-28: Edge cases, race conditions, NaN propagation

### Documentation:
- #29-31: Missing rationales, no type hints, unjustified values

---

## Estimated Refactoring Effort

| Issue | Effort | Risk | Priority |
|-------|--------|------|----------|
| #1 Two-species hardcoding | Large | Medium | Critical |
| #2 Fixed observations | Medium | Low | Critical |
| #3 Reproduction logic | Medium | Low | Critical |
| #4 GPU-CPU transfers | Medium-Large | Medium | High |
| #5 Config system | Medium | Low | Critical |
| #6 GPU neuroevolution | Large | High | Critical |
| #7-14 Duplications | Small-Medium | Low | Medium |
| #15-17 Performance | Small | Low | Low |
| #18-23 Code quality | Small | Low | Low |
| #24-28 Potential bugs | Small | Low | Low |

**Total estimated effort**: 4-6 weeks of careful refactoring with tests

**Recommended approach**: Fix critical issues (#1-6) first, then evaluate if other issues block specific extensions.
