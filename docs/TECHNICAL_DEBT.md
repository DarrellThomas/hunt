# HUNT Technical Debt

This document catalogs technical debt in the HUNT system. Items are divided into **Resolved** (fixed during architecture refactoring) and **Active** (still need attention).

---

## ✅ RESOLVED ISSUES (Architecture Refactoring Jan 2026)

These issues were resolved during the comprehensive architecture refactoring (Phases 1-4):

### ✅ FIXED #1: Hardcoded Two-Species Architecture

**Status**: RESOLVED by Phase 2 (N-Species Architecture)

**Solution**: Created `species.py` with `SpeciesManager` class
- Replaced hardcoded `self.prey` and `self.predators` lists with flexible dictionary
- Species defined in configuration, not code
- Can add 3rd, 4th, 5th species via JSON config

**Files**: `src/species.py`, `tests/test_n_species.py`

---

### ✅ FIXED #2: Fixed Observation Dimensions

**Status**: RESOLVED by Phase 2 (Dynamic Sensor System)

**Solution**: Created `sensors.py` with dynamic observation system
- `SensorSuite` automatically calculates observation dimensions
- Can add/remove sensors without changing neural network code
- Each species can have different observation configuration

**Files**: `src/sensors.py`, `tests/test_sensors.py`

---

### ✅ FIXED #4: GPU-CPU Transfer Bottleneck

**Status**: RESOLVED by Phase 3.1 (GPU-Resident River)

**Solution**: Created `river_gpu.py` with fully GPU-resident river calculations
- Eliminated 4× CPU-GPU transfers per step
- `get_flow_at_batch_gpu()` stays on GPU
- `is_on_island_batch_gpu()` stays on GPU
- Expected 20-30% performance improvement

**Files**: `src/river_gpu.py`, `tests/test_river_gpu.py`

---

### ✅ FIXED #5: Configuration System Not Extensible

**Status**: RESOLVED by Phase 1.2 (Dataclass Configuration)

**Solution**: Created `config_new.py` with dataclass-based configuration
- Type-safe configuration with validation
- JSON serialization for experiment saving
- Per-species configuration support
- Factory methods: `default_two_species()`, `default_bounded()`

**Files**: `src/config_new.py`, `tests/test_config_new.py`

---

### ✅ FIXED #6: Unclear Brain-Agent Coupling in GPU Version

**Status**: RESOLVED (GPU Neuroevolution Fix, Jan 2026)

**Solution**: Implemented per-agent weight storage in GPU simulation
- Each agent has individual neural network weights
- Offspring inherit parent weights with mutation at reproduction
- Removed incorrect global mutation code
- True neuroevolution now working on GPU

**Files**: `src/simulation_gpu.py` (lines 90-110, 489-520)

---

### ✅ FIXED #7: Copy-Pasted Distance Calculations

**Status**: RESOLVED by Phase 1.1 (Shared Utilities)

**Solution**: Created `utils.py` with shared distance functions
- `toroidal_distance_numpy()` for CPU
- `toroidal_distance_torch()` for GPU
- `bounded_distance_numpy/torch()` for bounded mode
- Single source of truth, tested for NumPy/PyTorch consistency

**Files**: `src/utils.py`, `tests/test_utils.py`

---

### ✅ FIXED #8: Reproduction Position Calculation Duplicated

**Status**: RESOLVED by Phase 1.1 (Shared Utilities)

**Solution**: Created `spawn_offset()` function in `utils.py`
- Configurable min/max spawn distance
- Supports both NumPy and PyTorch
- Eliminates magic numbers (20, 150)

**Files**: `src/utils.py` (lines 128-158)

---

### ✅ FIXED #9: Inconsistent Extinction Prevention

**Status**: RESOLVED by Phase 1.3 (Unified Extinction Prevention)

**Solution**: Unified extinction prevention in both CPU and GPU
- `_handle_extinction_prevention()` method in both world.py and simulation_gpu.py
- Configurable parameters (enabled, respawn_count, minimum_population)
- Optional world-size scaling
- Identical behavior in both versions

**Files**: `src/world.py` (lines 169-205), `src/simulation_gpu.py` (lines 432-487)

---

### ✅ PARTIALLY FIXED #3: Species-Specific Reproduction Logic

**Status**: PARTIALLY RESOLVED by Phase 2 (Agent Factory Methods)

**What's Fixed**:
- `Agent.from_config()` factory methods for config-based creation
- Lifecycle parameters (lifespan, reproduction age) in config

**What's Still TODO**:
- Reproduction logic still hardcoded in Agent.can_reproduce()
- Could be generalized with strategy pattern or trait system

**Files**: `src/agent.py` (lines 237-270, 370-411)

---

### ✅ PARTIALLY FIXED #21: Magic String Literals

**Status**: PARTIALLY RESOLVED by Phase 2

**What's Fixed**:
- `AgentRole` enum added in `species.py`
- `BoundaryMode` enum in `config_new.py`

**What's Still TODO**:
- River code still uses `'prey'` and `'predator'` string literals
- Not all code migrated to use enums yet

---

### ✅ PARTIALLY FIXED #23: Inconsistent Error Handling

**Status**: PARTIALLY RESOLVED

**What's Fixed**:
- `config_new.py` has comprehensive validation
- Config system raises clear errors for invalid configurations

**What's Still TODO**:
- No GPU availability check before using CUDA
- No validation of agent state (positions, velocities)
- Analysis scripts lack error handling

---

### ✅ PARTIALLY FIXED #29 & #30: Documentation and Type Hints

**Status**: PARTIALLY RESOLVED

**What's Fixed**:
- All new modules have comprehensive docstrings
- All new modules have type hints
- 3 new documentation guides (ADDING_SPECIES, BOUNDARY_MODES, RUNNING_EXPERIMENTS)
- Updated README with architecture overview

**What's Still TODO**:
- Legacy modules (agent.py, world.py, brain.py) lack type hints
- Some algorithms still lack rationale documentation

---

## ⚠️ ACTIVE TECHNICAL DEBT

These issues remain and should be addressed in future work:

---

## Important Issues (Will Cause Pain Later)

### #10: Statistics Tracking Inconsistency

**Location**: `world.py` lines 44-51, `main_gpu.py` lines 40-55

**Problem**:
- CPU tracks 6 metrics: counts, avg_age, avg_fitness
- GPU tracks 12 metrics: + swim speeds, habitat preferences
- Different save locations: `stats.npz` vs `stats_autosave.npz`
- GPU autosaves, CPU only saves on demand

**Impact**: Analysis scripts need to handle two formats

**Fix**: Unified stats system with versioning

---

### #11: River Path Generation Has Magic Numbers

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

**Impact**: Difficult to customize river generation

**Fix**: Document parameters, make configurable

---

### #12: Island Behavior System Half-Implemented

**Location**: `river.py` lines 195-223, `simulation_gpu.py` lines 258-324

**Problem**: Island behavior modifiers exist but are not used in CPU version

```python
# river.py defines island_behavior()
def island_behavior(self, agent_type, x, y):
    # Returns modifiers for speed, hunger, reproduction

# simulation_gpu.py uses it
island_mods = self.get_island_modifiers(...)

# world.py DOES NOT use it
# Agents on island still behave normally
```

**Impact**:
- CPU and GPU have different island mechanics
- Inconsistent experiment results

**Fix**: Integrate island modifiers into world.py

---

### #13: No Validation of Agent State

**Location**: Throughout `agent.py`, `world.py`, `simulation_gpu.py`

**Problem**: No checks for invalid states:
- Position can NaN if physics glitches
- Velocity can exceed speed limits between steps
- Energy can go slightly negative

**Example failure case**:
```python
# If acceleration is very large due to bug:
vel += acc * dt  # vel could explode
pos += vel * dt  # pos could become NaN
distances = sqrt((pos1 - pos2)**2)  # NaN propagates
```

**Impact**: Silent failures that are hard to debug

**Fix**: Add assertions or validation in debug mode

---

### #14: Observation Sampling In GPU Inconsistent

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

### #15: CPU Version Uses Lists for Agents

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

### #16: Unnecessary Array Copies in CPU Version

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

### #17: GPU Extinction Prevention Inefficient

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

### #18: Inconsistent Naming Conventions

**Examples**:
- `prey_count` vs `num_prey` vs `initial_prey`
- `predator_count` vs `num_predators` vs `initial_predators`
- `max_speed` vs `prey_max_speed` vs `PREY_MAX_SPEED`
- `pred_*` vs `predator_*` (abbreviated inconsistently)

**Impact**: Harder to search, cognitive load

**Note**: New code (config_new.py, species.py, etc.) uses consistent naming

---

### #19: Poor Function Naming

**Location**: `simulation_gpu.py` line 258

**Problem**:
```python
def get_island_modifiers(self, positions, agent_type):
    # Actually checks if on island AND returns modifiers
    # Name suggests it always returns modifiers
```

Better name: `compute_island_effects()` or `apply_island_modifiers()`

---

### #20: Unclear Variable Scope in GPU Step

**Location**: `simulation_gpu.py` lines 326-482

**Problem**: The `step()` method is 156 lines with variables used across large spans

**Impact**: Hard to track variable lifespan, easy to reuse stale values

**Fix**: Break into smaller methods: `_update_physics()`, `_handle_reproduction()`, etc.

---

### #22: Commented-Out Code and TODOs

**Status**: Clean - no commented-out code or stale TODOs found

---

## Potential Bugs (Low Severity)

### #24: Race Condition in Reproduction

**Location**: `world.py` lines 151-167

**Problem**:
```python
new_prey = []
for prey in self.prey:
    if prey.can_reproduce():
        child = prey.reproduce(mutation_rate)
        new_prey.append(child)
        prey.time_since_reproduction = 0  # Modify during iteration

self.prey.extend(new_prey)
```

**Issues**: Modifying agent state during iteration over same list

**Severity**: Low (unlikely to trigger)

**Fix**: Separate reproduction checks from state modification

---

### #25: GPU Reproduction May Overwrite Living Agents

**Location**: `simulation_gpu.py` lines 484-492

**Issue**: If more agents die after `dead_prey_idx` is computed, those indices might not be dead anymore

**Severity**: Low (deaths happen before reproduction in step order)

**Actual Risk**: Safe due to step ordering, but fragile if step order changes

---

### #26: Toroidal Distance Edge Case

**Location**: `utils.py` (now centralized)

**Problem**:
```python
dx = np.where(np.abs(dx) > world_width / 2,
              dx - np.sign(dx) * world_width, dx)
```

**Edge case**: If agent is exactly at `world_width / 2` distance

**Severity**: Very low (unlikely to matter)

---

### #27: Island Detection Near Path Endpoints

**Location**: `river.py` lines 85-89, 131-143

**Issue**: Near the endpoints (t ≈ 0 or t ≈ 1), `nearest_idx` could be edge point

**Severity**: Low (island is in middle 0.01-0.99 of river)

---

### #28: Possible NaN in Swim Speed

**Location**: `agent.py` lines 101-103

**Problem**:
```python
child_swim_speed = max(0.1, self.swim_speed + np.random.randn() * mutation_rate * 2.0)
# What if self.swim_speed is NaN? max(0.1, NaN) = NaN
```

**Severity**: Low (requires earlier NaN propagation)

**Fix**: Add validation or clipping

---

## Configuration Debt

### #31: Config Values Not Justified

Many constants in `config.py` lack justification:
- `PREY_MAX_SPEED = 3.0` - Why 3.0? What happens at 2.0 or 5.0?
- `CATCH_RADIUS = 8.0` - Why 8.0?
- `PRED_ENERGY_COST = 0.3` - Tuned empirically? Random guess?

**Note**: New config system (`config_new.py`) documents parameters better, but legacy config still used

**Fix**: Add comments or separate tuning guide

---

## Summary

### ✅ Resolved Issues (Architecture Refactoring)
- **Critical (6)**: #1, #2, #4, #5, #6 fully fixed; #3 partially fixed
- **Important (2)**: #7, #8, #9 fully fixed
- **Code Quality (2)**: #21, #23, #29, #30 partially fixed

### ⚠️ Active Technical Debt

**Important issues** (4):
- #10: Statistics tracking inconsistency
- #11: River magic numbers
- #12: Island behavior half-implemented (CPU)
- #13: No agent state validation
- #14: GPU observation sampling asymmetry

**Performance issues** (3):
- #15: CPU uses lists (acceptable, GPU exists)
- #16: Unnecessary array copies (CPU)
- #17: GPU extinction checks inefficient

**Code quality** (3):
- #18: Inconsistent naming (legacy code)
- #19: Poor function naming
- #20: Large step() method

**Potential bugs** (5):
- #24-28: Various edge cases (all low severity)

**Configuration** (1):
- #31: Config values not justified

---

## Recommendation

The architecture refactoring successfully addressed the **6 critical blockers** to extensibility. Remaining issues are:

1. **Important but not blocking**: Can add species, run experiments, extend system
2. **Performance optimizations**: Nice-to-have but not required
3. **Code quality**: Gradual improvement as code is touched
4. **Low-severity bugs**: Can be addressed if/when they cause problems

**Next Priority**: If extending the system further, address #10 (stats), #12 (island behavior), and #13 (validation) for consistency and robustness.

**Overall Status**: System is now production-ready for N-species simulations with good test coverage and documentation.
