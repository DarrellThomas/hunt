# Changelog

## 2026-01-17 - Architecture Refactoring Phase 3 (Optimization) ✅ COMPLETE

### Added - Phase 3.2: Generic Trait System
- **traits.py**: Flexible evolvable trait system (250 lines)
  - `Trait` dataclass: Generic definition for any evolvable property
    - `sample_initial()`: Generate initial trait value from normal distribution
    - `mutate()`: Apply Gaussian mutation with configurable std
    - Automatic bounds enforcement (min/max clamping)
  - `TraitCollection`: Manage multiple traits per agent type
    - `sample_initial_values()`: Initialize all traits for new agent
    - `mutate_values()`: Mutate all traits during reproduction
  - `COMMON_PREY_TRAITS`: Predefined traits (max_speed, swim_speed, max_acceleration)
  - `COMMON_PREDATOR_TRAITS`: Predefined traits (includes max_energy)
- **tests/test_traits.py**: 8 comprehensive tests (all passing)
  - Tests trait initialization
  - Tests initial value sampling (normal distribution)
  - Tests mutation with proper scaling
  - Tests bounds clamping
  - Tests TraitCollection management
  - Tests common trait definitions
  - Tests multi-generation inheritance
  - Tests mutation rate scaling

### Benefits - Phase 3.2: Trait System
- **Easy extensibility**: Add new evolvable traits without modifying agent code
- **Consistent evolution**: All traits use same mutation/inheritance logic
- **Configuration-driven**: Traits defined declaratively with clear parameters
- **Type-safe**: Dataclass provides validation and IDE support
- **Testable**: Generic system easier to test than agent-specific code
- **Flexible**: Can define traits for any numeric property (speed, vision, stamina, etc.)

### Usage Example - Phase 3.2
```python
# Define custom traits
herbivore_traits = TraitCollection({
    'speed': Trait('speed', 2.5, 0.2, 0.15, 0.5, 8.0),
    'vision': Trait('vision', 100.0, 10.0, 15.0, 20.0, 300.0),
    'stamina': Trait('stamina', 100.0, 10.0, 8.0, 20.0, 200.0),
})

# Create initial agent
initial_values = herbivore_traits.sample_initial_values()
agent_speed = initial_values['speed']  # Use in agent

# Reproduce with mutation
child_values = herbivore_traits.mutate_values(initial_values, mutation_rate=0.1)
```

### Next Steps - Phase 3.2
The trait system is ready to use but not yet integrated into agent classes.
Future work could optionally integrate traits into Agent/Prey/Predator classes
to replace hardcoded evolvable properties like swim_speed.

---

## 2026-01-17 - Architecture Refactoring Phase 3.1 (GPU River) ✅ COMPLETE

### Added - Phase 3.1: GPU-Resident River System
- **river_gpu.py**: Fully GPU-accelerated river implementation (230 lines)
  - `RiverGPU` class: All river calculations performed on GPU
  - `get_flow_at_batch_gpu()`: Vectorized flow computation using `torch.cdist`
  - `is_in_river_batch_gpu()`: GPU-resident river boundary detection
  - `is_on_island_batch_gpu()`: GPU-resident island detection
  - Eliminates all CPU-GPU transfers for river operations
  - Identical behavior to CPU river implementation
- **tests/test_river_gpu.py**: 7 comprehensive tests (all passing)
  - Tests GPU initialization
  - Tests flow consistency between CPU and GPU (max error < 1e-5)
  - Tests is_in_river consistency (100% match rate)
  - Tests is_on_island consistency (100% match rate)
  - Tests island has no flow
  - Tests flow direction correctness
  - Tests batched performance (1000+ agents)

### Changed - Phase 3.1: GPU Simulation Integration
- **simulation_gpu.py**: Updated to use RiverGPU
  - Import changed: `from river_gpu import RiverGPU`
  - Initialization: `RiverGPU(width, height, device=device)`
  - `get_island_modifiers()`: Now uses `is_on_island_batch_gpu()` (no CPU transfer)
  - Prey flow application: Direct GPU tensor operations (no `.cpu().numpy()`)
  - Predator flow application: Direct GPU tensor operations
  - Statistics gathering: GPU-resident island/river classification
  - All river operations now stay on GPU

### Performance Impact - Phase 3.1
- **Eliminated CPU-GPU transfers**: River checks no longer bounce data between devices
- **Vectorized operations**: `torch.cdist` replaces per-position loops
- **Expected speedup**: 20-30% for simulations with river enabled
- **Memory efficiency**: River path stored once on GPU, no repeated transfers
- **Scalability**: Batched operations handle thousands of agents efficiently

### Technical Details - Phase 3.1
- Uses `torch.cdist()` for fast distance calculations to river path points
- Island detection uses t-parameter along river path (0 to 1)
- Split channel logic preserved from CPU version
- Flow direction computed from path tangent vectors
- All boolean masks remain on GPU for downstream operations

---

## 2026-01-17 - Architecture Refactoring Phase 2 (N-Species) ✅ COMPLETE

### Added - Phase 2.2: Dynamic Sensor System
- **sensors.py**: Flexible sensor system for agent observations (270 lines)
  - `Sensor` abstract base class
  - `NearestAgentsSensor`: Observe nearest N agents of any species
  - `HungerSensor`: Sense own energy/hunger level
  - `IslandProximitySensor`: Detect if on island
  - `WallProximitySensor`: Sense distance to walls (for bounded mode)
  - `SensorSuite`: Combine multiple sensors into observation vector
  - `SensorSuite.from_config()`: Create sensors from ObservationConfig
- Sensors automatically calculate correct observation dimensions
- Supports observing any combination of species

### Added - Phase 2.1: N-Species Architecture
- **species.py**: Multi-species management system (180 lines)
  - `AgentRole` enum: PREY, PREDATOR, SCAVENGER, PRODUCER
  - `InteractionResult`: Track predator-prey interaction outcomes
  - `SpeciesManager`: Manage arbitrary numbers of species
    - Replaces hardcoded prey/predator lists with flexible dictionary
    - `initialize_populations()`: Create species from config
    - `get_all_positions/velocities()`: Extract data for observations
    - `get_species()`, `get_config()`: Access species data
    - `remove_dead()`, `add_agent()`: Population management
    - `stats_summary()`: Get population counts per species
- Fully dynamic - no hardcoded species names

### Changed - Phase 2: Agent System Updates
- **agent.py**: Updated for config-based creation
  - Added `sensor_suite` parameter to Prey and Predator `__init__`
  - Added `Prey.from_config()` class method
  - Added `Predator.from_config()` class method
  - Agents now support dynamic observation dimensions
  - Backward compatible with existing code

### Added - Phase 2: Tests
- **tests/test_n_species.py**: 5 comprehensive tests (all passing)
  - Tests SpeciesManager initialization
  - Tests SensorSuite creation from config
  - Tests agent creation from SpeciesConfig
  - Tests position/velocity extraction
  - Tests stats summary generation

### Impact - Phase 2
This enables true N-species simulations:
- **Add new species** without modifying code (just update config)
- **Custom observations** per species (each sees different things)
- **Flexible interactions** (any species can interact with any other)
- **Extensible sensors** (add new sensor types easily)
- **No hardcoded limits** on species count

Example: Can now create 3+ species ecosystems:
- Herbivores (eat plants), Predators (eat herbivores), Apex predators (eat predators)
- All configured via SimulationConfig JSON

---

## 2026-01-17 - Architecture Refactoring Phase 1 (Foundation) ✅ COMPLETE

### Added - Phase 1.3: Unified Extinction Prevention
- **world.py**: Added `_handle_extinction_prevention()` method
  - Emergency respawn at population = 0
  - Gradual repopulation below minimum threshold
  - Configurable parameters (enabled, respawn_count, minimum_population, scale_with_world_size)
  - Unified logic replaces previous hardcoded values
- **simulation_gpu.py**: Added matching `_handle_extinction_prevention()` method
  - GPU version now has minimum population threshold (previously only had emergency respawn)
  - Added `_respawn_prey()` and `_respawn_predators()` helper methods
  - Identical behavior to CPU version
- Both versions now support:
  - Emergency extinction prevention (respawn N agents when population hits 0)
  - Minimum population threshold (gradually add agents when below minimum)
  - World-size scaling (minimum population scales with world area)
  - Configurable enable/disable

### Changed - Phase 1.3
- **world.py**: Refactored extinction prevention from inline code to configurable method
- **simulation_gpu.py**: Refactored and enhanced extinction prevention logic
- Both versions now have identical extinction prevention behavior

### Added - Phase 1.2: Dataclass Configuration System
- **config_new.py**: Complete dataclass-based configuration system
  - `BoundaryMode` enum: Toroidal (wrap-around) vs Bounded (walled) worlds
  - `PhysicsConfig`, `LifecycleConfig`, `EnergyConfig`: Species parameters
  - `ObservationConfig`: Dynamic sensor configuration
  - `SpeciesConfig`: Complete species definition
  - `RiverConfig`: River and island parameters
  - `ExtinctionPreventionConfig`: Population management
  - `WorldConfig`: World-level parameters with boundary mode support
  - `InteractionConfig`: Species interaction definitions
  - `SimulationConfig`: Top-level configuration with validation
  - JSON serialization/deserialization support
  - Factory methods: `default_two_species()`, `default_bounded()`
- **tests/test_config_new.py**: 10 comprehensive unit tests (all passing)

### Added - Phase 1.1: Shared Utilities
- **utils.py**: Unified distance and position utilities
  - `toroidal_distance_numpy/torch`: Wrap-around distance calculations
  - `bounded_distance_numpy/torch`: Walled world distance calculations
  - `spawn_offset`: Random offspring placement
  - `wrap_position_numpy/torch`: Position wrapping for toroidal worlds
  - `clamp_position_numpy/torch`: Position clamping for bounded worlds
- **tests/test_utils.py**: 7 unit tests verifying NumPy/PyTorch consistency

### Changed
- **agent.py**: Now uses `toroidal_distance_numpy` from utils.py
- **simulation_gpu.py**: Now uses `toroidal_distance_torch` from utils.py

### Impact
This foundation enables:
- **Boundary mode selection**: Users can choose toroidal vs bounded worlds
- **N-species support**: Configuration system ready for arbitrary species count
- **Experiment saving**: JSON serialization for reproducible experiments
- **Validation**: Automatic checking of configuration validity
- **Code reuse**: Shared utilities reduce duplication

---

## 2026-01-17 - CRITICAL FIX: GPU Neuroevolution

### Fixed
- **🔴 CRITICAL BUG**: Fixed fundamental neuroevolution algorithm in GPU version
  - **Before**: All agents shared ONE global brain that mutated every 50 steps
  - **After**: Each agent has individual neural network weights
  - **Before**: Offspring did NOT inherit parent weights (no genetic algorithm)
  - **After**: Offspring properly inherit parent weights with mutation at reproduction
  - **Before**: Evolution was fake - all agents had identical brains
  - **After**: True neuroevolution - successful agents pass their traits to offspring

### Changed
- Added per-agent weight storage: `prey_weights` (100×2178), `pred_weights` (20×1826)
- Implemented `_calc_weight_count()` to calculate network parameter counts
- Implemented `_batch_forward()` for efficient batched inference with per-agent weights
- Updated reproduction logic to copy parent weights and apply mutation
- Removed incorrect global mutation code (lines 620-622 in old version)
- Updated emergency respawn to initialize random weights for new agents

### Technical Details
- Prey network: 32→32→32→2 = **2,178 parameters per agent**
- Predator network: 21→32→32→2 = **1,826 parameters per agent**
- Batched matrix multiply (`torch.bmm`) maintains GPU performance
- Weight inheritance: `child_weights = parent_weights.clone() + mutation`
- Mutation occurs ONLY at reproduction, not globally

### Impact
This fix enables **true evolutionary dynamics** in the GPU version:
- Agents that survive longer pass their successful behaviors to offspring
- Population evolves better hunting/evasion strategies over generations
- Natural selection drives improvement (previously impossible with shared brain)

---

## 2026-01-17 - Project Restructuring

### Added
- **Folder structure**: Organized codebase into `src/`, `docs/`, `results/`, `tests/`
- **Wrapper scripts**: `run.py` and `run_gpu.py` for convenient execution from project root
- **Updated README.md**: New root-level README with project structure overview

### Changed
- **Moved all source code** to `src/` directory
- **Moved all documentation** to `docs/` directory
- **Moved all result files** (.npz, .png) to `results/` directory
- **Moved test files** to `tests/` directory
- **Updated start_overnight.sh**: Now runs from src/ directory

### Migration Guide

**Before:**
```bash
python3 main.py
python3 main_gpu.py
```

**After (from root):**
```bash
python3 run.py
python3 run_gpu.py
```

**Or (from src/):**
```bash
cd src
python3 main.py
python3 main_gpu.py
```

All imports still work within `src/` directory. Wrapper scripts handle path setup for root execution.

---

## 2026-01-17 - Complete Platform Audit

### Added
- **ARCHITECTURE.md**: Complete system architecture documentation
- **TECHNICAL_DEBT.md**: 31 catalogued technical debt items
- **EXTENSION_POINTS.md**: Extensibility analysis
- **REFACTOR_PLAN.md**: Prioritized refactoring roadmap (6-8 weeks)
- **EXTENSION_ARCHITECTURE.md**: Concrete examples for future extensions
- **QUESTIONS.md**: Design clarification questions

### Findings
- Identified hardcoded 2-species architecture as main extensibility blocker
- Discovered GPU neuroevolution differs from CPU implementation
- Documented 6-8 week refactoring path to N-species support
- All existing functionality stable (no critical bugs)

---

## Earlier History

See git log for full commit history.
