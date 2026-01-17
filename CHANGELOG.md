# Changelog

## 2026-01-17 - Architecture Refactoring Phase 1 (Foundation)

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
