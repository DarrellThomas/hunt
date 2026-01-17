# HUNT - Predator-Prey Co-Evolution

A neuroevolution simulation where predators and prey both learn to survive through natural selection.

## Quick Start

**🎯 New to this? → [EZ_README.md](EZ_README.md) - Step-by-step Windows setup guide for beginners**

**Cross-platform (Windows/macOS/Linux):**

```bash
# CPU version (recommended for < 1000 agents)
python run.py

# GPU version (recommended for 1000+ agents)
python run_gpu.py
```

**Overnight training:**
- Linux/macOS: `./start_overnight.sh`
- Windows (CMD): `start_overnight.bat`
- Windows (PowerShell): `.\start_overnight.ps1`

**Or run directly from src/:**

```bash
cd src
python main.py          # CPU version
python main_gpu.py      # GPU version
```

## Project Structure

```
hunt/
├── src/               # Source code
│   ├── agent.py       # Agent classes with config-based creation
│   ├── brain.py       # Neural network implementation
│   ├── config.py      # Legacy configuration (deprecated)
│   ├── config_new.py  # New dataclass configuration system ⭐
│   ├── species.py     # N-species management ⭐
│   ├── sensors.py     # Dynamic sensor system ⭐
│   ├── traits.py      # Generic trait evolution system ⭐
│   ├── utils.py       # Shared utilities (distance, spawn) ⭐
│   ├── river.py       # CPU river implementation
│   ├── river_gpu.py   # GPU-resident river ⭐
│   ├── world.py       # CPU simulation
│   ├── simulation_gpu.py  # GPU simulation
│   ├── main.py        # CPU visualizer
│   ├── main_gpu.py    # GPU visualizer
│   └── analyze_*.py   # Analysis scripts
├── docs/              # Documentation
│   ├── THESIS.md      # Design philosophy
│   ├── ARCHITECTURE.md     # System architecture
│   ├── ADDING_SPECIES.md   # How to add new species ⭐
│   ├── BOUNDARY_MODES.md   # Toroidal vs bounded worlds ⭐
│   ├── RUNNING_EXPERIMENTS.md  # Experiment guide ⭐
│   ├── TECHNICAL_DEBT.md   # Known issues
│   └── ...
├── tests/             # Comprehensive test suite (66 tests) ⭐
│   ├── test_utils.py       # Distance/spawn utilities
│   ├── test_config_new.py  # Configuration validation
│   ├── test_sensors.py     # Sensor behaviors
│   ├── test_species.py     # Multi-species management
│   ├── test_boundary_modes.py  # Toroidal vs bounded
│   ├── test_river_gpu.py   # GPU river correctness
│   ├── test_traits.py      # Trait evolution system
│   └── test_integration.py # Full simulation tests
├── results/           # Output files
│   ├── *.npz          # Statistics data
│   └── *.png          # Visualizations
├── prompts/           # AI-assisted refactoring prompts
├── run.py             # Convenience wrapper (CPU)
└── run_gpu.py         # Convenience wrapper (GPU)
```

⭐ = New in architecture refactoring (Jan 2026)

See [docs/](docs/) for complete documentation.

## Controls

- **SPACE** - Pause/Resume
- **S** - Save statistics
- **ESC** - Quit

## Features

- **Neural network brains** evolve through natural selection
- **N-species architecture** - Add new species via configuration (no code changes!)
- **Boundary modes** - Toroidal (wrap-around) or bounded (walled) worlds
- **GPU acceleration** - Handle 10,000+ agents at 50k+ steps/second
- **Dynamic sensors** - Each species can observe different things
- **River/island environment** - GPU-resident for maximum performance
- **Generic trait system** - Evolve any numeric property
- **Comprehensive tests** - 66 tests covering all subsystems
- **Automatic data collection** and analysis

### Recent Improvements (Jan 2026)

The project underwent a comprehensive architecture refactoring:

1. **Phase 1: Foundation**
   - Shared utilities for distance calculations (toroidal/bounded)
   - Dataclass-based configuration system
   - Unified extinction prevention

2. **Phase 2: N-Species Architecture**
   - Add species via JSON configuration
   - Dynamic sensor system
   - Factory methods for config-based agent creation

3. **Phase 3: Optimization**
   - GPU-resident river (20-30% speedup)
   - Generic trait evolution framework

4. **Phase 4: Quality**
   - 66 comprehensive tests
   - Documentation guides for common tasks

See [CHANGELOG.md](CHANGELOG.md) for complete refactoring history.

## Requirements

**Cross-Platform Support:**
- ✅ Windows 10/11
- ✅ macOS (Intel/Apple Silicon)
- ✅ Linux

**Dependencies:**
- Python 3.8+
- NumPy, Pygame, Matplotlib
- PyTorch + CUDA (for GPU version, optional)

## Documentation

### Getting Started
- [QUICKSTART.md](docs/QUICKSTART.md) - Detailed setup guide
- [RUNNING_EXPERIMENTS.md](docs/RUNNING_EXPERIMENTS.md) - How to run experiments ⭐

### Core Concepts
- [THESIS.md](docs/THESIS.md) - Design philosophy
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System architecture
- [BOUNDARY_MODES.md](docs/BOUNDARY_MODES.md) - Toroidal vs bounded worlds ⭐

### Extending the System
- [ADDING_SPECIES.md](docs/ADDING_SPECIES.md) - How to add new species ⭐
- [EXTENSION_POINTS.md](docs/EXTENSION_POINTS.md) - Extensibility analysis

### Results and Analysis
- [RESULTS.md](docs/RESULTS.md) - Experimental results
- [CHANGELOG.md](CHANGELOG.md) - Complete development history

⭐ = New guides added in Jan 2026

Built with emergent complexity and survival of the fittest.
