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
│   ├── agent.py       # Prey and Predator classes
│   ├── brain.py       # Neural network
│   ├── config.py      # Configuration parameters
│   ├── river.py       # River/island environment
│   ├── world.py       # CPU simulation
│   ├── simulation_gpu.py  # GPU simulation
│   ├── main.py        # CPU visualizer
│   ├── main_gpu.py    # GPU visualizer
│   └── analyze_*.py   # Analysis scripts
├── docs/              # Documentation
│   ├── THESIS.md      # Design philosophy
│   ├── ARCHITECTURE.md     # System architecture
│   ├── TECHNICAL_DEBT.md   # Known issues
│   └── ...
├── results/           # Output files
│   ├── *.npz          # Statistics data
│   └── *.png          # Visualizations
├── tests/             # Test files
├── run.py             # Convenience wrapper (CPU)
└── run_gpu.py         # Convenience wrapper (GPU)
```

See [docs/](docs/) for complete documentation.

## Controls

- **SPACE** - Pause/Resume
- **S** - Save statistics
- **ESC** - Quit

## Features

- Neural network brains evolve through natural selection
- River/island environmental system
- GPU acceleration for 10,000+ agents
- Automatic data collection and analysis
- Individual trait evolution (swim speed, lifespans)

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

- [THESIS.md](docs/THESIS.md) - Design philosophy
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - Complete system architecture
- [QUICKSTART.md](docs/QUICKSTART.md) - Detailed setup guide
- [RESULTS.md](docs/RESULTS.md) - Experimental results

Built with emergent complexity and survival of the fittest.
