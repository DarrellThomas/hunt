let# HUNT - Quick Start Guide

## Run the Simulation

### Option 1: CPU Version (Recommended for <1000 agents)
**1600x1200 resolution, 240 agents, 91 FPS**
```bash
python3 main.py
```

### Option 2: GPU Version - 4K FULL SCREEN (Recommended for maximum scale)
**3840x2160 resolution, 12,000 agents, 14 FPS**
```bash
python3 main_gpu.py
```

## Controls

- **SPACE** - Pause/Resume simulation
- **S** - Save statistics to `stats.npz`
- **ESC** - Quit

## What You'll See

- 🟢 **Green dots** = Prey (learning to evade)
- 🔴 **Red dots** = Predators (learning to hunt)
- **Bottom panel** = Statistics (population, age, fitness, FPS)

## The Evolution

Watch as both populations evolve through natural selection:
- Predators learn hunting strategies
- Prey develop evasion behaviors
- Population cycles emerge naturally
- Fitness increases over generations

## Customize Parameters

Edit `main.py` or `main_gpu.py` to adjust:

```python
# World size
world = World(width=1600, height=1200, ...)

# Population
initial_prey=200, initial_predators=40

# Mutation rate (how fast evolution happens)
mutation_rate=0.1
```

## Analyze Results

After running, analyze the saved statistics:

```bash
python3 analyze_stats.py
```

This generates `ecosystem_stats.png` with:
- Population dynamics over time
- Average age trends
- Fitness evolution
- Predator/prey ratio

## Requirements

**CPU Version:**
- Python 3.8+
- NumPy
- Pygame
- Matplotlib

**GPU Version:**
- All of the above plus:
- PyTorch with CUDA
- NVIDIA GPU (tested on RTX 5090)

## Performance Tips

**For smooth performance:**
- CPU version: Use for up to 500 agents
- GPU version: Use for 1,000+ agents
- Reduce world size if FPS drops below 15
- Lower `steps_per_frame` if rendering is slow

**For faster evolution:**
- Increase `steps_per_frame` (simulation runs faster than display)
- Increase `mutation_rate` (more variation per generation)
- Decrease population size (faster iterations)

## Project Structure

```
hunt/
├── main.py              # CPU visualization (1600x1200)
├── main_gpu.py          # GPU visualization (3840x2160, 4K)
├── brain.py             # Neural network with mutation
├── agent.py             # Prey and Predator classes
├── world.py             # CPU ecosystem simulation
├── simulation_gpu.py    # GPU ecosystem simulation
├── analyze_stats.py     # Statistics visualization
├── THESIS.md           # Design philosophy
├── README.md           # Project overview
└── RESULTS.md          # Experimental results
```

## Troubleshooting

**"ModuleNotFoundError: No module named 'torch'"**
- GPU version only: Install PyTorch with `pip install torch`
- Or use CPU version: `python3 main.py`

**Simulation is laggy:**
- Reduce agent count
- Use smaller world
- Try CPU version for fewer agents
- Check GPU memory with `nvidia-smi`

**Window doesn't fit screen:**
- Edit world dimensions in `main.py` or `main_gpu.py`
- Common sizes: 800x600, 1280x720, 1920x1080, 2560x1440, 3840x2160

## Quick Benchmark

Test performance on your hardware:

```bash
# Test CPU version
python3 test_large_world.py

# Test GPU version (4K)
python3 test_4k.py
```

---

**Ready to watch evolution in action?**

```bash
python3 main_gpu.py
```

Let the hunt begin! 🏹
