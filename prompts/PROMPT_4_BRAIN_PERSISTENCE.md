# HUNT Platform: Brain Weight Persistence & Analysis Titles

**Budget**: ~$15
**Scope**: Add brain weight saving/loading and CLI title flags for analysis

## Overview

Two features needed:
1. Save/load brain weights so evolved intelligence persists across sessions
2. Add title flags to analysis scripts for labeling experiments

---

## Feature 1: Brain Weight Persistence

### The Problem

Currently the simulation saves stats (population, swim speed, habitat %) but **NOT the actual neural network weights**. When the sim stops, all evolved brains are lost.

After 545K timesteps of evolution, the learned behaviors exist only in GPU memory. We need to persist them.

### Requirements

#### 1.1 Auto-save weights periodically

Add to the existing autosave in `main_gpu.py`:

```python
# Every N steps, save brain weights alongside stats
if self.timestep % 10000 == 0:  # Less frequent than stats (every 1000)
    self.save_brain_weights()
```

#### 1.2 Save weights on exit

When simulation ends (window close, Ctrl+C), save final brain state:

```python
# In main loop cleanup
finally:
    ecosystem.save_brain_weights(f'brains_final_{ecosystem.timestep}.npz')
    renderer.close()
```

#### 1.3 CLI for manual save with custom name

```bash
# During simulation, press 'B' to save brains with timestamp
# Or provide name via command line flag when starting:
python main_gpu.py --brain-checkpoint=island_refuge
# Saves to: brains_island_refuge.npz
```

#### 1.4 Implementation in `simulation_gpu.py`

```python
def save_brain_weights(self, name=None):
    """Save all agent brain weights to disk.
    
    Args:
        name: Optional name for the save file. If None, uses timestamp.
    """
    if name is None:
        filename = f'brains_step_{self.timestep}.npz'
    else:
        # Clean the name, ensure .npz extension
        name = name.replace(' ', '_')
        if not name.endswith('.npz'):
            filename = f'brains_{name}.npz'
        else:
            filename = name
    
    np.savez(filename,
        # Brain weights
        prey_weights=self.prey_weights.cpu().numpy(),
        pred_weights=self.pred_weights.cpu().numpy(),
        
        # Metadata for reconstruction
        timestep=self.timestep,
        prey_alive=self.prey_alive.cpu().numpy(),
        pred_alive=self.pred_alive.cpu().numpy(),
        
        # Traits (so we can restore full agent state)
        prey_swim_speed=self.prey_swim_speed.cpu().numpy(),
        pred_swim_speed=self.pred_swim_speed.cpu().numpy(),
        
        # Positions (optional, for visualization)
        prey_pos=self.prey_pos.cpu().numpy(),
        pred_pos=self.pred_pos.cpu().numpy(),
    )
    print(f"Brain weights saved to {filename}")
    print(f"  Prey brains: {self.prey_alive.sum().item()} alive")
    print(f"  Predator brains: {self.pred_alive.sum().item()} alive")


def load_brain_weights(self, filename):
    """Load agent brain weights from disk.
    
    Args:
        filename: Path to the .npz file
    """
    data = np.load(filename)
    
    self.prey_weights = torch.tensor(data['prey_weights'], device=self.device)
    self.pred_weights = torch.tensor(data['pred_weights'], device=self.device)
    
    # Restore alive masks
    self.prey_alive = torch.tensor(data['prey_alive'], device=self.device)
    self.pred_alive = torch.tensor(data['pred_alive'], device=self.device)
    
    # Restore traits
    self.prey_swim_speed = torch.tensor(data['prey_swim_speed'], device=self.device)
    self.pred_swim_speed = torch.tensor(data['pred_swim_speed'], device=self.device)
    
    # Restore positions
    self.prey_pos = torch.tensor(data['prey_pos'], device=self.device)
    self.pred_pos = torch.tensor(data['pred_pos'], device=self.device)
    
    print(f"Brain weights loaded from {filename}")
    print(f"  Timestep: {data['timestep']}")
    print(f"  Prey: {self.prey_alive.sum().item()} alive")
    print(f"  Predators: {self.pred_alive.sum().item()} alive")
```

#### 1.5 CLI flag for main_gpu.py

```python
import argparse

parser = argparse.ArgumentParser(description='HUNT GPU Simulation')
parser.add_argument('--load-brains', type=str, help='Load brain weights from file')
parser.add_argument('--brain-checkpoint', type=str, default=None,
                    help='Name prefix for brain checkpoint saves')
args = parser.parse_args()

# In main():
if args.load_brains:
    ecosystem.load_brain_weights(args.load_brains)
    print(f"Resuming from {args.load_brains}")
```

#### 1.6 Keyboard shortcut during simulation

```python
# In event handling loop
if event.type == pygame.KEYDOWN:
    if event.key == pygame.K_b:
        # Save brains with timestamp
        ecosystem.save_brain_weights()
    if event.key == pygame.K_n:
        # Save brains with custom name prompt (or use checkpoint name)
        name = args.brain_checkpoint or f'manual_{ecosystem.timestep}'
        ecosystem.save_brain_weights(name)
```

---

## Feature 2: Analysis Script Title Flag

### The Problem

`analyze_evolution.py` generates graphs but has no way to label them for different experiments (e.g., "Island Refuge", "No River Baseline", "High Mutation").

### Requirements

#### 2.1 Add --title flag

```bash
python analyze_evolution.py stats_autosave.npz --title="Island Refuge Experiment"
```

#### 2.2 Implementation

```python
# analyze_evolution.py

import argparse

def main():
    parser = argparse.ArgumentParser(description='Analyze HUNT evolution statistics')
    parser.add_argument('stats_file', type=str, nargs='?', default='stats_autosave.npz',
                        help='Path to stats .npz file')
    parser.add_argument('--title', type=str, default='HUNT Ecosystem - Evolutionary Dynamics',
                        help='Title for the analysis graphs')
    parser.add_argument('--output', type=str, default=None,
                        help='Save figure to file instead of displaying')
    args = parser.parse_args()
    
    # Load and plot...
    
    # Use the title
    fig.suptitle(args.title, fontsize=16, fontweight='bold')
    
    # Optional: save to file
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {args.output}")
    else:
        plt.show()

if __name__ == '__main__':
    main()
```

#### 2.3 Example usage

```bash
# Basic usage
python analyze_evolution.py

# With custom title
python analyze_evolution.py stats_autosave.npz --title="Island Refuge - 545K Steps"

# Save to file
python analyze_evolution.py stats_autosave.npz --title="Island Refuge" --output=island_refuge_analysis.png

# Analyze specific file with title
python analyze_evolution.py experiments/high_mutation_stats.npz --title="High Mutation Rate Experiment"
```

---

## Feature 3: Brain Analysis Tool (Bonus)

Create a simple tool to inspect saved brains:

```python
# analyze_brains.py

import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Analyze saved brain weights')
    parser.add_argument('brain_file', type=str, help='Path to brain .npz file')
    parser.add_argument('--summary', action='store_true', help='Print weight statistics')
    parser.add_argument('--compare', type=str, help='Compare with another brain file')
    args = parser.parse_args()
    
    data = np.load(args.brain_file)
    
    print(f"Brain checkpoint: {args.brain_file}")
    print(f"  Timestep: {data['timestep']}")
    print(f"  Prey alive: {data['prey_alive'].sum()}")
    print(f"  Predators alive: {data['pred_alive'].sum()}")
    
    if args.summary:
        prey_weights = data['prey_weights']
        pred_weights = data['pred_weights']
        
        print(f"\nPrey brain weights:")
        print(f"  Shape: {prey_weights.shape}")
        print(f"  Mean: {prey_weights.mean():.4f}")
        print(f"  Std: {prey_weights.std():.4f}")
        print(f"  Range: [{prey_weights.min():.4f}, {prey_weights.max():.4f}]")
        
        print(f"\nPredator brain weights:")
        print(f"  Shape: {pred_weights.shape}")
        print(f"  Mean: {pred_weights.mean():.4f}")
        print(f"  Std: {pred_weights.std():.4f}")
        print(f"  Range: [{pred_weights.min():.4f}, {pred_weights.max():.4f}]")
    
    if args.compare:
        other = np.load(args.compare)
        print(f"\nComparing with: {args.compare}")
        
        prey_diff = np.abs(data['prey_weights'] - other['prey_weights']).mean()
        pred_diff = np.abs(data['pred_weights'] - other['pred_weights']).mean()
        
        print(f"  Prey weight divergence: {prey_diff:.4f}")
        print(f"  Predator weight divergence: {pred_diff:.4f}")

if __name__ == '__main__':
    main()
```

Usage:
```bash
python analyze_brains.py brains_island_refuge.npz --summary

python analyze_brains.py brains_step_100000.npz --compare brains_step_500000.npz
```

---

## Deliverables

1. **Modified `simulation_gpu.py`**:
   - `save_brain_weights(name=None)` method
   - `load_brain_weights(filename)` method

2. **Modified `main_gpu.py`**:
   - `--load-brains` CLI flag
   - `--brain-checkpoint` CLI flag for naming saves
   - Auto-save brains every 10K steps
   - Save brains on exit
   - 'B' key to manually save brains

3. **Modified `analyze_evolution.py`**:
   - `--title` CLI flag
   - `--output` CLI flag for saving figure
   - Proper argparse setup

4. **New `analyze_brains.py`**:
   - Summary statistics for brain weights
   - Compare two brain checkpoints

---

## Testing

```bash
# Test brain save/load
python main_gpu.py --brain-checkpoint=test_run
# Let run for a bit, press 'B' to save
# Check that brains_test_run.npz exists

# Test loading
python main_gpu.py --load-brains=brains_test_run.npz
# Verify populations and positions restored

# Test analysis title
python analyze_evolution.py stats_autosave.npz --title="Test Title" --output=test.png
# Verify title appears on saved image

# Test brain analysis
python analyze_brains.py brains_test_run.npz --summary
```

---

## Begin

Start with `simulation_gpu.py` - add the save/load methods first, then wire up the CLI and keyboard shortcuts.
