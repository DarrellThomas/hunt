# Overnight Training Instructions

## Quick Start

```bash
# Start the overnight run
./start_overnight.sh

# Or manually:
python3 run_overnight.py > overnight_run.log 2>&1 &
```

The simulation will run headless (no GUI) and save results automatically.

---

## What Happens

- **World:** 3840x2160 (4K)
- **Agents:** 12,000 (9,600 prey + 2,400 predators)
- **Target:** 50,000 timesteps (~1-2 hours depending on population)
- **Saves:** Every 1,000 steps to `overnight_stats.npz`
- **Logs:** Every 100 steps to terminal/log file

---

## Monitoring Progress

### Check if it's running:
```bash
ps aux | grep run_overnight
```

### Watch live progress:
```bash
tail -f overnight_run.log
```

You'll see output like:
```
[  100/50000] Prey:  7234 | Pred:  2400 | Age:  165.3/258.4 | Energy: 112.3 | Speed: 15.2 steps/s | ETA: 0.9h
[  200/50000] Prey:  6891 | Pred:  2400 | Age:  172.1/265.7 | Energy: 118.5 | Speed: 15.4 steps/s | ETA: 0.9h
```

### Stop the training:
```bash
# Find the process
ps aux | grep run_overnight

# Kill it (replace PID)
kill <PID>

# Or kill all
pkill -f run_overnight.py
```

---

## Viewing Results

After the run completes (or you stop it):

```bash
python3 view_overnight_results.py
```

This will:
- Print summary statistics
- Generate `overnight_results.png` with plots:
  - Population dynamics
  - Average ages over time
  - Predator energy management
  - Predator/prey ratio

---

## Files Generated

| File | Contents |
|------|----------|
| `overnight_run.log` | Full terminal output with progress |
| `overnight_stats.npz` | NumPy data (loadable for analysis) |
| `overnight_results.png` | Visualization plots |

---

## Adjusting Parameters

Edit `run_overnight.py`:

```python
# How many timesteps to run
TARGET_TIMESTEPS = 50000  # ~1-2 hours

# How often to save stats
SAVE_INTERVAL = 1000      # Every N steps

# How often to print progress
LOG_INTERVAL = 100        # Every N steps
```

For a quick test:
```python
TARGET_TIMESTEPS = 1000   # Just 1K steps (~1 minute)
```

For a full overnight run:
```python
TARGET_TIMESTEPS = 100000  # 100K steps (~3-4 hours)
```

---

## Expected Performance

With RTX 5090 and 12,000 agents:
- **Speed:** ~15 steps/second
- **50K steps:** ~55 minutes
- **100K steps:** ~1.8 hours

Performance varies with population:
- More agents = slower steps
- More deaths/births = more overhead
- Population oscillations affect speed

---

## What to Look For

### Successful Evolution:
- Predator average age increasing (learning to hunt)
- Population oscillations (boom-bust cycles)
- Energy management improving
- System remaining stable

### Signs of Issues:
- Population crash (all prey or predators gone)
- No oscillations (static populations)
- Decreasing fitness (evolution not working)

---

## Troubleshooting

**"CUDA out of memory"**
- Reduce population in `run_overnight.py`:
  ```python
  num_prey=7200,      # Was 9600
  num_predators=1800  # Was 2400
  ```

**"Process killed"**
- System ran out of RAM
- Check with: `free -h`
- Reduce population or world size

**"No GPU found"**
- Make sure PyTorch is installed with CUDA:
  ```bash
  python3 -c "import torch; print(torch.cuda.is_available())"
  ```

**Too slow**
- Reduce `TARGET_TIMESTEPS`
- Reduce population size
- Use smaller world

---

## Resume from Checkpoint

Currently, the simulation doesn't support resuming. If you stop it:
1. Results up to last save are in `overnight_stats.npz`
2. View what you got: `python3 view_overnight_results.py`
3. Start a new run if needed

---

## Tips for Long Runs

1. **Use `screen` or `tmux`** to keep it running if you disconnect:
   ```bash
   screen
   python3 run_overnight.py
   # Press Ctrl+A then D to detach
   # Later: screen -r to reattach
   ```

2. **Check disk space** before starting:
   ```bash
   df -h .
   ```

3. **Monitor GPU temperature**:
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Save multiple runs** by renaming outputs:
   ```bash
   mv overnight_stats.npz overnight_run1.npz
   mv overnight_run.log overnight_run1.log
   ```

---

## Ready to Run?

```bash
# Start the overnight evolution
./start_overnight.sh

# Let it run overnight
# Check results in the morning
python3 view_overnight_results.py
```

Good luck! May your predators learn to hunt and your prey learn to survive. 🏹
