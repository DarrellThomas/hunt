# THESIS: Co-Evolution Through Survival

## The Core Idea

Create a predator-prey ecosystem where intelligence emerges not from careful reward engineering, but from the fundamental pressure of survival itself. Predators that fail to hunt starve. Prey that fail to evade die. The rest is emergent.

## Why This Matters

Most RL treats learning as optimization against a static reward function. But nature doesn't work that way. In real ecosystems, there's no fixed objective - your "reward function" is whatever helps you survive against opponents who are also adapting. This is co-evolution: an endless arms race where each side's improvement forces the other to improve.

## The Approach: Neuroevolution

Instead of traditional RL (PPO, DQN), I'll use **neuroevolution** - a more biologically plausible approach that fits the theme perfectly:

### How It Works
1. Each agent (predator and prey) has a neural network brain
2. Successful agents reproduce, passing their neural weights to offspring with mutations
3. Failed agents die and their genes are removed from the pool
4. No backpropagation, no gradient descent - just survival of the fittest

### Why Neuroevolution?
- **Biologically authentic**: This is literally how evolution works
- **Naturally handles co-evolution**: Both populations evolve simultaneously through selection pressure
- **No reward engineering**: Fitness = survival time + reproduction success
- **Emergent complexity**: Complex behaviors arise from simple selection pressure
- **Parallelizable**: Can simulate many agents at once on GPU

### Alternative: PPO with Shared Policies
Could use PPO where all prey share one policy network and all predators share another. This would be faster to train but less authentic - it's group learning rather than individual evolution. I'll start with neuroevolution for authenticity, but keep PPO as a backup if training is too slow.

## The World

### Physics
- 2D continuous space (e.g., 800x600 units)
- Toroidal topology (wrap around edges) to avoid corners
- Simple physics: position, velocity, acceleration
- Maximum speed limits (prey faster than predators? or equal?)

### Agents

**Prey (Green)**
- Sensory input: Positions and velocities of N nearest predators, N nearest prey (for flocking)
- Actions: Acceleration vector (continuous 2D)
- Food: Always available (grass grazing), no hunger
- Death conditions: Caught by predator (within catch radius), or old age (max lifespan ~500 timesteps)
- Reproduction: Every 200 timesteps survived, spawn offspring
- Starting population: 50

**Predators (Red)**
- Sensory input: Positions and velocities of N nearest prey, own hunger level
- Actions: Acceleration vector (continuous 2D)
- Food: Must catch prey to restore energy
- Energy: Starts at 100, depletes by 1 per timestep
- Eating: Catching prey restores 50 energy
- Death conditions: Energy reaches 0 (starvation), or old age (max lifespan ~800 timesteps)
- Reproduction: If energy > 80, spawn offspring (costs 30 energy)
- Starting population: 10

### Neural Network Architecture
Simple feedforward network:
- Input layer: Sensory observations (positions, velocities, hunger)
- Hidden layers: 2 layers of 32 neurons each with tanh activation
- Output layer: 2 neurons (ax, ay) with tanh activation

Small networks evolve faster and are less prone to overfitting.

## Population Dynamics

### Birth
- Offspring spawn near parent with slightly randomized position
- Neural weights copied from parent with Gaussian noise (mutation rate ~0.1)
- Offspring inherit parent's energy budget (for predators)

### Death
- Dead agents immediately removed from simulation
- When population drops too low (prey < 10 or predators < 3), spawn new random agents to prevent extinction

### Expected Dynamics
- Early phase: Random movement, predators struggle to catch prey
- Middle phase: Predators learn to chase, prey learn to evade
- Late phase: Arms race - sophisticated hunting vs sophisticated evasion
- Possible outcomes:
  - Stable equilibrium (population oscillations within bounds)
  - Boom-bust cycles (predator-prey oscillations)
  - One side dominates (collapse scenario - still interesting!)

## Training Strategy

### Simulation Loop
1. All agents observe environment and compute actions via neural networks
2. Physics update (velocities, positions)
3. Check collisions (predator catches prey if distance < catch_radius)
4. Update energy levels
5. Handle deaths
6. Handle reproduction
7. Replace dead agents with offspring of survivors (weighted by fitness)

### Fitness Metrics
- **Prey fitness**: Survival time + number of offspring produced
- **Predator fitness**: Number of prey caught + survival time + number of offspring

### Training Duration
- Run for multiple generations (10,000+ timesteps)
- Monitor population sizes and behavior complexity
- Stop when interesting emergent behaviors stabilize or after reasonable compute time

### Computational Efficiency
- Use NumPy for vectorized operations
- Parallelize neural network forward passes
- Could use PyTorch for GPU acceleration if needed (RTX 5090 available)
- Target: ~1000 agents simulated at 30-60 FPS

## Visualization

### Pygame Display
- 800x600 window
- Green dots: Prey
- Red dots: Predators
- Dot size indicates energy/age
- Simple trails showing recent movement
- UI overlay: Population counts, generation number, average fitness

### Watchability
- Slow down simulation to ~30 FPS for human viewing
- Zoom/pan controls optional
- Pause/resume functionality
- Save/load best generation

## Expected Emergent Behaviors

### Prey
- **Flocking**: Safety in numbers
- **Edge avoidance**: Stay away from predators
- **Zigzagging**: Unpredictable evasion
- **Confusion effect**: Sudden direction changes when predator nearby

### Predators
- **Pursuit**: Direct chasing of nearest prey
- **Ambush**: Position ahead of prey trajectory
- **Pack hunting**: Multiple predators coordinate (if they learn this!)
- **Energy management**: Balance hunting effort with starvation risk

### Population Dynamics
- Oscillations: Predator boom → prey decline → predator bust → prey recovery
- Spatial patterns: Clusters of prey with predators on periphery
- Evolutionary innovations: Sudden strategy shifts when mutation succeeds

## Success Metrics

1. **Learning occurs**: Both species improve over time (measurable via fitness curves)
2. **Co-evolution**: Predator improvements force prey improvements and vice versa
3. **Sustainability**: System doesn't immediately collapse
4. **Emergent complexity**: Behaviors more sophisticated than initial random movement
5. **Watchable**: Visualization is interesting to observe for minutes

## Risks and Mitigations

### Risk: Populations collapse immediately
- **Mitigation**: Tune death rates, reproduction rates, starting populations
- **Mitigation**: Add minimum population thresholds with random spawns

### Risk: Learning is too slow
- **Mitigation**: Increase mutation rate
- **Mitigation**: Switch to PPO if neuroevolution proves too slow
- **Mitigation**: Reduce world size or agent counts

### Risk: One side dominates permanently
- **Mitigation**: Balance predator/prey speeds and sensing ranges
- **Mitigation**: Adjust energy costs and catch radius
- **This might actually be interesting data!**

### Risk: No emergent complexity
- **Mitigation**: Ensure agents have sufficient sensory input to enable complex strategies
- **Mitigation**: Run longer - evolution is slow
- **Mitigation**: Increase network capacity if needed

## Implementation Plan

1. **Core simulation**: World physics, agent movement, collision detection
2. **Neural networks**: Feedforward nets with NumPy or PyTorch
3. **Evolution loop**: Reproduction, mutation, selection
4. **Population dynamics**: Birth, death, aging, energy management
5. **Visualization**: Pygame rendering with real-time statistics
6. **Training**: Run until interesting behaviors emerge
7. **Analysis**: Plot fitness over time, population dynamics, behavior examples

## Timeline Estimate

Given the hardware (Threadripper, RTX 5090) and budget (~$100 API calls, though I won't need external APIs for this):
- Implementation: A few hours
- Initial training experiments: Test different parameters
- Full training run: Could be minutes to hours depending on emergence speed
- Iteration: Tune parameters based on what emerges

## Conclusion

This project explores co-evolution through the most fundamental mechanism: survival. No carefully crafted reward functions, no curriculum learning, no human guidance. Just predators that must hunt and prey that must evade, both adapting to each other through the simplest rule: survive or die.

If it works, we'll see intelligence emerge from necessity. If it fails, we'll learn why survival pressure alone isn't sufficient. Either way, we'll watch something alive.

Let the hunt begin.
