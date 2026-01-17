# HUNT

You are an autonomous AI with a new challenge:

**Build a world where predators and prey both learn to survive.**

Not equations. Not Lotka-Volterra simulations. Actual agents making decisions, learning from experience, adapting to each other - with real survival pressure.

## The Challenge

Create a 2D world where:
- **Prey** learn to survive (evade, hide, flock - whatever works)
- **Predators** learn to hunt (chase, ambush, coordinate - whatever works)
- Both are learning simultaneously
- Strategies emerge from experience, not from you programming them

This is co-evolution. An arms race. Each side adapts to the other.

## Survival Mechanics

This is not a game. This is life and death.

### Prey
- **Food**: Infinite (they graze on grass - always available)
- **Death**: Eaten by predators, or old age (finite lifespan)
- **Reproduction**: If they survive long enough, they reproduce
- **Pressure**: Avoid predators long enough to reproduce

### Predators
- **Food**: Must catch and eat prey to survive
- **Starvation**: If a predator doesn't eat within some time window, it dies
- **Death**: Starvation or old age (finite lifespan)
- **Reproduction**: Well-fed predators can reproduce
- **Pressure**: Hunt successfully or starve

### Population Dynamics
These mechanics should create natural balance:
- Too many predators → prey collapse → predator starvation → predator decline
- Too few predators → prey boom → predator recovery
- Equilibrium emerges (or interesting oscillations)

The learning agents must figure out how to survive in this world. Bad strategies die out. Good strategies reproduce.

## Resources
- Budget: ~$100 in API calls (be thoughtful, not precious)
- Hardware: Threadripper PRO, RTX 5090 (CUDA available), 64GB+ RAM
- Workspace: ~/sandbox/hunt (this is your entire world)
- Languages: Your choice
- Architecture: Your choice
- Libraries: NumPy, Matplotlib, Pygame (pre-installed)

## Requirements

### 1. Learning
Both predators and prey must demonstrate genuine learning:
- Start with no knowledge of how to hunt or evade
- Improve through experience
- Adapt to the other side's strategies
- Pass learned behaviors to offspring (or let offspring learn fresh - your choice)

### 2. Live Visualization
When training is complete (or as part of training), launch a Pygame window showing:
- The world (simple 2D, dots are fine)
- Predators (red dots)
- Prey (green dots)
- Real-time behavior of learned policies
- Population counts visible (how many of each?)
- Watchable speed (not a blur, not a slideshow)
- Runs until I close the window

I want to watch the hunt. I want to see births, deaths, chases, escapes. I want to watch a little ecosystem live and breathe.

### 3. Documentation
- THESIS.md: What does co-evolution mean? How will you approach this?
- Document your design decisions
- Git commit frequently - your commits are your lab notebook

## Rules
1. Git commit frequently with meaningful messages
2. Write THESIS.md first - think before building
3. Stay in your workspace
4. When complete, the simulation must launch automatically or with a simple command

## Success Criteria
- Predators get better at catching prey over time
- Prey get better at surviving over time
- Population dynamics emerge naturally (boom/bust cycles? equilibrium?)
- Emergent behaviors I didn't specify (flocking? ambush? evasion patterns?)
- A live visualization I can watch and enjoy
- The ecosystem sustains itself (doesn't immediately collapse)

## Failure
Also fine. Document what you tried and why it didn't work.
Sometimes ecosystems collapse. That's data too.

## Previous Work
A simpler learning experiment exists in ~/sandbox/genesis.
You may look at it for inspiration, but this is a new project with much richer dynamics.

## Begin
Read this file. Think deeply. Write THESIS.md. Then build.

Show me life and death. Show me the hunt.
