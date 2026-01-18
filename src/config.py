"""
Configuration file for HUNT simulation.
Modify these parameters between runs to experiment with different initial conditions.

These are INITIAL constants. As agents evolve, they live longer,
reproduce more, and can live on less food.
"""

# ============================================================================
# PREY PARAMETERS
# ============================================================================
PREY_MAX_SPEED = 3.0              # Initial maximum speed
PREY_MAX_ACCELERATION = 0.5       # Initial acceleration capability
PREY_SWIM_SPEED = 2.0             # Initial swimming ability (resistance to current, evolvable)
PREY_MAX_LIFESPAN = 500           # Initial maximum age (mean)
PREY_LIFESPAN_VARIANCE = 50       # Variance in lifespan
PREY_REPRODUCTION_AGE = 200       # Initial age when reproduction possible (mean)
PREY_REPRODUCTION_VARIANCE = 20   # Variance in reproduction age

# ============================================================================
# PREDATOR PARAMETERS
# ============================================================================
PRED_MAX_SPEED = 2.5              # Initial maximum speed (slightly slower than prey)
PRED_MAX_ACCELERATION = 0.4       # Initial acceleration capability
PRED_SWIM_SPEED = 1.5             # Initial swimming ability (resistance to current, evolvable)
PRED_MAX_LIFESPAN = 800           # Initial maximum age (mean)
PRED_LIFESPAN_VARIANCE = 80       # Variance in lifespan
PRED_MAX_ENERGY = 150             # Initial maximum energy/health
PRED_ENERGY_COST = 0.3            # Initial energy cost per step (hunger rate)
PRED_ENERGY_GAIN = 60             # Initial energy gained per kill
PRED_REPRODUCTION_THRESHOLD = 120 # Initial energy needed to reproduce
PRED_REPRODUCTION_COST = 40       # Initial energy cost of reproduction
PRED_REPRODUCTION_COOLDOWN = 150  # Initial time between reproductions (mean)
PRED_REPRODUCTION_VARIANCE = 15   # Variance in reproduction cooldown

# ============================================================================
# WORLD PARAMETERS
# ============================================================================
CATCH_RADIUS = 8.0                # Distance at which predator catches prey
FRICTION = 0.1                    # Velocity damping per timestep (0.0-1.0, higher = more friction)

# ============================================================================
# RIVER PARAMETERS
# ============================================================================
RIVER_ENABLED = True              # Enable/disable river feature
RIVER_WIDTH = 500                 # Width of the river in pixels
RIVER_FLOW_SPEED = 1.25            # Speed of water current (added to agent velocity)
RIVER_CURVINESS = 0.0            # How curvy the river is (0=straight, 1=very curvy)
RIVER_SPLIT = True                # Whether river splits to create an island
RIVER_SPLIT_START = 0.3           # Where split starts (0-1 along river path)
RIVER_SPLIT_END = 0.7             # Where split ends (0-1 along river path)
RIVER_ISLAND_WIDTH = 200          # Width of island between split channels

# ============================================================================
# ISLAND BEHAVIOR MODIFIERS
# ============================================================================
# These multipliers affect agent behavior while on the island

# Prey modifiers
ISLAND_PREY_SPEED_MULTIPLIER = 2.0          # Speed modifier (1.0 = normal, 1.5 = 50% faster, 0.5 = 50% slower)
ISLAND_PREY_REPRODUCTION_MULTIPLIER = .8   # Reproduction wait time modifier (1.0 = normal, 0.5 = reproduce 2x faster, 2.0 = reproduce 2x slower)

# Predator modifiers
ISLAND_PRED_SPEED_MULTIPLIER = 0.2          # Speed modifier (1.0 = normal, 1.5 = 50% faster, 0.5 = 50% slower)
ISLAND_PRED_HUNGER_MULTIPLIER = 3.0         # Hunger rate modifier (1.0 = normal, 2.0 = get hungry 2x faster, 0.5 = get hungry 2x slower)
ISLAND_PRED_REPRODUCTION_MULTIPLIER = 3.0   # Reproduction wait time modifier (1.0 = normal, 0.5 = reproduce 2x faster, 2.0 = reproduce 2x slower)
