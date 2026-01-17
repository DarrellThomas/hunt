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

# ============================================================================
# RIVER PARAMETERS
# ============================================================================
RIVER_ENABLED = True              # Enable/disable river feature
RIVER_WIDTH = 500                 # Width of the river in pixels
RIVER_FLOW_SPEED = 10.0            # Speed of water current (added to agent velocity)
RIVER_CURVINESS = 0.05             # How curvy the river is (0=straight, 1=very curvy)
RIVER_SPLIT = True                # Whether river splits to create an island
RIVER_SPLIT_START = 0.01           # Where split starts (0-1 along river path)
RIVER_SPLIT_END = 0.99             # Where split ends (0-1 along river path)
RIVER_ISLAND_WIDTH = 200          # Width of island between split channels

# ============================================================================
# ISLAND BEHAVIOR MODIFIERS
# ============================================================================
# These multipliers affect agent behavior while on the island
# Set to 1.0 for no change, >1.0 to increase, <1.0 to decrease

# Prey modifiers
ISLAND_PREY_SPEED_MULTIPLIER = 1.0          # Speed multiplier for prey on island
ISLAND_PREY_REPRODUCTION_MULTIPLIER = 1.0   # Reproduction rate for prey (lower value = faster reproduction)

# Predator modifiers
ISLAND_PRED_SPEED_MULTIPLIER = 1.0          # Speed multiplier for predators on island
ISLAND_PRED_HUNGER_MULTIPLIER = 1.0         # Hunger rate for predators (higher = hungrier faster)
ISLAND_PRED_REPRODUCTION_MULTIPLIER = 1.0   # Reproduction rate for predators (lower value = faster reproduction)
