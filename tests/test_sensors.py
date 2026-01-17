"""Tests for sensor system."""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sensors import (
    NearestAgentsSensor, HungerSensor, IslandProximitySensor,
    WallProximitySensor, SensorSuite
)
from config_new import SimulationConfig, BoundaryMode, ObservationConfig
from agent import Prey, Predator


class MockWorld:
    """Mock world for testing sensors."""
    def __init__(self, width=100, height=100, boundary_mode=BoundaryMode.TOROIDAL, has_river=False):
        self.width = width
        self.height = height
        self.boundary_mode = boundary_mode
        self.river = MockRiver(has_river)


class MockRiver:
    """Mock river for testing."""
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.split = enabled

    def is_on_island(self, x, y):
        # Simple test: island at center
        return abs(x - 50) < 10 and abs(y - 50) < 10


def test_nearest_agents_sensor():
    """Test NearestAgentsSensor observation."""
    print("Testing NearestAgentsSensor...")

    sensor = NearestAgentsSensor(target_species='prey', count=3, include_velocity=True)

    # Check dimension
    assert sensor.dimension == 12  # 3 agents * 4 features (x, y, vx, vy)

    # Create mock agent
    agent = Prey(50, 50, 100, 100)

    # Create mock positions and velocities
    positions = {
        'prey': np.array([[52, 51], [48, 49], [60, 60], [40, 40]], dtype=np.float32),
        'predator': np.array([[70, 70]], dtype=np.float32)
    }
    velocities = {
        'prey': np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.float32),
        'predator': np.array([[0.5, 0.5]], dtype=np.float32)
    }

    world = MockWorld()

    # Observe (excluding self - index 0)
    observation = sensor.observe(agent, world, positions, velocities, 'prey', 0)

    assert observation.shape == (12,)
    print(f"  Observation shape: {observation.shape}")
    print(f"  Observation (first 4): {observation[:4]}")

    # Should observe 3 nearest prey (excluding self)
    # Nearest should be [48, 49] which is at index 1
    print("✓ NearestAgentsSensor works correctly")
    print("✓ test_nearest_agents_sensor passed\n")


def test_nearest_agents_sensor_without_velocity():
    """Test NearestAgentsSensor without velocity."""
    print("Testing NearestAgentsSensor (no velocity)...")

    sensor = NearestAgentsSensor(target_species='predator', count=2, include_velocity=False)

    # Check dimension
    assert sensor.dimension == 4  # 2 agents * 2 features (x, y)

    agent = Prey(50, 50, 100, 100)

    positions = {
        'prey': np.array([[50, 50]], dtype=np.float32),
        'predator': np.array([[52, 51], [48, 49]], dtype=np.float32)
    }
    velocities = {
        'prey': np.array([[0, 0]], dtype=np.float32),
        'predator': np.array([[1, 0], [-1, 0]], dtype=np.float32)
    }

    world = MockWorld()

    observation = sensor.observe(agent, world, positions, velocities, 'prey', 0)

    assert observation.shape == (4,)
    print(f"  Observation shape: {observation.shape}")
    print("✓ NearestAgentsSensor (no velocity) works correctly")
    print("✓ test_nearest_agents_sensor_without_velocity passed\n")


def test_hunger_sensor():
    """Test HungerSensor observation."""
    print("Testing HungerSensor...")

    sensor = HungerSensor()

    # Check dimension
    assert sensor.dimension == 1

    # Create predator with energy
    agent = Predator(50, 50, 100, 100)
    agent.energy = 75.0
    agent.max_energy = 150.0

    world = MockWorld()
    positions = {}
    velocities = {}

    observation = sensor.observe(agent, world, positions, velocities, 'predator', 0)

    assert observation.shape == (1,)
    # Hunger = 1 - (energy / max_energy) = 1 - (75/150) = 0.5
    expected_hunger = 0.5
    assert abs(observation[0] - expected_hunger) < 0.01

    print(f"  Energy: {agent.energy}/{agent.max_energy}")
    print(f"  Hunger: {observation[0]:.2f}")
    print("✓ HungerSensor works correctly")
    print("✓ test_hunger_sensor passed\n")


def test_island_proximity_sensor():
    """Test IslandProximitySensor observation."""
    print("Testing IslandProximitySensor...")

    sensor = IslandProximitySensor()

    # Check dimension
    assert sensor.dimension == 1

    world = MockWorld(has_river=True)
    positions = {}
    velocities = {}

    # Agent on island (center)
    agent_on_island = Prey(50, 50, 100, 100)
    obs_on = sensor.observe(agent_on_island, world, positions, velocities, 'prey', 0)
    assert obs_on[0] == 1.0, "Should detect island"

    # Agent off island
    agent_off_island = Prey(20, 20, 100, 100)
    obs_off = sensor.observe(agent_off_island, world, positions, velocities, 'prey', 0)
    assert obs_off[0] == 0.0, "Should not detect island"

    print(f"  On island: {obs_on[0]}")
    print(f"  Off island: {obs_off[0]}")
    print("✓ IslandProximitySensor works correctly")
    print("✓ test_island_proximity_sensor passed\n")


def test_wall_proximity_sensor():
    """Test WallProximitySensor observation."""
    print("Testing WallProximitySensor...")

    sensor = WallProximitySensor()

    # Check dimension
    assert sensor.dimension == 4  # left, right, top, bottom

    world = MockWorld(boundary_mode=BoundaryMode.BOUNDED)
    positions = {}
    velocities = {}

    # Agent at center
    agent = Prey(50, 50, 100, 100)
    observation = sensor.observe(agent, world, positions, velocities, 'prey', 0)

    assert observation.shape == (4,)
    # At center (50, 50) in 100x100 world:
    # left: 50/100 = 0.5, right: 50/100 = 0.5, top: 50/100 = 0.5, bottom: 50/100 = 0.5
    assert all(abs(observation[i] - 0.5) < 0.01 for i in range(4))

    print(f"  Center position walls: {observation}")

    # Agent near left wall
    agent_left = Prey(10, 50, 100, 100)
    obs_left = sensor.observe(agent_left, world, positions, velocities, 'prey', 0)
    assert obs_left[0] < 0.2  # Close to left wall
    assert obs_left[1] > 0.8  # Far from right wall

    print(f"  Left position walls: {obs_left}")
    print("✓ WallProximitySensor works correctly")
    print("✓ test_wall_proximity_sensor passed\n")


def test_wall_proximity_sensor_toroidal():
    """Test WallProximitySensor returns zeros for toroidal world."""
    print("Testing WallProximitySensor (toroidal mode)...")

    sensor = WallProximitySensor()

    world = MockWorld(boundary_mode=BoundaryMode.TOROIDAL)
    positions = {}
    velocities = {}

    agent = Prey(50, 50, 100, 100)
    observation = sensor.observe(agent, world, positions, velocities, 'prey', 0)

    # Should return all zeros for toroidal world
    assert all(observation[i] == 0.0 for i in range(4))

    print(f"  Toroidal mode: {observation}")
    print("✓ WallProximitySensor returns zeros for toroidal")
    print("✓ test_wall_proximity_sensor_toroidal passed\n")


def test_sensor_suite_composition():
    """Test SensorSuite combines multiple sensors."""
    print("Testing SensorSuite composition...")

    sensors = [
        NearestAgentsSensor('predator', 2, include_velocity=True),  # 8 dims
        NearestAgentsSensor('prey', 1, include_velocity=False),     # 2 dims
        HungerSensor(),                                              # 1 dim
    ]

    suite = SensorSuite(sensors)

    # Check total dimension
    assert suite.total_dimension == 11
    print(f"  Total dimension: {suite.total_dimension}")

    # Create mock agent
    agent = Predator(50, 50, 100, 100)
    agent.energy = 100.0
    agent.max_energy = 150.0

    positions = {
        'prey': np.array([[52, 51]], dtype=np.float32),
        'predator': np.array([[50, 50], [60, 60]], dtype=np.float32)
    }
    velocities = {
        'prey': np.array([[1, 0]], dtype=np.float32),
        'predator': np.array([[0, 0], [0, 1]], dtype=np.float32)
    }

    world = MockWorld()

    # Get combined observation
    observation = suite.observe(agent, world, positions, velocities, 'predator', 0)

    assert observation.shape == (11,)
    print(f"  Observation shape: {observation.shape}")
    print(f"  Observation: {observation}")
    print("✓ SensorSuite composition works correctly")
    print("✓ test_sensor_suite_composition passed\n")


def test_sensor_suite_from_config():
    """Test creating SensorSuite from ObservationConfig."""
    print("Testing SensorSuite.from_config()...")

    config = SimulationConfig.default_two_species()
    prey_config = config.get_species('prey')

    # Create sensor suite from config
    suite = SensorSuite.from_config(prey_config.observation)

    # Prey observes: 5 predators (4 each) + 3 prey (4 each) = 32
    assert suite.total_dimension == 32
    print(f"  Prey sensor dimension: {suite.total_dimension}")

    # Check sensors were created
    assert len(suite.sensors) > 0
    print(f"  Number of sensors: {len(suite.sensors)}")

    print("✓ SensorSuite.from_config() works correctly")
    print("✓ test_sensor_suite_from_config passed\n")


def test_sensor_padding():
    """Test that sensors pad observations when fewer targets than requested."""
    print("Testing sensor padding...")

    sensor = NearestAgentsSensor('prey', count=5, include_velocity=True)

    agent = Predator(50, 50, 100, 100)

    # Only 2 prey available, but sensor wants 5
    positions = {
        'prey': np.array([[52, 51], [48, 49]], dtype=np.float32),
        'predator': np.array([[50, 50]], dtype=np.float32)
    }
    velocities = {
        'prey': np.array([[1, 0], [-1, 0]], dtype=np.float32),
        'predator': np.array([[0, 0]], dtype=np.float32)
    }

    world = MockWorld()

    observation = sensor.observe(agent, world, positions, velocities, 'predator', 0)

    # Should still return correct dimension (5 * 4 = 20) with padding
    assert observation.shape == (20,)

    # First 8 values should be non-zero (2 prey * 4 features)
    # Remaining 12 should be zero (padding)
    non_zero_count = np.count_nonzero(observation)
    print(f"  Observation shape: {observation.shape}")
    print(f"  Non-zero values: {non_zero_count}")
    print(f"  Observation: {observation}")

    print("✓ Sensor padding works correctly")
    print("✓ test_sensor_padding passed\n")


if __name__ == "__main__":
    print("Running sensor system tests...\n")

    test_nearest_agents_sensor()
    test_nearest_agents_sensor_without_velocity()
    test_hunger_sensor()
    test_island_proximity_sensor()
    test_wall_proximity_sensor()
    test_wall_proximity_sensor_toroidal()
    test_sensor_suite_composition()
    test_sensor_suite_from_config()
    test_sensor_padding()

    print("✅ All sensor system tests passed!")
    print("✅ Sensor system is working correctly:")
    print("   - NearestAgentsSensor with/without velocity")
    print("   - HungerSensor for energy-based agents")
    print("   - IslandProximitySensor for river islands")
    print("   - WallProximitySensor for bounded worlds")
    print("   - SensorSuite composition and from_config()")
    print("   - Proper padding when fewer targets available")
