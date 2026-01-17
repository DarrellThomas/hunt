"""Tests for N-species architecture (Phase 2)."""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config_new import SimulationConfig
from species import SpeciesManager, AgentRole
from sensors import SensorSuite, NearestAgentsSensor, HungerSensor
from agent import Prey, Predator


def test_species_manager_initialization():
    """Test SpeciesManager initialization from config."""
    print("Testing SpeciesManager initialization...")

    config = SimulationConfig.default_two_species()
    manager = SpeciesManager(config)

    # Define agent factory
    def create_agent(sp_config):
        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 100)
        if sp_config.has_energy_system:
            return Predator.from_config(x, y, sp_config, 100, 100)
        else:
            return Prey.from_config(x, y, sp_config, 100, 100)

    manager.initialize_populations(create_agent)

    # Check populations
    assert 'prey' in manager.populations
    assert 'predator' in manager.populations
    assert len(manager.get_species('prey')) == 200
    assert len(manager.get_species('predator')) == 40

    print(f"✓ Created {len(manager.get_species('prey'))} prey")
    print(f"✓ Created {len(manager.get_species('predator'))} predators")
    print("✓ test_species_manager_initialization passed\n")


def test_sensor_suite_creation():
    """Test SensorSuite creation from config."""
    print("Testing SensorSuite creation...")

    config = SimulationConfig.default_two_species()
    prey_config = config.get_species('prey')
    predator_config = config.get_species('predator')

    # Create sensor suites
    prey_sensors = SensorSuite.from_config(prey_config.observation)
    pred_sensors = SensorSuite.from_config(predator_config.observation)

    # Check dimensions
    assert prey_sensors.total_dimension == 32  # 5*4 + 3*4 = 32
    assert pred_sensors.total_dimension == 21  # 5*4 + 1 = 21

    print(f"✓ Prey sensor dimension: {prey_sensors.total_dimension}")
    print(f"✓ Predator sensor dimension: {pred_sensors.total_dimension}")
    print("✓ test_sensor_suite_creation passed\n")


def test_agent_from_config():
    """Test creating agents from SpeciesConfig."""
    print("Testing agent creation from config...")

    config = SimulationConfig.default_two_species()
    prey_config = config.get_species('prey')
    predator_config = config.get_species('predator')

    # Create prey
    prey = Prey.from_config(50, 50, prey_config, 100, 100)
    assert prey.sensor_suite is not None
    assert prey.sensor_suite.total_dimension == 32
    assert prey.max_speed == 3.0
    assert prey.max_acceleration == 0.5

    print(f"✓ Prey created with max_speed={prey.max_speed}")
    print(f"✓ Prey sensor dimension={prey.sensor_suite.total_dimension}")

    # Create predator
    predator = Predator.from_config(50, 50, predator_config, 100, 100)
    assert predator.sensor_suite is not None
    assert predator.sensor_suite.total_dimension == 21
    assert predator.max_speed == 2.5
    assert predator.max_acceleration == 0.4
    assert predator.max_energy == 150.0

    print(f"✓ Predator created with max_speed={predator.max_speed}")
    print(f"✓ Predator sensor dimension={predator.sensor_suite.total_dimension}")
    print(f"✓ Predator max_energy={predator.max_energy}")
    print("✓ test_agent_from_config passed\n")


def test_position_and_velocity_extraction():
    """Test getting all positions and velocities."""
    print("Testing position/velocity extraction...")

    config = SimulationConfig.default_two_species()
    manager = SpeciesManager(config)

    def create_agent(sp_config):
        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 100)
        if sp_config.has_energy_system:
            return Predator.from_config(x, y, sp_config, 100, 100)
        else:
            return Prey.from_config(x, y, sp_config, 100, 100)

    manager.initialize_populations(create_agent)

    # Get positions
    positions = manager.get_all_positions()
    velocities = manager.get_all_velocities()

    assert 'prey' in positions
    assert 'predator' in positions
    assert positions['prey'].shape == (200, 2)
    assert positions['predator'].shape == (40, 2)
    assert velocities['prey'].shape == (200, 2)
    assert velocities['predator'].shape == (40, 2)

    print(f"✓ Prey positions shape: {positions['prey'].shape}")
    print(f"✓ Predator positions shape: {positions['predator'].shape}")
    print("✓ test_position_and_velocity_extraction passed\n")


def test_stats_summary():
    """Test stats summary generation."""
    print("Testing stats summary...")

    config = SimulationConfig.default_two_species()
    manager = SpeciesManager(config)

    def create_agent(sp_config):
        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 100)
        if sp_config.has_energy_system:
            return Predator.from_config(x, y, sp_config, 100, 100)
        else:
            return Prey.from_config(x, y, sp_config, 100, 100)

    manager.initialize_populations(create_agent)

    stats = manager.stats_summary()
    assert stats['prey'] == 200
    assert stats['predator'] == 40
    assert manager.total_population() == 240

    print(f"✓ Stats: {stats}")
    print(f"✓ Total population: {manager.total_population()}")
    print("✓ test_stats_summary passed\n")


if __name__ == "__main__":
    print("Running N-species architecture tests (Phase 2)...\n")

    test_species_manager_initialization()
    test_sensor_suite_creation()
    test_agent_from_config()
    test_position_and_velocity_extraction()
    test_stats_summary()

    print("✅ All N-species architecture tests passed!")
    print("✅ Phase 2 core components are working correctly:")
    print("   - SpeciesManager handles multiple species")
    print("   - SensorSuite provides dynamic observations")
    print("   - Agents can be created from SpeciesConfig")
    print("   - Position/velocity extraction works")
