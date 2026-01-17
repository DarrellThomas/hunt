"""Unit tests for config_new.py configuration system."""

import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config_new import (
    BoundaryMode,
    PhysicsConfig,
    LifecycleConfig,
    EnergyConfig,
    ObservationConfig,
    SpeciesConfig,
    RiverConfig,
    ExtinctionPreventionConfig,
    WorldConfig,
    InteractionConfig,
    SimulationConfig
)


def test_boundary_mode():
    """Test BoundaryMode enum."""
    assert BoundaryMode.TOROIDAL.value == "toroidal"
    assert BoundaryMode.BOUNDED.value == "bounded"
    print("✓ test_boundary_mode passed")


def test_physics_config_validation():
    """Test PhysicsConfig validation."""
    # Valid config
    config = PhysicsConfig(max_speed=3.0, max_acceleration=0.5, swim_speed=2.0)
    config.validate()  # Should not raise

    # Invalid: negative speed
    try:
        config = PhysicsConfig(max_speed=-1.0, max_acceleration=0.5)
        config.validate()
        assert False, "Should have raised assertion error"
    except AssertionError:
        pass  # Expected

    print("✓ test_physics_config_validation passed")


def test_observation_config_dimension():
    """Test ObservationConfig dimension calculation."""
    # Observe 5 predators + 3 prey = 5*4 + 3*4 = 32
    obs = ObservationConfig(observe_species={'predator': 5, 'prey': 3})
    assert obs.base_dimension == 32

    # Add hunger sensor: 32 + 1 = 33
    obs = ObservationConfig(
        observe_species={'predator': 5, 'prey': 3},
        sense_hunger=True
    )
    assert obs.base_dimension == 33

    # Add island sensor: 33 + 1 = 34
    obs = ObservationConfig(
        observe_species={'predator': 5, 'prey': 3},
        sense_hunger=True,
        sense_island=True
    )
    assert obs.base_dimension == 34

    print("✓ test_observation_config_dimension passed")


def test_species_config():
    """Test SpeciesConfig."""
    prey = SpeciesConfig(
        name='prey',
        physics=PhysicsConfig(max_speed=3.0, max_acceleration=0.5),
        lifecycle=LifecycleConfig(
            max_lifespan=500,
            lifespan_variance=100,
            reproduction_age=100,
            reproduction_variance=20
        ),
        observation=ObservationConfig(observe_species={'predator': 5}),
        initial_count=200,
        color=(0, 255, 0),
        energy=None
    )

    assert prey.has_energy_system == False
    assert prey.input_size == 20  # 5 * 4 = 20

    prey.validate()  # Should not raise

    print("✓ test_species_config passed")


def test_default_two_species():
    """Test default_two_species factory."""
    config = SimulationConfig.default_two_species()

    # Check species
    assert len(config.species) == 2
    assert config.species[0].name == 'prey'
    assert config.species[1].name == 'predator'

    # Check prey config
    prey = config.species[0]
    assert prey.physics.max_speed == 3.0
    assert prey.has_energy_system == False
    assert prey.input_size == 32  # 5*4 + 3*4

    # Check predator config
    pred = config.species[1]
    assert pred.physics.max_speed == 2.5
    assert pred.has_energy_system == True
    assert pred.input_size == 21  # 5*4 + 1 hunger

    # Check interactions
    assert len(config.interactions) == 1
    assert config.interactions[0].predator_species == 'predator'
    assert config.interactions[0].prey_species == 'prey'

    # Check world
    assert config.world.boundary_mode == BoundaryMode.TOROIDAL
    assert config.world.width == 1600
    assert config.world.height == 1200

    # Validate entire config
    config.validate()

    print("✓ test_default_two_species passed")


def test_default_bounded():
    """Test default_bounded factory."""
    config = SimulationConfig.default_bounded()
    assert config.world.boundary_mode == BoundaryMode.BOUNDED
    config.validate()
    print("✓ test_default_bounded passed")


def test_json_serialization():
    """Test JSON serialization and deserialization."""
    # Create config
    original = SimulationConfig.default_two_species()

    # Serialize to JSON
    json_str = original.to_json()
    assert isinstance(json_str, str)
    assert len(json_str) > 0

    # Parse JSON (ensure it's valid)
    json_data = json.loads(json_str)
    assert 'species' in json_data
    assert 'world' in json_data

    # Deserialize back
    restored = SimulationConfig.from_json(json_str)

    # Check that restored config matches original
    assert len(restored.species) == len(original.species)
    assert restored.species[0].name == original.species[0].name
    assert restored.species[1].name == original.species[1].name
    assert restored.world.width == original.world.width
    assert restored.world.boundary_mode == original.world.boundary_mode
    assert restored.mutation_rate == original.mutation_rate

    # Validate restored config
    restored.validate()

    print("✓ test_json_serialization passed")


def test_get_species():
    """Test get_species method."""
    config = SimulationConfig.default_two_species()

    prey = config.get_species('prey')
    assert prey is not None
    assert prey.name == 'prey'

    pred = config.get_species('predator')
    assert pred is not None
    assert pred.name == 'predator'

    none_species = config.get_species('nonexistent')
    assert none_species is None

    print("✓ test_get_species passed")


def test_validation_catches_invalid_interactions():
    """Test that validation catches invalid species references."""
    config = SimulationConfig.default_two_species()

    # Add invalid interaction
    config.interactions.append(
        InteractionConfig(
            predator_species='tyrannosaurus',  # Doesn't exist
            prey_species='prey',
            catch_radius=8.0,
            energy_gain=50.0
        )
    )

    try:
        config.validate()
        assert False, "Should have raised assertion error for invalid species"
    except AssertionError as e:
        assert "Unknown predator species" in str(e)

    print("✓ test_validation_catches_invalid_interactions passed")


def test_to_dict():
    """Test to_dict conversion."""
    config = SimulationConfig.default_two_species()
    data = config.to_dict()

    assert isinstance(data, dict)
    assert 'world' in data
    assert 'species' in data
    assert 'interactions' in data

    # Check that enum is converted to string
    assert data['world']['boundary_mode'] == 'toroidal'

    # Check that nested objects are dicts
    assert isinstance(data['species'][0], dict)
    assert isinstance(data['species'][0]['physics'], dict)

    print("✓ test_to_dict passed")


if __name__ == "__main__":
    print("Running config_new.py unit tests...\n")

    test_boundary_mode()
    test_physics_config_validation()
    test_observation_config_dimension()
    test_species_config()
    test_default_two_species()
    test_default_bounded()
    test_json_serialization()
    test_get_species()
    test_validation_catches_invalid_interactions()
    test_to_dict()

    print("\n✅ All config_new.py tests passed!")
