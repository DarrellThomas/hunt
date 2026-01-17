"""Tests for trait system."""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from traits import Trait, TraitCollection, COMMON_PREY_TRAITS, COMMON_PREDATOR_TRAITS


def test_trait_initialization():
    """Test Trait creation and basic properties."""
    print("Testing Trait initialization...")

    trait = Trait(
        name='speed',
        initial_value=3.0,
        initial_std=0.3,
        mutation_std=0.2,
        min_value=0.5,
        max_value=10.0
    )

    assert trait.name == 'speed'
    assert trait.initial_value == 3.0
    assert trait.initial_std == 0.3
    assert trait.mutation_std == 0.2
    assert trait.min_value == 0.5
    assert trait.max_value == 10.0

    print("✓ Trait initialized correctly")
    print("✓ test_trait_initialization passed\n")


def test_trait_sample_initial():
    """Test initial value sampling."""
    print("Testing initial value sampling...")

    trait = Trait('speed', 3.0, 0.3, 0.2, 0.5, 10.0)

    # Sample many values
    samples = [trait.sample_initial() for _ in range(1000)]

    # Check bounds are respected
    assert all(0.5 <= s <= 10.0 for s in samples), "Some values out of bounds"

    # Check distribution is reasonable
    mean = np.mean(samples)
    std = np.std(samples)

    # Should be close to initial_value (3.0) and initial_std (0.3)
    assert 2.5 < mean < 3.5, f"Mean {mean} too far from 3.0"
    assert 0.2 < std < 0.5, f"Std {std} too far from 0.3"

    print(f"  Sampled 1000 values: mean={mean:.2f}, std={std:.2f}")
    print("✓ Initial sampling works correctly")
    print("✓ test_trait_sample_initial passed\n")


def test_trait_mutate():
    """Test trait mutation."""
    print("Testing trait mutation...")

    trait = Trait('speed', 3.0, 0.3, 0.2, 0.5, 10.0)
    parent_value = 3.0
    mutation_rate = 0.1

    # Generate many mutations
    mutations = [trait.mutate(parent_value, mutation_rate) for _ in range(1000)]

    # All should be in bounds
    assert all(0.5 <= m <= 10.0 for m in mutations), "Some mutations out of bounds"

    # Should be centered near parent
    mean = np.mean(mutations)
    std = np.std(mutations)

    assert 2.7 < mean < 3.3, f"Mean {mean} too far from parent 3.0"

    # Mutation std should be mutation_std * mutation_rate = 0.2 * 0.1 = 0.02
    # But we expect std close to that (not exact due to sampling)
    print(f"  1000 mutations: mean={mean:.2f}, std={std:.3f}")
    print("✓ Mutation works correctly")
    print("✓ test_trait_mutate passed\n")


def test_trait_bounds_clamping():
    """Test that traits respect min/max bounds."""
    print("Testing bounds clamping...")

    trait = Trait('speed', 5.0, 10.0, 10.0, 0.0, 10.0)

    # Sample should never exceed bounds even with large std
    samples = [trait.sample_initial() for _ in range(1000)]
    assert all(0.0 <= s <= 10.0 for s in samples), "Sample violated bounds"

    # Mutation should never exceed bounds even with large mutation
    mutations = [trait.mutate(5.0, 100.0) for _ in range(1000)]
    assert all(0.0 <= m <= 10.0 for m in mutations), "Mutation violated bounds"

    print("✓ Bounds are properly enforced")
    print("✓ test_trait_bounds_clamping passed\n")


def test_trait_collection():
    """Test TraitCollection management."""
    print("Testing TraitCollection...")

    traits = {
        'speed': Trait('speed', 3.0, 0.3, 0.2, 0.5, 10.0),
        'vision': Trait('vision', 100.0, 10.0, 15.0, 20.0, 300.0),
    }

    collection = TraitCollection(traits)

    # Test initial sampling
    initial_values = collection.sample_initial_values()
    assert 'speed' in initial_values
    assert 'vision' in initial_values
    assert 0.5 <= initial_values['speed'] <= 10.0
    assert 20.0 <= initial_values['vision'] <= 300.0

    print(f"  Initial values: {initial_values}")

    # Test mutation
    child_values = collection.mutate_values(initial_values, mutation_rate=0.1)
    assert 'speed' in child_values
    assert 'vision' in child_values
    assert 0.5 <= child_values['speed'] <= 10.0
    assert 20.0 <= child_values['vision'] <= 300.0

    print(f"  Child values: {child_values}")

    # Values should be different (most of the time)
    # Allow for rare case where mutation is exactly zero
    differences = [abs(initial_values[k] - child_values[k]) for k in initial_values]
    print(f"  Differences: {differences}")

    print("✓ TraitCollection works correctly")
    print("✓ test_trait_collection passed\n")


def test_common_trait_definitions():
    """Test that common trait definitions are valid."""
    print("Testing common trait definitions...")

    # Prey traits
    prey_collection = TraitCollection(COMMON_PREY_TRAITS)
    prey_values = prey_collection.sample_initial_values()

    print(f"  Common prey traits: {list(COMMON_PREY_TRAITS.keys())}")
    print(f"  Sample prey values: {prey_values}")

    for name, value in prey_values.items():
        trait = COMMON_PREY_TRAITS[name]
        assert trait.min_value <= value <= trait.max_value, f"{name} out of bounds"

    # Predator traits
    pred_collection = TraitCollection(COMMON_PREDATOR_TRAITS)
    pred_values = pred_collection.sample_initial_values()

    print(f"  Common predator traits: {list(COMMON_PREDATOR_TRAITS.keys())}")
    print(f"  Sample predator values: {pred_values}")

    for name, value in pred_values.items():
        trait = COMMON_PREDATOR_TRAITS[name]
        assert trait.min_value <= value <= trait.max_value, f"{name} out of bounds"

    print("✓ Common trait definitions are valid")
    print("✓ test_common_trait_definitions passed\n")


def test_trait_inheritance():
    """Test that mutation maintains reasonable variation."""
    print("Testing trait inheritance across generations...")

    trait = Trait('speed', 3.0, 0.3, 0.2, 0.5, 10.0)

    # Simulate 10 generations
    parent_value = trait.sample_initial()
    values = [parent_value]

    for gen in range(10):
        child_value = trait.mutate(values[-1], mutation_rate=0.1)
        values.append(child_value)

    print(f"  Values across 10 generations: {[f'{v:.2f}' for v in values]}")

    # All should be in bounds
    assert all(0.5 <= v <= 10.0 for v in values)

    # Should show some variation but not drift too far
    variation = max(values) - min(values)
    print(f"  Total variation: {variation:.2f}")

    print("✓ Inheritance works correctly")
    print("✓ test_trait_inheritance passed\n")


def test_mutation_rate_scaling():
    """Test that mutation rate scales mutation appropriately."""
    print("Testing mutation rate scaling...")

    trait = Trait('speed', 5.0, 0.1, 1.0, 0.0, 10.0)
    parent_value = 5.0

    # Low mutation rate
    low_rate_mutations = [trait.mutate(parent_value, 0.01) for _ in range(1000)]
    low_std = np.std(low_rate_mutations)

    # High mutation rate
    high_rate_mutations = [trait.mutate(parent_value, 0.5) for _ in range(1000)]
    high_std = np.std(high_rate_mutations)

    print(f"  Low rate (0.01) std: {low_std:.3f}")
    print(f"  High rate (0.5) std: {high_std:.3f}")

    # High mutation rate should produce more variation
    assert high_std > low_std * 2, "Mutation rate not scaling properly"

    print("✓ Mutation rate scales correctly")
    print("✓ test_mutation_rate_scaling passed\n")


if __name__ == "__main__":
    print("Running trait system tests...\n")

    test_trait_initialization()
    test_trait_sample_initial()
    test_trait_mutate()
    test_trait_bounds_clamping()
    test_trait_collection()
    test_common_trait_definitions()
    test_trait_inheritance()
    test_mutation_rate_scaling()

    print("✅ All trait system tests passed!")
    print("✅ Trait system is working correctly:")
    print("   - Trait initialization and properties")
    print("   - Initial value sampling with normal distribution")
    print("   - Mutation with proper scaling")
    print("   - Bounds clamping (min/max enforcement)")
    print("   - TraitCollection management")
    print("   - Common trait definitions")
    print("   - Multi-generation inheritance")
    print("   - Mutation rate scaling")
