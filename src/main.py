"""
CPU simulation runner with visualization.

Simplified to use unified renderer from renderer.py.
"""

import numpy as np
from world import World
from renderer import Renderer, RenderConfig, create_state_from_cpu_world


def main():
    """Main entry point for CPU simulation with visualization."""
    # Create world
    # 4x larger (2x width, 2x height) - matching original dimensions
    world = World(width=1600, height=1200, initial_prey=200, initial_predators=40)

    # Create unified renderer
    render_config = RenderConfig(
        width=1600,
        height=1200,
        fullscreen=False,
        target_fps=30,
        show_stats=True
    )
    renderer = Renderer(render_config, river=world.river)

    # Simulation parameters
    mutation_rate = 0.1
    print_interval = 500
    steps_per_frame = 1  # 1 = real-time, higher = fast-forward

    # Print startup info
    print("\n" + "="*60)
    print("HUNT - Predator-Prey Co-Evolution")
    print("="*60)
    print(f"Initial Population: {len(world.prey)} prey, {len(world.predators)} predators")
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  S - Save statistics")
    print("  ESC - Quit")
    print("="*60 + "\n")

    # Main loop
    running = True
    try:
        while running:
            # Update simulation (if not paused)
            if not renderer.is_paused():
                for _ in range(steps_per_frame):
                    world.step(mutation_rate)

                    # Print stats periodically
                    if world.timestep % print_interval == 0:
                        world.print_stats()

                    # Check if ecosystem collapsed
                    if len(world.prey) == 0 or len(world.predators) == 0:
                        print("\n!!! ECOSYSTEM COLLAPSED !!!")
                        print(f"Final state: {len(world.prey)} prey, {len(world.predators)} predators")
                        running = False
                        break

            # Convert world state to render format
            state = create_state_from_cpu_world(world)

            # Render (returns continue_running, save_requested)
            running, save_requested = renderer.render(state)

            # Save stats if requested
            if save_requested:
                world.save_stats()
                print(f"Stats saved at timestep {world.timestep}")

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")

    finally:
        # Final statistics
        print("\n" + "="*60)
        print("SIMULATION COMPLETE")
        print("="*60)
        world.print_stats()
        print(f"\nTotal timesteps: {world.timestep}")

        # Save statistics
        world.save_stats()
        print("Final stats saved to stats.npz")

        renderer.close()


if __name__ == "__main__":
    main()
