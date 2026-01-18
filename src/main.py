"""
CPU simulation runner with visualization.

Simplified to use unified renderer from renderer.py.
"""

# Suppress pygame's pkg_resources deprecation warning (pygame issue, not ours)
import warnings
warnings.filterwarnings('ignore', message='.*pkg_resources.*', category=UserWarning)

import sys
import argparse
import subprocess
import numpy as np
from pathlib import Path
from world import World
from renderer import Renderer, RenderConfig, create_state_from_cpu_world
from config import INITIAL_PREY_POPULATION, INITIAL_PREDATOR_POPULATION


def main():
    """Main entry point for CPU simulation with visualization."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='HUNT CPU Simulation - Predator-Prey Co-Evolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls (during simulation):
  SPACE       - Pause/Resume
  S           - Save statistics to stats.npz
  ESC         - Quit

Examples:
  # Run CPU simulation
  python main.py

  # Run with custom title
  python main.py --title="Test Run"

  # Run GPU version instead (faster, larger populations)
  python main.py --with_gpu

  # Run GPU version with title
  python main.py --with_gpu --title="Island Experiment"

Notes:
  - CPU version: Smaller populations (~200 prey, ~50 predators), windowed
  - GPU version: Large populations (~8000+ prey), fullscreen, brain persistence
  - Use --with_gpu for serious experiments and long runs
        """
    )

    parser.add_argument('--title', type=str, default='CPU Run',
                        help='Title for this experimental run')
    parser.add_argument('--with_gpu', action='store_true',
                        help='Launch GPU version (main_gpu.py) instead - RECOMMENDED for large-scale experiments')

    args = parser.parse_args()

    # If --with_gpu flag is set, launch main_gpu.py instead
    if args.with_gpu:
        print("Launching GPU version (main_gpu.py)...")
        print("(Use 'python main_gpu.py --help' for GPU-specific options)\n")

        # Build command for main_gpu.py, passing through args
        gpu_script = Path(__file__).parent / 'main_gpu.py'
        cmd = [sys.executable, str(gpu_script)]

        # Pass through title if provided
        if args.title != 'CPU Run':  # Only pass if explicitly set
            cmd.extend(['--title', args.title])

        # Execute main_gpu.py
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\nGPU simulation interrupted.")
        return

    # Continue with CPU simulation
    print(f"\n{'='*60}")
    print("HUNT CPU Mode")
    print(f"{'='*60}")
    print(f"Run Title: {args.title}")
    print("Note: For larger populations and GPU acceleration, use --with_gpu")
    print(f"{'='*60}\n")

    # Create world
    # CPU is slower, so use smaller population (2.5% of config values)
    initial_prey = int(INITIAL_PREY_POPULATION * 0.025)  # 8000 * 0.025 = 200
    initial_predators = int(INITIAL_PREDATOR_POPULATION * 0.025)  # 2000 * 0.025 = 50

    world = World(width=1600, height=1200, initial_prey=initial_prey, initial_predators=initial_predators)

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
