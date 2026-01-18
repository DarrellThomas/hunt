"""
GPU simulation runner with visualization.

Simplified to use unified renderer from renderer.py.
"""

# Suppress pygame's pkg_resources deprecation warning (pygame issue, not ours)
import warnings
warnings.filterwarnings('ignore', message='.*pkg_resources.*', category=UserWarning)

import argparse
import numpy as np
from simulation_gpu import GPUEcosystem
from renderer import Renderer, RenderConfig, create_state_from_gpu_ecosystem
from config import INITIAL_PREY_POPULATION, INITIAL_PREDATOR_POPULATION


def main():
    """Main entry point for GPU simulation with visualization."""
    import pygame

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='HUNT GPU Simulation - Large-Scale Predator-Prey Co-Evolution with Brain Persistence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls (during simulation):
  SPACE       - Pause/Resume
  S           - Save statistics to stats_autosave.npz
  B           - Save brain checkpoint (neural network weights)
  ESC         - Quit (auto-saves stats and brains)

Auto-Save:
  Stats:  Every 100 timesteps
  Brains: Every 10,000 timesteps
  Both automatically saved on exit

Examples:
  # Basic run
  python main_gpu.py

  # Named experiment with title
  python main_gpu.py --title="Island Refuge Experiment"

  # Named brain checkpoints
  python main_gpu.py --title="High Mutation" --brain-checkpoint=highmut
  # Saves: brains_highmut_autosave_10000.npz, brains_highmut_autosave_20000.npz, etc.

  # Resume from checkpoint
  python main_gpu.py --load-brains=brains_step_50000.npz --title="Island Refuge (continued)"

  # Combination: resume with named checkpoints
  python main_gpu.py --load-brains=brains_highmut_final_75000.npz --brain-checkpoint=highmut_v2

Analysis Tools:
  # View evolution graphs
  python analyze_evolution.py stats_autosave.npz --title="Custom Title"

  # Inspect brain checkpoint
  python analyze_brains.py brains_step_100000.npz --verbose

  # Compare two checkpoints
  python analyze_brains.py brains_early.npz --compare brains_late.npz

Features:
  - GPU-accelerated: 10,000+ agents at 60 FPS
  - Fullscreen: Auto-adapts to monitor resolution
  - Brain persistence: Save/load evolved neural networks
  - Metadata tracking: All parameters saved for reproducibility
  - Dynamic populations: Can grow 3x beyond initial (configurable)
        """
    )

    parser.add_argument('--title', type=str, default='Untitled Run',
                        help='Title for this experimental run (saved in metadata and stats)')
    parser.add_argument('--load-brains', type=str, default=None,
                        help='Load brain weights from .npz file to continue evolution (e.g., brains_step_50000.npz)')
    parser.add_argument('--brain-checkpoint', type=str, default=None,
                        help='Name prefix for brain checkpoint saves (e.g., "island" creates brains_island_autosave_10000.npz)')
    args = parser.parse_args()

    # Detect native monitor resolution for true fullscreen
    pygame.init()
    display_info = pygame.display.Info()
    screen_width = display_info.current_w
    screen_height = display_info.current_h
    pygame.quit()  # Close init, renderer will re-init

    # Reserve space for stats panel (100 pixels)
    stats_panel_height = 100
    sim_width = screen_width
    sim_height = screen_height - stats_panel_height

    # Calculate agent counts proportional to screen area
    # Use config values as baseline, scale to screen size
    base_pixels = 3200 * 2400  # Reference screen size
    sim_pixels = sim_width * sim_height
    scale_factor = sim_pixels / base_pixels

    # Scale initial populations from config based on screen area
    num_prey = int(INITIAL_PREY_POPULATION * scale_factor)
    num_predators = int(INITIAL_PREDATOR_POPULATION * scale_factor)

    print(f"Detected monitor: {screen_width}x{screen_height}")
    print(f"Simulation area: {sim_width}x{sim_height} (reserving {stats_panel_height}px for stats)")
    print(f"Creating simulation with {num_prey:,} prey, {num_predators:,} predators")

    # Create GPU simulation at native resolution minus stats panel
    # Note: Capacity is POPULATION_CAPACITY_MULTIPLIER × initial population (default 3x)
    # This allows populations to grow/shrink naturally beyond starting values
    ecosystem = GPUEcosystem(
        width=sim_width,
        height=sim_height,
        num_prey=num_prey,
        num_predators=num_predators,
        device='cuda'
    )

    # Set run title from CLI argument
    ecosystem.metadata['run_title'] = args.title

    # Load brain weights if specified
    if args.load_brains:
        ecosystem.load_brain_weights(args.load_brains)
        print(f"\n▶ Resuming evolution from {args.load_brains}")

    # Create unified renderer (fullscreen at native resolution)
    render_config = RenderConfig(
        width=sim_width,
        height=sim_height,
        fullscreen=True,
        target_fps=60,
        show_stats=True
    )
    renderer = Renderer(render_config, river=ecosystem.river)

    # Print startup info
    print("\n" + "="*60)
    print(f"HUNT GPU - {num_prey + num_predators:,} Agent Co-Evolution")
    print("="*60)
    print(f"Run Title: {args.title}")
    print(f"World: {ecosystem.width}x{ecosystem.height} (native resolution)")
    print(f"Initial: {num_prey:,} prey, {num_predators:,} predators")
    print(f"Device: {ecosystem.device}")
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  S - Save stats to stats_autosave.npz")
    print("  B - Save brain weights checkpoint")
    print("  ESC - Quit")
    print(f"Auto-save stats: Every 100 timesteps")
    print(f"Auto-save brains: Every 10,000 timesteps")
    print("="*60 + "\n")

    # Main loop
    running = True
    autosave_stats_interval = 100  # Auto-save stats every 100 timesteps
    autosave_brains_interval = 10000  # Auto-save brains every 10,000 timesteps
    brain_save_requested = False  # Flag for B key press

    try:
        while running:
            # Update simulation (if not paused)
            if not renderer.is_paused():
                ecosystem.step()

                # Check for extinction - stop if experiment is over
                if ecosystem.extinct:
                    ecosystem.save_stats()  # Final save
                    print("\n" + "="*60)
                    print("EXTINCTION DETECTED - Stopping simulation")
                    print("="*60)
                    running = False
                    break

                # Auto-save stats periodically
                if ecosystem.timestep % autosave_stats_interval == 0 and ecosystem.timestep > 0:
                    ecosystem.save_stats()

                # Auto-save brains periodically
                if ecosystem.timestep % autosave_brains_interval == 0 and ecosystem.timestep > 0:
                    brain_name = args.brain_checkpoint or f'autosave_{ecosystem.timestep}'
                    ecosystem.save_brain_weights(brain_name)

                # Print progress
                if ecosystem.timestep % 100 == 0:
                    state = ecosystem.get_state_cpu()
                    print(f"Step {ecosystem.timestep}: "
                          f"{state['prey_count']} prey, "
                          f"{state['pred_count']} predators")

            # Convert ecosystem state to render format
            state = create_state_from_gpu_ecosystem(ecosystem)

            # Render (returns continue_running, save_requested)
            running, save_requested = renderer.render(state)

            # Manual save on user request (S key)
            if save_requested:
                ecosystem.save_stats()

            # Manual brain save on B key (we'll need to add this to renderer)
            # For now, check pygame events directly
            import pygame
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_b:
                    brain_name = args.brain_checkpoint or f'manual_{ecosystem.timestep}'
                    ecosystem.save_brain_weights(brain_name)
                    print(f"✓ Manual brain save triggered")

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")

    finally:
        # Final saves
        ecosystem.save_stats()

        # Save final brain state
        final_brain_name = args.brain_checkpoint or f'final_{ecosystem.timestep}'
        ecosystem.save_brain_weights(final_brain_name)

        # Final statistics
        print("\n" + "="*60)
        print("SIMULATION COMPLETE")
        print("="*60)
        final_state = ecosystem.get_state_cpu()
        print(f"Final: {final_state['prey_count']} prey, "
              f"{final_state['pred_count']} predators")
        print(f"Total timesteps: {ecosystem.timestep}")
        print(f"Run title: {args.title}")

        renderer.close()


if __name__ == "__main__":
    main()
