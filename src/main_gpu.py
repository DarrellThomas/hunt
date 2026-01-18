"""
GPU simulation runner with visualization.

Simplified to use unified renderer from renderer.py.
"""

# Suppress pygame's pkg_resources deprecation warning (pygame issue, not ours)
import warnings
warnings.filterwarnings('ignore', message='.*pkg_resources.*', category=UserWarning)

import numpy as np
from simulation_gpu import GPUEcosystem
from renderer import Renderer, RenderConfig, create_state_from_gpu_ecosystem


def main():
    """Main entry point for GPU simulation with visualization."""
    import pygame

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
    # Base: 3200x2400 = 7,680,000 pixels with 10,000 agents
    base_pixels = 3200 * 2400
    sim_pixels = sim_width * sim_height
    scale_factor = sim_pixels / base_pixels

    num_prey = int(8000 * scale_factor)
    num_predators = int(2000 * scale_factor)

    print(f"Detected monitor: {screen_width}x{screen_height}")
    print(f"Simulation area: {sim_width}x{sim_height} (reserving {stats_panel_height}px for stats)")
    print(f"Creating simulation with {num_prey:,} prey, {num_predators:,} predators")

    # Create GPU simulation at native resolution minus stats panel
    ecosystem = GPUEcosystem(
        width=sim_width,
        height=sim_height,
        num_prey=num_prey,
        num_predators=num_predators,
        device='cuda'
    )

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
    print(f"World: {ecosystem.width}x{ecosystem.height} (native resolution)")
    print(f"Initial: {num_prey:,} prey, {num_predators:,} predators")
    print(f"Device: {ecosystem.device}")
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  S - Save stats to stats_autosave.npz")
    print("  ESC - Quit")
    print(f"Auto-save: Every 100 timesteps")
    print("="*60 + "\n")

    # Main loop
    running = True
    autosave_interval = 100  # Auto-save every 100 timesteps
    try:
        while running:
            # Update simulation (if not paused)
            if not renderer.is_paused():
                ecosystem.step()

                # Auto-save periodically
                if ecosystem.timestep % autosave_interval == 0 and ecosystem.timestep > 0:
                    ecosystem.save_stats()

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

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")

    finally:
        # Final save
        ecosystem.save_stats()

        # Final statistics
        print("\n" + "="*60)
        print("SIMULATION COMPLETE")
        print("="*60)
        final_state = ecosystem.get_state_cpu()
        print(f"Final: {final_state['prey_count']} prey, "
              f"{final_state['pred_count']} predators")
        print(f"Total timesteps: {ecosystem.timestep}")

        renderer.close()


if __name__ == "__main__":
    main()
