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
    # Create GPU simulation
    # Large world: 3200x2400 with 10,000 agents for GPU acceleration
    ecosystem = GPUEcosystem(
        width=3200,
        height=2400,
        num_prey=8000,
        num_predators=2000,
        device='cuda'
    )

    # Create unified renderer
    render_config = RenderConfig(
        width=3200,
        height=2400,
        fullscreen=True,  # GPU version typically runs fullscreen
        target_fps=60,
        show_stats=True
    )
    renderer = Renderer(render_config, river=ecosystem.river)

    # Print startup info
    print("\n" + "="*60)
    print("HUNT GPU - 10,000 Agent Co-Evolution")
    print("="*60)
    print(f"World: {ecosystem.width}x{ecosystem.height}")
    print(f"Initial: {ecosystem.num_prey} prey, {ecosystem.num_predators} predators")
    print(f"Device: {ecosystem.device}")
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  ESC - Quit")
    print("="*60 + "\n")

    # Main loop
    running = True
    try:
        while running:
            # Update simulation (if not paused)
            if not renderer.is_paused():
                ecosystem.step()

                # Print progress
                if ecosystem.timestep % 100 == 0:
                    state = ecosystem.get_state_cpu()
                    print(f"Step {ecosystem.timestep}: "
                          f"{state['prey_count']} prey, "
                          f"{state['pred_count']} predators")

            # Convert ecosystem state to render format
            state = create_state_from_gpu_ecosystem(ecosystem)

            # Render (returns continue_running, save_requested)
            running, _save_requested = renderer.render(state)

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user.")

    finally:
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
