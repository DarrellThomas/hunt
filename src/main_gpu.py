"""
GPU-accelerated visualization for massive ecosystem.
3200x2400 world, 10,000 agents.
"""

import pygame
import numpy as np
from simulation_gpu import GPUEcosystem


class GPUVisualizer:
    """Pygame visualizer for GPU ecosystem."""

    def __init__(self, ecosystem, screen_width=3200, screen_height=2400, fps=60):
        pygame.init()
        self.ecosystem = ecosystem
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fps = fps

        # Try fullscreen first, fallback to windowed
        try:
            self.screen = pygame.display.set_mode((screen_width, screen_height + 100), pygame.FULLSCREEN)
        except:
            self.screen = pygame.display.set_mode((screen_width, screen_height + 100))

        pygame.display.set_caption("HUNT GPU - 10,000 Agent Co-Evolution")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        # Colors
        self.bg_color = (10, 10, 15)
        self.prey_color = (50, 255, 50)
        self.predator_color = (255, 50, 50)
        self.text_color = (200, 200, 200)
        self.ui_bg_color = (20, 20, 30)

        # Statistics tracking
        self.stats = {
            'timesteps': [],
            'prey_count': [],
            'pred_count': [],
            'prey_avg_age': [],
            'pred_avg_age': [],
            'pred_avg_energy': [],
            # Swim speed evolution tracking
            'prey_avg_swim': [],
            'prey_std_swim': [],
            'pred_avg_swim': [],
            'pred_std_swim': [],
            # Habitat preference tracking (land vs water specialization)
            'prey_in_river_pct': [],
            'pred_in_river_pct': [],
        }

        print("GPU Visualizer initialized")
        print(f"Display: {screen_width}x{screen_height}")
        print(f"Target FPS: {fps}")
        print(f"Auto-save: Every 1000 steps to stats_autosave.npz")

    def draw_river(self):
        """Draw the river using smooth polygon rendering."""
        if not self.ecosystem.river.enabled:
            return

        # Get or cache polygon data (only compute once, not every frame)
        if not hasattr(self, '_river_polygons') or self._river_polygons is None:
            self._river_polygons = self.ecosystem.river.get_river_polygons()

        if self._river_polygons is None:
            return

        polygons = self._river_polygons

        # More natural, professional colors
        river_color = (65, 105, 175)  # Steel blue - more natural than pure blue
        river_edge_color = (45, 85, 145)  # Darker edge for depth
        island_color = (160, 140, 100)  # Sandy tan
        island_edge_color = (130, 110, 70)  # Darker border

        # Draw river as filled polygon
        if polygons['river_polygon'] and len(polygons['river_polygon']) >= 3:
            pygame.draw.polygon(self.screen, river_color, polygons['river_polygon'])

            # Optional: draw subtle darker edge for depth
            pygame.draw.polygon(self.screen, river_edge_color, polygons['river_polygon'], 2)

        # Draw island as filled polygon
        if polygons['island_polygon'] and len(polygons['island_polygon']) >= 3:
            pygame.draw.polygon(self.screen, island_color, polygons['island_polygon'])

            # Optional: darker edge for definition
            pygame.draw.polygon(self.screen, island_edge_color, polygons['island_polygon'], 2)

    def draw(self):
        """Draw current state."""
        # Clear
        self.screen.fill(self.bg_color)

        # Draw river first (so agents appear on top)
        self.draw_river()

        # Get state from GPU
        state = self.ecosystem.get_state_cpu()

        # Draw prey (small dots)
        prey_positions = state['prey_pos']
        if len(prey_positions) > 0:
            for pos in prey_positions:
                x, y = int(pos[0]), int(pos[1])
                pygame.draw.circle(self.screen, self.prey_color, (x, y), 2)

        # Draw predators (larger dots)
        pred_positions = state['pred_pos']
        if len(pred_positions) > 0:
            for pos in pred_positions:
                x, y = int(pos[0]), int(pos[1])
                pygame.draw.circle(self.screen, self.predator_color, (x, y), 4)

        # Draw UI
        self.draw_ui(state)

        pygame.display.flip()

    def draw_ui(self, state):
        """Draw statistics UI."""
        ui_y = self.screen_height
        pygame.draw.rect(self.screen, self.ui_bg_color, (0, ui_y, self.screen_width, 100))

        # Stats
        stats = [
            f"Timestep: {self.ecosystem.timestep}",
            f"Prey: {state['prey_count']}",
            f"Predators: {state['pred_count']}",
            f"Prey Avg Age: {state['prey_avg_age']:.1f}",
            f"Pred Avg Age: {state['pred_avg_age']:.1f}",
            f"Pred Avg Energy: {state['pred_avg_energy']:.1f}",
        ]

        x_offset = 20
        y_offset = ui_y + 15
        for i, text in enumerate(stats):
            if i == 3:
                x_offset = 500
                y_offset = ui_y + 15
            surface = self.small_font.render(text, True, self.text_color)
            self.screen.blit(surface, (x_offset, y_offset))
            y_offset += 25

        # FPS
        fps_text = f"FPS: {self.clock.get_fps():.1f}"
        surface = self.font.render(fps_text, True, self.text_color)
        self.screen.blit(surface, (self.screen_width - 200, ui_y + 30))

        # Instructions
        instructions = "SPACE: Pause | ESC: Quit | 10,000 agents on RTX 5090"
        surface = self.small_font.render(instructions, True, self.text_color)
        self.screen.blit(surface, (self.screen_width - 700, ui_y + 70))

    def handle_events(self):
        """Handle events."""
        paused = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False, False
                elif event.key == pygame.K_SPACE:
                    paused = True
        return True, paused

    def run(self, steps_per_frame=1, print_interval=500, autosave_interval=1000):
        """Run simulation with automatic data collection."""
        running = True
        paused = False

        print("\n" + "="*70)
        print("HUNT GPU - MAXIMUM SCALE")
        print("="*70)
        print(f"World: {self.ecosystem.width}x{self.ecosystem.height}")
        print(f"Agents: {self.ecosystem.num_prey} prey + {self.ecosystem.num_predators} predators")
        print(f"Total: {self.ecosystem.num_prey + self.ecosystem.num_predators} agents")
        print(f"Device: {self.ecosystem.device}")
        print(f"Stats autosave: Every {autosave_interval} steps")
        print("="*70 + "\n")

        try:
            while running:
                running, pause_toggle = self.handle_events()
                if pause_toggle:
                    paused = not paused

                if not paused:
                    for _ in range(steps_per_frame):
                        self.ecosystem.step(mutation_rate=0.1)

                        # Collect stats every 10 steps
                        if self.ecosystem.timestep % 10 == 0:
                            state = self.ecosystem.get_state_cpu()
                            self.stats['timesteps'].append(self.ecosystem.timestep)
                            self.stats['prey_count'].append(state['prey_count'])
                            self.stats['pred_count'].append(state['pred_count'])
                            self.stats['prey_avg_age'].append(state['prey_avg_age'])
                            self.stats['pred_avg_age'].append(state['pred_avg_age'])
                            self.stats['pred_avg_energy'].append(state['pred_avg_energy'])
                            # Swim speed evolution
                            self.stats['prey_avg_swim'].append(state['prey_avg_swim'])
                            self.stats['prey_std_swim'].append(state['prey_std_swim'])
                            self.stats['pred_avg_swim'].append(state['pred_avg_swim'])
                            self.stats['pred_std_swim'].append(state['pred_std_swim'])
                            # Habitat preference
                            self.stats['prey_in_river_pct'].append(state['prey_in_river_pct'])
                            self.stats['pred_in_river_pct'].append(state['pred_in_river_pct'])

                        # Autosave stats
                        if self.ecosystem.timestep % autosave_interval == 0:
                            np.savez('stats_autosave.npz', **self.stats)
                            print(f"  → Stats autosaved at timestep {self.ecosystem.timestep}")

                        # Print progress
                        if self.ecosystem.timestep % print_interval == 0:
                            state = self.ecosystem.get_state_cpu()
                            print(f"Step {self.ecosystem.timestep}: "
                                  f"{state['prey_count']} prey, "
                                  f"{state['pred_count']} predators, "
                                  f"FPS: {self.clock.get_fps():.1f}")

                self.draw()
                self.clock.tick(self.fps)

        except KeyboardInterrupt:
            print("\n\nSimulation interrupted.")
        finally:
            print("\n" + "="*70)
            print("SIMULATION COMPLETE")
            print("="*70)
            state = self.ecosystem.get_state_cpu()
            print(f"Final: {state['prey_count']} prey, {state['pred_count']} predators")
            print(f"Total timesteps: {self.ecosystem.timestep}")

            # Final save
            np.savez('stats_autosave.npz', **self.stats)
            print(f"\n✓ Final stats saved to: stats_autosave.npz")
            print(f"✓ Total data points collected: {len(self.stats['timesteps'])}")

            pygame.quit()


def main():
    """Entry point."""
    # FULL 4K: 3840x2160 world, 12,000 agents
    ecosystem = GPUEcosystem(
        width=3840,
        height=2160,
        num_prey=9600,  # Scale up proportionally
        num_predators=2400,
        device='cuda'
    )

    viz = GPUVisualizer(ecosystem, screen_width=3840, screen_height=2160, fps=60)
    viz.run(steps_per_frame=1, print_interval=500)


if __name__ == "__main__":
    main()
