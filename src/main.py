"""
Main training loop with live Pygame visualization.
Watch predators and prey co-evolve in real-time.
"""

import pygame
import numpy as np
import sys
from world import World


class Visualizer:
    """Pygame visualizer for the ecosystem."""

    def __init__(self, world, screen_width=800, screen_height=600, fps=30):
        """
        Initialize the visualizer.

        Args:
            world: World instance to visualize
            screen_width, screen_height: Display dimensions
            fps: Target frames per second
        """
        pygame.init()
        self.world = world
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fps = fps

        self.screen = pygame.display.set_mode((screen_width, screen_height + 100))  # Extra space for stats
        pygame.display.set_caption("HUNT - Predator-Prey Co-Evolution")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        # Colors
        self.bg_color = (20, 20, 30)
        self.prey_color = (50, 255, 50)
        self.predator_color = (255, 50, 50)
        self.text_color = (200, 200, 200)
        self.ui_bg_color = (30, 30, 40)

        # For tracking deaths and births
        self.birth_effects = []
        self.death_effects = []

    def draw_river(self):
        """Draw the river using smooth polygon rendering."""
        if not self.world.river.enabled:
            return

        # Get or cache polygon data (only compute once, not every frame)
        if not hasattr(self, '_river_polygons') or self._river_polygons is None:
            self._river_polygons = self.world.river.get_river_polygons()

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
        """Draw the current state of the world."""
        # Clear screen
        self.screen.fill(self.bg_color)

        # Draw river first (so agents appear on top)
        self.draw_river()

        # Draw prey
        for prey in self.world.prey:
            x = int(prey.pos[0])
            y = int(prey.pos[1])
            # Size based on age (younger = smaller)
            size = min(5, 3 + prey.age // 100)
            pygame.draw.circle(self.screen, self.prey_color, (x, y), size)

        # Draw predators
        for predator in self.world.predators:
            x = int(predator.pos[0])
            y = int(predator.pos[1])
            # Size based on energy
            size = max(4, int(6 * predator.energy / predator.max_energy))
            pygame.draw.circle(self.screen, self.predator_color, (x, y), size)

            # Draw hunger indicator (circle around predator if hungry)
            if predator.energy < 30:
                pygame.draw.circle(self.screen, (255, 200, 0), (x, y), 10, 1)

        # Draw UI panel
        self.draw_ui()

        pygame.display.flip()

    def draw_ui(self):
        """Draw UI panel with statistics."""
        # UI background
        ui_y = self.screen_height
        pygame.draw.rect(self.screen, self.ui_bg_color, (0, ui_y, self.screen_width, 100))

        # Statistics
        stats_text = [
            f"Timestep: {self.world.timestep}",
            f"Prey: {len(self.world.prey)}",
            f"Predators: {len(self.world.predators)}",
        ]

        if len(self.world.prey) > 0:
            avg_prey_age = np.mean([p.age for p in self.world.prey])
            stats_text.append(f"Prey Avg Age: {avg_prey_age:.1f}")

        if len(self.world.predators) > 0:
            avg_predator_age = np.mean([p.age for p in self.world.predators])
            avg_predator_energy = np.mean([p.energy for p in self.world.predators])
            stats_text.append(f"Predator Avg Age: {avg_predator_age:.1f}")
            stats_text.append(f"Predator Avg Energy: {avg_predator_energy:.1f}")

        # Draw stats in columns
        x_offset = 10
        y_offset = ui_y + 10
        for i, text in enumerate(stats_text):
            if i == 3:  # Start second column
                x_offset = 300
                y_offset = ui_y + 10
            surface = self.small_font.render(text, True, self.text_color)
            self.screen.blit(surface, (x_offset, y_offset))
            y_offset += 20

        # Instructions
        instructions = "SPACE: Pause | ESC: Quit | S: Save Stats"
        surface = self.small_font.render(instructions, True, self.text_color)
        self.screen.blit(surface, (self.screen_width - 350, ui_y + 70))

    def handle_events(self):
        """
        Handle pygame events.

        Returns:
            (running, paused): Tuple of booleans
        """
        paused = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False, False
                elif event.key == pygame.K_SPACE:
                    paused = True
                elif event.key == pygame.K_s:
                    self.world.save_stats()

        return True, paused

    def run(self, max_timesteps=None, mutation_rate=0.1, steps_per_frame=1, print_interval=100):
        """
        Run the simulation with visualization.

        Args:
            max_timesteps: Maximum timesteps to run (None = infinite)
            mutation_rate: Mutation rate for offspring
            steps_per_frame: Number of simulation steps per rendered frame
            print_interval: How often to print stats
        """
        running = True
        paused = False

        print("\n" + "="*60)
        print("HUNT - Predator-Prey Co-Evolution")
        print("="*60)
        print(f"Initial Population: {len(self.world.prey)} prey, {len(self.world.predators)} predators")
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  S - Save statistics")
        print("  ESC - Quit")
        print("="*60 + "\n")

        try:
            while running:
                # Handle events
                running, pause_toggle = self.handle_events()
                if pause_toggle:
                    paused = not paused

                # Update simulation (if not paused)
                if not paused:
                    for _ in range(steps_per_frame):
                        self.world.step(mutation_rate)

                        # Print stats periodically
                        if self.world.timestep % print_interval == 0:
                            self.world.print_stats()

                        # Check termination condition
                        if max_timesteps and self.world.timestep >= max_timesteps:
                            running = False
                            break

                        # Check if ecosystem collapsed
                        if len(self.world.prey) == 0 or len(self.world.predators) == 0:
                            print("\n!!! ECOSYSTEM COLLAPSED !!!")
                            print(f"Final state: {len(self.world.prey)} prey, {len(self.world.predators)} predators")
                            running = False
                            break

                # Draw
                self.draw()
                self.clock.tick(self.fps)

        except KeyboardInterrupt:
            print("\n\nSimulation interrupted by user.")

        finally:
            # Final statistics
            print("\n" + "="*60)
            print("SIMULATION COMPLETE")
            print("="*60)
            self.world.print_stats()
            print(f"\nTotal timesteps: {self.world.timestep}")

            # Save statistics
            self.world.save_stats()

            pygame.quit()


def main():
    """Main entry point."""
    # Create world - 4x larger (2x width, 2x height)
    world = World(width=1600, height=1200, initial_prey=200, initial_predators=40)

    # Create visualizer
    viz = Visualizer(world, screen_width=1600, screen_height=1200, fps=30)

    # Run simulation
    # mutation_rate: How much offspring differ from parents
    # steps_per_frame: Increase for faster training (but less watchable)
    viz.run(
        max_timesteps=None,  # Run forever until closed
        mutation_rate=0.1,
        steps_per_frame=1,  # 1 = real-time, higher = fast-forward
        print_interval=500
    )


if __name__ == "__main__":
    main()
