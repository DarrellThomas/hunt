"""Unified rendering system for HUNT simulation.

Eliminates code duplication between CPU and GPU visualizers.
Both simulation types produce a standardized SimulationState that the renderer accepts.
"""

import pygame
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class RenderConfig:
    """Configuration for renderer."""
    width: int
    height: int
    fullscreen: bool = False
    target_fps: int = 30
    show_stats: bool = True

    # Colors
    background_color: Tuple[int, int, int] = (20, 25, 35)

    # Agent rendering
    agent_radius: int = 3

    # Species colors (can be extended for N species)
    species_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)

    def __post_init__(self):
        # Set default colors if none provided
        if not self.species_colors:
            self.species_colors = {
                'prey': (50, 255, 50),      # Green
                'predator': (255, 50, 50),  # Red
            }


@dataclass
class SimulationState:
    """Standardized state format that renderer accepts.

    Both CPU and GPU simulations produce this format.
    """
    timestep: int

    # Agent positions by species: {'prey': np.array([[x,y], ...]), 'predator': ...}
    positions: Dict[str, np.ndarray]

    # Population counts
    populations: Dict[str, int]

    # Optional stats (for detailed display)
    stats: Optional[Dict[str, Any]] = None


class Renderer:
    """Unified renderer for HUNT simulation.

    Handles all pygame rendering logic for both CPU and GPU simulations.
    """

    def __init__(self, config: RenderConfig, river=None):
        """Initialize renderer.

        Args:
            config: Rendering configuration
            river: Optional River instance for environmental rendering
        """
        pygame.init()

        self.config = config
        self.river = river

        # Set up display
        flags = pygame.FULLSCREEN if config.fullscreen else 0

        # Add space for stats panel at bottom
        display_height = config.height + 100 if config.show_stats else config.height
        self.screen = pygame.display.set_mode((config.width, display_height), flags)
        pygame.display.set_caption("HUNT - Predator-Prey Co-Evolution")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

        # Cache river polygons (computed once)
        self._river_polygons = None
        if river and river.enabled:
            self._river_polygons = river.get_river_polygons()

        # Pause state
        self._paused = False

    def render(self, state: SimulationState) -> Tuple[bool, bool]:
        """Render a single frame.

        Args:
            state: Current simulation state

        Returns:
            Tuple of (continue_running, save_requested)
            continue_running: False if user closed window or pressed ESC
            save_requested: True if user pressed 'S' to save stats
        """
        # Handle events
        save_requested = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False, False
                elif event.key == pygame.K_SPACE:
                    self._paused = not self._paused
                elif event.key == pygame.K_s:
                    save_requested = True

        # Clear screen
        self.screen.fill(self.config.background_color)

        # Draw river and island (background layer)
        self._draw_river()

        # Draw agents
        self._draw_agents(state.positions)

        # Draw stats overlay
        if self.config.show_stats:
            self._draw_stats(state)

        # Flip display
        pygame.display.flip()
        self.clock.tick(self.config.target_fps)

        return True, save_requested

    def _draw_river(self):
        """Draw river and island using smooth polygons."""
        if self._river_polygons is None:
            return

        # Professional, natural colors
        water_color = (65, 105, 175)      # Steel blue
        water_edge = (45, 85, 145)        # Darker edge for depth
        island_color = (160, 140, 100)    # Sandy tan
        island_edge = (130, 110, 70)      # Darker border

        # Draw river polygon
        river_poly = self._river_polygons.get('river_polygon', [])
        if len(river_poly) >= 3:
            pygame.draw.polygon(self.screen, water_color, river_poly)
            pygame.draw.polygon(self.screen, water_edge, river_poly, width=2)

        # Draw island polygon
        island_poly = self._river_polygons.get('island_polygon', [])
        if island_poly and len(island_poly) >= 3:
            pygame.draw.polygon(self.screen, island_color, island_poly)
            pygame.draw.polygon(self.screen, island_edge, island_poly, width=2)

    def _draw_agents(self, positions: Dict[str, np.ndarray]):
        """Draw all agents by species.

        Args:
            positions: Dictionary mapping species name to position array
        """
        for species_name, pos_array in positions.items():
            if len(pos_array) == 0:
                continue

            # Get color for this species (default to gray if not defined)
            color = self.config.species_colors.get(species_name, (200, 200, 200))

            # Draw all agents of this species
            for pos in pos_array:
                pygame.draw.circle(
                    self.screen, color,
                    (int(pos[0]), int(pos[1])),
                    self.config.agent_radius
                )

    def _draw_stats(self, state: SimulationState):
        """Draw statistics overlay panel.

        Args:
            state: Current simulation state
        """
        # Stats panel background
        ui_y = self.config.height
        pygame.draw.rect(self.screen, (30, 30, 40), (0, ui_y, self.config.width, 100))

        # Build stats list
        stats_text = [f"Timestep: {state.timestep}"]

        # Population counts with color coding
        for species_name, count in state.populations.items():
            stats_text.append(f"{species_name.capitalize()}: {count}")

        # Additional stats if provided
        if state.stats:
            if 'prey_avg_age' in state.stats:
                stats_text.append(f"Prey Avg Age: {state.stats['prey_avg_age']:.1f}")
            if 'pred_avg_age' in state.stats:
                stats_text.append(f"Pred Avg Age: {state.stats['pred_avg_age']:.1f}")
            if 'pred_avg_energy' in state.stats:
                stats_text.append(f"Pred Avg Energy: {state.stats['pred_avg_energy']:.1f}")

        # Draw stats in columns
        x_offset = 10
        y_offset = ui_y + 10
        for i, text in enumerate(stats_text):
            if i == 3:  # Start second column
                x_offset = 300
                y_offset = ui_y + 10

            surface = self.small_font.render(text, True, (200, 200, 200))
            self.screen.blit(surface, (x_offset, y_offset))
            y_offset += 20

        # Instructions
        instructions = "SPACE: Pause | ESC: Quit | S: Save Stats"
        surface = self.small_font.render(instructions, True, (200, 200, 200))
        self.screen.blit(surface, (self.config.width - 350, ui_y + 70))

        # FPS counter
        fps = self.clock.get_fps()
        fps_text = self.small_font.render(f"FPS: {fps:.1f}", True, (200, 200, 200))
        self.screen.blit(fps_text, (self.config.width - 100, ui_y + 10))

        # Pause indicator
        if self._paused:
            pause_text = self.font.render("PAUSED", True, (255, 255, 0))
            self.screen.blit(pause_text, (self.config.width // 2 - 50, ui_y + 40))

    def is_paused(self) -> bool:
        """Check if renderer is paused."""
        return self._paused

    def close(self):
        """Clean up pygame."""
        pygame.quit()


def create_state_from_cpu_world(world) -> SimulationState:
    """Convert CPU World to SimulationState.

    Args:
        world: World instance from world.py

    Returns:
        SimulationState ready for rendering
    """
    positions = {}
    populations = {}

    # Get prey positions and count
    if hasattr(world, 'prey') and world.prey:
        positions['prey'] = np.array([a.pos for a in world.prey])
        populations['prey'] = len(world.prey)
    else:
        positions['prey'] = np.empty((0, 2))
        populations['prey'] = 0

    # Get predator positions and count
    if hasattr(world, 'predators') and world.predators:
        positions['predator'] = np.array([a.pos for a in world.predators])
        populations['predator'] = len(world.predators)
    else:
        positions['predator'] = np.empty((0, 2))
        populations['predator'] = 0

    # Get optional stats
    stats = {}
    if positions['prey'].shape[0] > 0:
        stats['prey_avg_age'] = np.mean([a.age for a in world.prey])
    if positions['predator'].shape[0] > 0:
        stats['pred_avg_age'] = np.mean([a.age for a in world.predators])
        stats['pred_avg_energy'] = np.mean([a.energy for a in world.predators])

    return SimulationState(
        timestep=world.timestep,
        positions=positions,
        populations=populations,
        stats=stats
    )


def create_state_from_gpu_ecosystem(ecosystem) -> SimulationState:
    """Convert GPU Ecosystem to SimulationState.

    Args:
        ecosystem: GPUEcosystem instance from simulation_gpu.py

    Returns:
        SimulationState ready for rendering
    """
    # GPU version already has get_state_cpu() that returns dict
    gpu_state = ecosystem.get_state_cpu()

    positions = {
        'prey': gpu_state['prey_pos'],
        'predator': gpu_state['pred_pos']
    }
    populations = {
        'prey': gpu_state['prey_count'],
        'predator': gpu_state['pred_count']
    }

    # Extract additional stats
    stats = {
        'prey_avg_age': gpu_state.get('prey_avg_age', 0),
        'pred_avg_age': gpu_state.get('pred_avg_age', 0),
        'pred_avg_energy': gpu_state.get('pred_avg_energy', 0),
    }

    return SimulationState(
        timestep=ecosystem.timestep,
        positions=positions,
        populations=populations,
        stats=stats
    )
