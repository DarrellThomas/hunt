# HUNT Platform: River & Island Visual Polish

**Budget**: ~$20
**Scope**: Fix rendering artifacts and achieve professional-quality river/island visuals

## Problem Statement

The current river and island rendering has several visual issues (see attached screenshots):

### Issue 1: River Fill Gaps
The blue river color doesn't extend to the actual river boundaries. There's a visible dark gap between the water and the edges.

### Issue 2: Island Rendered as Overlapping Circles
The island appears as a series of overlapping tan/brown circles rather than a smooth, continuous landmass. This looks unprofessional and cartoonish.

### Issue 3: Discontinuities at Split Points
Where the river splits to flow around the island, there are harsh rectangular artifacts and discontinuities. The transition from single channel to split channels is not smooth.

### Issue 4: Overall Cartoonish Appearance
The rendering lacks the polish expected from a scientific simulation platform. Curves are jagged, colors are flat, and the overall aesthetic is primitive.

## Visual Reference

What we want:
- Smooth, continuous river banks (no gaps, no jagged edges)
- Island as a single smooth polygon/shape (not overlapping circles)
- Seamless transitions where river splits and rejoins
- Natural-looking curves throughout
- Optional: subtle gradients or edge effects for depth

## Current Implementation Analysis

Based on the audit, the river rendering likely works like this:

```python
# river.py - Path generation
def _generate_path(self):
    # Creates discrete points along river centerline
    self.path_x = np.array([...])  # ~50 points
    self.path_y = np.array([...])
    
# Rendering (in visualizer)
def draw_river(self):
    # Likely drawing circles at each path point
    for i in range(len(river.path_x)):
        pygame.draw.circle(surface, BLUE, (path_x[i], path_y[i]), river_width/2)
    
    # Island probably similar - circles at path points in split region
    for i in range(split_start_idx, split_end_idx):
        pygame.draw.circle(surface, TAN, (path_x[i], path_y[i]), island_width/2)
```

This approach creates the overlapping circles artifact.

## Solution: Polygon-Based Rendering

Replace circle-based rendering with proper polygon rendering:

### Step 1: Generate River Bank Polygons

```python
def get_river_polygons(self) -> dict:
    """Generate polygons for river rendering.
    
    Returns:
        dict with keys:
            'river_polygon': List of (x,y) points forming river outline
            'island_polygon': List of (x,y) points forming island outline (or None)
    """
    # Calculate perpendicular offsets at each path point
    river_left_bank = []
    river_right_bank = []
    
    for i in range(len(self.path_x)):
        # Get tangent direction at this point
        if i == 0:
            dx = self.path_x[1] - self.path_x[0]
            dy = self.path_y[1] - self.path_y[0]
        elif i == len(self.path_x) - 1:
            dx = self.path_x[-1] - self.path_x[-2]
            dy = self.path_y[-1] - self.path_y[-2]
        else:
            # Average of forward and backward tangents for smoothness
            dx = (self.path_x[i+1] - self.path_x[i-1]) / 2
            dy = (self.path_y[i+1] - self.path_y[i-1]) / 2
        
        # Normalize
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx, dy = dx/length, dy/length
        
        # Perpendicular vector (rotate 90 degrees)
        perp_x, perp_y = -dy, dx
        
        # Calculate bank positions
        half_width = self.width / 2
        
        # Check if we're in the split region
        t = i / len(self.path_x)
        if self.split and self.split_start <= t <= self.split_end:
            # In split region - river flows around island
            # Outer banks are wider, inner banks form island edge
            outer_offset = half_width + self.island_width / 2
            inner_offset = self.island_width / 2
            
            river_left_bank.append((
                self.path_x[i] + perp_x * outer_offset,
                self.path_y[i] + perp_y * outer_offset
            ))
            river_right_bank.append((
                self.path_x[i] - perp_x * outer_offset,
                self.path_y[i] - perp_y * outer_offset
            ))
        else:
            # Normal river section
            river_left_bank.append((
                self.path_x[i] + perp_x * half_width,
                self.path_y[i] + perp_y * half_width
            ))
            river_right_bank.append((
                self.path_x[i] - perp_x * half_width,
                self.path_y[i] - perp_y * half_width
            ))
    
    # Create closed polygon: left bank forward, right bank backward
    river_polygon = river_left_bank + river_right_bank[::-1]
    
    # Generate island polygon if split enabled
    island_polygon = None
    if self.split:
        island_polygon = self._generate_island_polygon()
    
    return {
        'river_polygon': river_polygon,
        'island_polygon': island_polygon
    }

def _generate_island_polygon(self) -> list:
    """Generate smooth island outline as a single polygon."""
    island_points = []
    
    # Find indices for split region
    start_idx = int(self.split_start * len(self.path_x))
    end_idx = int(self.split_end * len(self.path_x))
    
    # Generate points along top edge of island (left to right)
    for i in range(start_idx, end_idx + 1):
        # Get perpendicular direction
        if i == start_idx:
            dx = self.path_x[i+1] - self.path_x[i]
            dy = self.path_y[i+1] - self.path_y[i]
        elif i == end_idx:
            dx = self.path_x[i] - self.path_x[i-1]
            dy = self.path_y[i] - self.path_y[i-1]
        else:
            dx = (self.path_x[i+1] - self.path_x[i-1]) / 2
            dy = (self.path_y[i+1] - self.path_y[i-1]) / 2
        
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx, dy = dx/length, dy/length
        perp_x, perp_y = -dy, dx
        
        # Taper island at ends for smooth transition
        t_local = (i - start_idx) / (end_idx - start_idx)  # 0 to 1 within island
        taper = np.sin(t_local * np.pi)  # 0 at ends, 1 in middle
        
        half_island = (self.island_width / 2) * taper
        
        island_points.append((
            self.path_x[i] + perp_x * half_island,
            self.path_y[i] + perp_y * half_island
        ))
    
    # Generate points along bottom edge (right to left)
    for i in range(end_idx, start_idx - 1, -1):
        if i == start_idx:
            dx = self.path_x[i+1] - self.path_x[i]
            dy = self.path_y[i+1] - self.path_y[i]
        elif i == end_idx:
            dx = self.path_x[i] - self.path_x[i-1]
            dy = self.path_y[i] - self.path_y[i-1]
        else:
            dx = (self.path_x[i+1] - self.path_x[i-1]) / 2
            dy = (self.path_y[i+1] - self.path_y[i-1]) / 2
        
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            dx, dy = dx/length, dy/length
        perp_x, perp_y = -dy, dx
        
        t_local = (i - start_idx) / (end_idx - start_idx)
        taper = np.sin(t_local * np.pi)
        half_island = (self.island_width / 2) * taper
        
        island_points.append((
            self.path_x[i] - perp_x * half_island,
            self.path_y[i] - perp_y * half_island
        ))
    
    return island_points
```

### Step 2: Update Visualizer Rendering

```python
# In main.py / main_gpu.py visualizer

def draw_river(self, surface, river):
    """Draw river and island using smooth polygons."""
    if not river or not river.enabled:
        return
    
    # Get pre-computed polygons (cache these, don't recompute every frame)
    if not hasattr(self, '_river_polygons') or self._river_polygons is None:
        self._river_polygons = river.get_river_polygons()
    
    polygons = self._river_polygons
    
    # Draw river as filled polygon
    river_color = (65, 105, 175)  # Steel blue - more natural than pure blue
    if len(polygons['river_polygon']) >= 3:
        pygame.draw.polygon(surface, river_color, polygons['river_polygon'])
        
        # Optional: draw subtle darker edge for depth
        edge_color = (45, 85, 145)
        pygame.draw.polygon(surface, edge_color, polygons['river_polygon'], width=2)
    
    # Draw island as filled polygon
    if polygons['island_polygon'] and len(polygons['island_polygon']) >= 3:
        island_color = (160, 140, 100)  # Muted tan/brown
        pygame.draw.polygon(surface, island_color, polygons['island_polygon'])
        
        # Optional: darker edge for definition
        island_edge = (130, 110, 70)
        pygame.draw.polygon(surface, island_edge, polygons['island_polygon'], width=2)
        
        # Optional: beach/shore effect - slightly lighter inner area
        # This would require generating a smaller inner polygon
```

### Step 3: Smooth the Path Generation

Increase path resolution and use spline interpolation for smoother curves:

```python
def _generate_path(self, world_width, world_height):
    """Generate smooth river path using spline interpolation."""
    from scipy.interpolate import splprep, splev
    
    # Generate control points (fewer points, will be interpolated)
    num_control_points = 10
    control_x = np.linspace(0, world_width, num_control_points)
    
    # Add curviness with sine waves
    base_y = world_height / 2
    control_y = base_y + np.sin(np.linspace(0, 2*np.pi, num_control_points)) * \
                (world_height * self.curviness)
    
    # Fit spline to control points
    tck, u = splprep([control_x, control_y], s=0, k=3)  # Cubic spline
    
    # Evaluate spline at many points for smooth curve
    num_path_points = 200  # More points = smoother rendering
    u_new = np.linspace(0, 1, num_path_points)
    smooth_path = splev(u_new, tck)
    
    self.path_x = smooth_path[0]
    self.path_y = smooth_path[1]
    
    # Compute tangent directions for each point
    self._compute_tangents()

def _compute_tangents(self):
    """Precompute normalized tangent vectors at each path point."""
    self.tangent_x = np.zeros(len(self.path_x))
    self.tangent_y = np.zeros(len(self.path_y))
    
    for i in range(len(self.path_x)):
        if i == 0:
            dx = self.path_x[1] - self.path_x[0]
            dy = self.path_y[1] - self.path_y[0]
        elif i == len(self.path_x) - 1:
            dx = self.path_x[-1] - self.path_x[-2]
            dy = self.path_y[-1] - self.path_y[-2]
        else:
            dx = self.path_x[i+1] - self.path_x[i-1]
            dy = self.path_y[i+1] - self.path_y[i-1]
        
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            self.tangent_x[i] = dx / length
            self.tangent_y[i] = dy / length
```

### Step 4: Handle Edge Cases

```python
def get_river_polygons(self):
    """Generate polygons with proper handling of split transitions."""
    
    # ... (bank generation code) ...
    
    # CRITICAL: Smooth transition at split start/end
    # Instead of abrupt width change, taper over several points
    
    transition_points = 10  # Number of points to transition over
    
    for i in range(len(self.path_x)):
        t = i / len(self.path_x)
        
        # Calculate distance from split boundaries
        dist_to_split_start = abs(t - self.split_start)
        dist_to_split_end = abs(t - self.split_end)
        
        # Smooth transition factor (0 = normal river, 1 = full split)
        if t < self.split_start:
            # Approaching split from left
            transition_zone = self.split_start - 0.05  # 5% transition zone
            if t > transition_zone:
                split_factor = (t - transition_zone) / (self.split_start - transition_zone)
            else:
                split_factor = 0
        elif t > self.split_end:
            # Past split on right
            transition_zone = self.split_end + 0.05
            if t < transition_zone:
                split_factor = 1 - (t - self.split_end) / (transition_zone - self.split_end)
            else:
                split_factor = 0
        else:
            # In split region
            split_factor = 1.0
        
        # Use split_factor to interpolate between normal and split widths
        normal_half_width = self.width / 2
        split_half_width = self.width / 2 + self.island_width / 2
        
        effective_half_width = normal_half_width + split_factor * (split_half_width - normal_half_width)
        
        # Apply to bank positions...
```

## Deliverables

1. **Modified `river.py`**:
   - `get_river_polygons()` method returning proper polygon outlines
   - `_generate_island_polygon()` with tapered ends
   - Increased path resolution (200+ points)
   - Optional: spline interpolation for smoother curves
   - Smooth transitions at split boundaries

2. **Modified `main.py` and `main_gpu.py`**:
   - Replace circle-based rendering with `pygame.draw.polygon()`
   - Cache polygon data (don't recompute every frame)
   - Improved color palette (more natural blues/tans)
   - Optional: edge highlighting for depth

3. **Visual verification**:
   - No gaps between river fill and boundaries
   - Island is single smooth shape (no overlapping circles)
   - Smooth transitions where river splits/rejoins
   - Professional appearance suitable for scientific visualization

## Color Palette Suggestions

```python
# More natural, professional colors
RIVER_COLORS = {
    'water': (65, 105, 175),      # Steel blue
    'water_dark': (45, 85, 145),  # Darker edge
    'water_light': (85, 125, 195), # Shallow water (optional)
}

ISLAND_COLORS = {
    'land': (160, 140, 100),      # Sandy tan
    'land_edge': (130, 110, 70),  # Darker border
    'vegetation': (90, 120, 70),  # Optional green areas
}

BACKGROUND = (20, 25, 35)  # Dark blue-gray (current looks fine)
```

## Testing

1. **Visual inspection**: Run simulation and verify:
   - River banks are smooth and continuous
   - Island is a single cohesive shape
   - No artifacts at split transitions
   - Colors look natural and professional

2. **Edge cases**:
   - River with `split=False` (no island)
   - River with high curviness
   - River with very wide/narrow settings
   - Different world sizes

3. **Performance**: Polygon rendering should be faster than drawing hundreds of circles

## Optional Enhancements (If Time Permits)

1. **Gradient fills**: Water darker in center, lighter at edges
2. **Shore effect**: Thin lighter band where water meets land
3. **Flow visualization**: Subtle animated lines showing current direction
4. **Vegetation on island**: Small green dots/patches

## Begin

1. Read current `river.py` to understand path generation
2. Read visualizer code to understand current rendering approach
3. Implement `get_river_polygons()` in river.py
4. Update visualizer to use polygon rendering
5. Test and iterate on visual quality
