# Visualization

MOPS integrates with PyVista for interactive 3D visualization. This enables stress field plots, deformation visualization, and export to ParaView.

## Installation

PyVista is an optional dependency:

```bash
pip install pyvista

# For Jupyter notebook support
pip install pyvista jupyterlab
```

## Quick Start

```python
from mops import solve

results = solve(model)

# Plot von Mises stress (default)
results.plot("von_mises", model=model)

# Plot displacement magnitude
results.plot("displacement_magnitude", model=model)

# Plot deformed shape
results.plot("von_mises", model=model, deformed=True, scale_factor=100)
```

## Available Fields

### Nodal Fields

Fields defined at nodes (point data):

| Field | Description |
|-------|-------------|
| `displacement` | Full displacement vector |
| `displacement_magnitude` | Displacement magnitude |
| `displacement_x` | X-component of displacement |
| `displacement_y` | Y-component of displacement |
| `displacement_z` | Z-component of displacement |

### Element Fields

Fields defined at elements (cell data):

| Field | Description |
|-------|-------------|
| `von_mises` | Von Mises equivalent stress |
| `stress_xx` | Normal stress in X direction |
| `stress_yy` | Normal stress in Y direction |
| `stress_zz` | Normal stress in Z direction |
| `stress_xy` | Shear stress XY |
| `stress_yz` | Shear stress YZ |
| `stress_xz` | Shear stress XZ |
| `principal_1` | Maximum principal stress (σ₁) |
| `principal_2` | Intermediate principal stress (σ₂) |
| `principal_3` | Minimum principal stress (σ₃) |
| `tresca` | Tresca equivalent stress |
| `hydrostatic` | Hydrostatic (mean) stress |

## plot_results() Function

Full signature with all options:

```python
from mops.viz import plot_results

plot_results(
    results,                      # Results from solve()
    model=model,                  # Model (required for in-memory results)
    field="von_mises",           # Field to visualize

    # Appearance
    cmap="jet",                  # Colormap (default: field-appropriate)
    show_edges=True,             # Show mesh edges
    show_scalar_bar=True,        # Show color legend
    scalar_bar_title="Stress",   # Custom legend title
    clim=(0, 100),               # Color limits (auto if None)
    opacity=1.0,                 # Surface transparency (0-1)

    # Deformation
    deformed=False,              # Show deformed shape
    scale_factor=1.0,            # Deformation magnification

    # Window settings
    window_size=(1024, 768),     # Window dimensions
    background="white",          # Background color
    title="My Plot",             # Plot title

    # Output options
    screenshot="plot.png",       # Save to file
    notebook=None,               # Force notebook mode (auto-detect)
    off_screen=False,            # Render without display
    return_plotter=False,        # Return plotter object
)
```

## Colormaps

Default colormaps are chosen based on field type:

| Field Type | Default Colormap |
|------------|------------------|
| `von_mises`, `stress_*`, `tresca` | `jet` |
| `displacement*` | `viridis` |
| `principal*` | `coolwarm` |
| `hydrostatic` | `RdBu_r` |

Override with any Matplotlib colormap:

```python
results.plot("von_mises", model=model, cmap="plasma")
results.plot("displacement_magnitude", model=model, cmap="inferno")
```

## Deformed Shape

Visualize the deformed structure with optional scaling:

```python
# Actual deformation
results.plot("von_mises", model=model, deformed=True)

# Magnified deformation (100x)
results.plot(
    "displacement_magnitude",
    model=model,
    deformed=True,
    scale_factor=100,
    title="Cantilever Deformation (100x)",
)
```

This helps visualize small deformations that wouldn't be visible at true scale.

## Screenshots

Save plots to image files:

```python
# Save while displaying
results.plot("von_mises", model=model, screenshot="stress.png")

# Save without display (off-screen rendering)
results.plot(
    "von_mises",
    model=model,
    screenshot="stress.png",
    off_screen=True,
)
```

Supported formats: PNG, JPG, PDF, SVG (via PyVista).

## Jupyter Notebook Integration

In Jupyter notebooks, plots render inline automatically:

```python
# In notebook
results.plot("von_mises", model=model)  # Renders in cell output
```

For static backends (needed for some environments):

```python
import pyvista as pv
pv.set_jupyter_backend('static')  # Use static images

# Or for interactive (if supported)
pv.set_jupyter_backend('trame')
```

## Export to VTU (ParaView)

Export results to VTU format for advanced visualization in ParaView:

```python
from mops.viz import export_vtu

# Export all standard fields
export_vtu(results, "analysis.vtu", model=model)

# Export specific fields only
export_vtu(
    results,
    "analysis.vtu",
    model=model,
    fields=["displacement", "von_mises", "principal_1"],
)

# Export deformed geometry
export_vtu(
    results,
    "deformed.vtu",
    model=model,
    deformed=True,
    scale_factor=10,
)
```

VTU files can be opened in:
- ParaView (full-featured visualization)
- VisIt
- Mayavi
- Any VTK-compatible tool

## Mesh Visualization

Visualize the mesh before solving:

```python
from mops.viz import plot_mesh

plot_mesh(
    model,
    show_edges=True,       # Mesh edges
    show_nodes=False,      # Node markers
    node_size=5.0,         # Node marker size
    opacity=1.0,           # Surface opacity
    color="lightgray",     # Surface color
    edge_color="black",    # Edge color
)
```

## Advanced Usage

### Custom Plotter

Get the PyVista plotter for custom visualization:

```python
import pyvista as pv

plotter = results.plot(
    "von_mises",
    model=model,
    return_plotter=True,
)

# Add custom elements
plotter.add_axes()
plotter.add_bounding_box()

# Custom camera
plotter.camera_position = "iso"

# Show
plotter.show()
```

### Multiple Subplots

Create comparison plots:

```python
import pyvista as pv

# Create plotter with subplots
plotter = pv.Plotter(shape=(1, 2))

# Left plot: Von Mises
plotter.subplot(0, 0)
plotter.add_title("Von Mises Stress")
# ... add mesh with von_mises data

# Right plot: Displacement
plotter.subplot(0, 1)
plotter.add_title("Displacement")
# ... add mesh with displacement data

plotter.link_views()  # Sync camera between views
plotter.show()
```

### Extracting PyVista Mesh

Create a PyVista mesh for custom processing:

```python
from mops.viz import create_pyvista_mesh

# Get raw PyVista mesh
pv_mesh = create_pyvista_mesh(
    model.mesh.coords,
    model.mesh.elements,
    model.mesh.element_type,
)

# Add any data
pv_mesh.cell_data["my_field"] = custom_data

# Use PyVista operations
pv_mesh.clip(normal="x", origin=(50, 0, 0))
pv_mesh.plot()
```

## Troubleshooting

### PyVista Not Displaying

For headless environments (servers, Docker, WSL):

```python
import pyvista as pv
pv.start_xvfb()  # Start virtual framebuffer

# Or use off-screen rendering
results.plot("von_mises", model=model, off_screen=True, screenshot="plot.png")
```

### VTK Errors on macOS

Some macOS versions need:

```python
import pyvista as pv
pv.global_theme.render_lines_as_tubes = True
pv.global_theme.smooth_shading = True
```

### Slow Rendering for Large Models

For models with millions of elements:

```python
# Reduce mesh quality for faster preview
results.plot(
    "von_mises",
    model=model,
    show_edges=False,  # Turn off edges for speed
)

# Export to ParaView for large models
export_vtu(results, "large_model.vtu", model=model)
```

## Practical Examples

### Stress Concentration Analysis

```python
# Plot with custom color limits to highlight stress concentration
max_vm = results.max_von_mises()
yield_stress = 250  # MPa

results.plot(
    "von_mises",
    model=model,
    cmap="jet",
    clim=(0, yield_stress),  # Cap at yield
    title=f"Von Mises Stress (max: {max_vm:.0f} MPa)",
)
```

### Comparing Before/After

```python
# Side-by-side comparison
import pyvista as pv

plotter = pv.Plotter(shape=(1, 2))

# Before (original)
plotter.subplot(0, 0)
plotter.add_title("Original Design")
# Add first results

# After (optimized)
plotter.subplot(0, 1)
plotter.add_title("Optimized Design")
# Add second results

plotter.show()
```

### Animation of Deformation

```python
import pyvista as pv
import numpy as np

# Create animation frames
plotter = pv.Plotter(off_screen=True)
plotter.open_gif("deformation.gif")

for scale in np.linspace(0, 100, 20):
    nodes, elements, etype = model.mesh.coords, model.mesh.elements, model.mesh.element_type
    disp = results.displacement()
    deformed_nodes = nodes + disp * scale

    mesh = create_pyvista_mesh(deformed_nodes, elements, etype)
    mesh.cell_data["VM"] = results.von_mises()

    plotter.clear()
    plotter.add_mesh(mesh, scalars="VM", cmap="jet")
    plotter.write_frame()

plotter.close()
```

## Next Steps

- [Results](results.md): Understanding result data structures
- [Getting Started](getting-started.md): Complete workflow example
- [Elements](elements.md): Element-specific stress output
