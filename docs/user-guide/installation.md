# Installation

This guide covers installing MOPS from source. MOPS consists of two components:
- **mops-core**: Rust library containing the solver engine
- **mops-python**: Python bindings and high-level API

## Prerequisites

### Required

- **Rust 1.75+**: Install from [rustup.rs](https://rustup.rs/)
- **Python 3.10+**: With pip available
- **C compiler**: GCC or Clang (for building native extensions)

### Optional

- **PyVista**: For 3D visualization (`pip install pyvista`)
- **Gmsh**: For mesh generation (`pip install gmsh`)
- **h5py**: For HDF5 result persistence (`pip install h5py`)

## Build from Source

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/mops.git
cd mops
```

### 2. Build the Rust Core

```bash
cd mops-core
cargo build --release
```

This compiles the solver library with optimizations. The first build downloads dependencies and may take a few minutes.

To verify the build:

```bash
cargo test
```

### 3. Install Python Package

```bash
cd ../mops-python
pip install -e .
```

The `-e` flag installs in development mode, allowing you to modify Python code without reinstalling.

This step:
1. Compiles the PyO3 bindings (Rust → Python bridge)
2. Installs the `mops` Python package

### 4. Verify Installation

```python
import mops
print(f"MOPS version: {mops.version()}")
print(mops.solver_info())
```

Expected output:
```
MOPS version: 0.1.0
MOPS Solver Information:
  - Direct solver: faer (sparse Cholesky)
  - Iterative solver: available (PCG + AMG)
```

## Optional Dependencies

### Visualization (PyVista)

For interactive visualization in Jupyter notebooks:

```bash
pip install pyvista jupyterlab
```

PyVista enables:
- Interactive 3D stress plots
- Displacement visualization
- Export to VTU format (ParaView compatible)

### Mesh Generation (Gmsh)

For creating meshes from CAD geometry:

```bash
pip install gmsh
```

### Result Persistence (HDF5)

For saving/loading results to HDF5 format:

```bash
pip install h5py
```

## Platform-Specific Notes

### Linux

Ensure you have a C compiler installed:

```bash
# Ubuntu/Debian
sudo apt install build-essential

# Fedora
sudo dnf install gcc
```

### macOS

Install Xcode command line tools:

```bash
xcode-select --install
```

### Windows

Install Visual Studio Build Tools or use WSL2 with Linux instructions.

## Troubleshooting

### Rust not found

Ensure Rust is in your PATH:

```bash
source ~/.cargo/env
```

### Python extension build fails

Check that:
1. Rust version is 1.75+: `rustc --version`
2. Python development headers are installed: `python3-dev` or `python-devel`

### PyVista not displaying

For headless environments or WSL, set the backend:

```python
import pyvista
pyvista.start_xvfb()  # For headless
# Or
pyvista.set_jupyter_backend('static')  # For Jupyter
```

## Next Steps

- [Getting Started](getting-started.md): Run your first analysis
- [Query DSL](queries.md): Learn to select mesh entities
- [Element Library](elements.md): Understand available element types
