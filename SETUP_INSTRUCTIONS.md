# OmnEEG Project Setup

## Quick Setup

```bash
# Install system dependencies
brew install gcc openblas fftw pkg-config

# Set environment variables
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:/opt/homebrew/opt/openblas/lib/pkgconfig:/opt/homebrew/opt/fftw/lib/pkgconfig"
export LDFLAGS="-L/opt/homebrew/opt/openblas/lib -L/opt/homebrew/opt/fftw/lib"
export CPPFLAGS="-I/opt/homebrew/opt/openblas/include -I/opt/homebrew/opt/fftw/include"

# Install dependencies
uv sync
```

## pyshtools Installation

**Critical**: pyshtools requires Fortran compiler and numerical libraries on macOS:

- `gcc` (Fortran compiler)
- `openblas` (BLAS library) 
- `fftw` (FFT library)
- Environment variables for build system


## Project Usage

```bash
# Run demo
uv run python demo.py

# Interactive session
uv run python
```

## Dependencies
- Core: torch, matplotlib, numpy, mne, h5py, pyyaml, tqdm
- Data: pandas>=2.3.1  
- Specialized: pyshtools (spherical harmonic analysis)

## Troubleshooting

**"Unknown compiler(s): gfortran"** → `brew install gcc`
**"Dependency BLAS not found"** → Install openblas + set LDFLAGS/CPPFLAGS
**"Dependency fftw3 not found"** → `brew install fftw` 