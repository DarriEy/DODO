# DODO: Differentiable Optimization of Distributed Outflow

<p align="center">
  <img src="src/logo/Dodo_Logo.jpg" alt="DODO logo" width="300" />
</p>

**DODO** is an end-to-end differentiable framework for calibrating distributed hydrological models with river routing. It couples [dFUSE](https://github.com/DarriEy/dFUSE) (differentiable FUSE) with [dRoute](https://github.com/DarriEy/dRoute) (differentiable Muskingum-Cunge routing) to enable gradient-based optimization of both rainfall-runoff parameters and channel routing parameters simultaneously.

## Key Features

- **End-to-end differentiability**: Gradients flow from outlet streamflow through routing and back to spatially-distributed FUSE parameters
- **Spatially varying parameters**: Each HRU can have its own calibrated parameters (870+ parameters for 29 HRUs)
- **Enzyme AD integration**: True reverse-mode automatic differentiation for both FUSE and routing components
- **Multiple loss functions**: NSE, log-NSE, KGE, peak-weighted, and combined objectives for balanced calibration
- **PyTorch integration**: Seamless integration with PyTorch optimizers, schedulers, and ecosystem


## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         DODO Framework                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Forcing   │───▶│    dFUSE    │───▶│   Runoff [mm/day]   │  │
│  │ [P, PET, T] │    │  (per HRU)  │    │   per HRU           │  │
│  └─────────────┘    └─────────────┘    └──────────┬──────────┘  │
│                            ▲                      │             │
│                            │                      ▼             │
│                    ┌───────┴───────┐    ┌─────────────────────┐ │
│                    │ FUSE Params   │    │  HRU→Reach Mapping  │ │
│                    │ [n_hru × 29]  │    │  (area conversion)  │ │
│                    └───────────────┘    └──────────┬──────────┘ │
│                                                    │            │
│                                                    ▼            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │  Observed Q │◀───│   dRoute    │◀───│ Lateral Inflows     │  │
│  │  [m³/s]     │    │ (Musk-Cunge)│    │ [m³/s] per reach    │  │
│  └──────┬──────┘    └─────────────┘    └─────────────────────┘  │
│         │                  ▲                                    │
│         │                  │                                    │
│         │          ┌───────┴───────┐                            │
│         │          │  Manning's n  │                            │
│         │          │  [n_reaches]  │                            │
│         │          └───────────────┘                            │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Loss Function (NSE, KGE, etc.)             │    │
│  └─────────────────────────────────────────────────────────┘    │
│         │                                                       │
│         │  Backpropagation (Enzyme AD)                          │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           Parameter Updates (Adam optimizer)            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

1. **dFUSE** - Differentiable FUSE implementation
   ```bash
   git clone https://github.com/DarriEy/dFUSE.git
   cd dFUSE && mkdir build && cd build
   cmake .. -DDFUSE_USE_ENZYME=ON
   make -j4
   pip install -e ../python
   ```

2. **dRoute** - Differentiable river routing
   ```bash
   git clone https://github.com/DarriEy/dRoute.git
   cd dRoute && mkdir build && cd build
   cmake .. -DDMC_USE_ENZYME=ON
   make -j4
   ```

3. **Python dependencies**
   ```bash
   pip install torch numpy xarray pandas geopandas matplotlib
   ```

### DODO Setup

```bash
git clone https://github.com/DarriEy/DODO.git
cd DODO

# Update paths in run_coupled_optimization.py:
# - CODE_DIR: Path to dFUSE and dRoute
# - DATA_PATH: Path to your domain data
```

## Quick Start

To run the example:

```
python run_coupled_optimization.py
```

To use in workflows:

```python
from run_coupled_optimization import CoupledFUSERoute, train_model, load_data
import dfuse

# Load data
forcing, observed, topo_file, hru_areas = load_data(DATA_PATH)

# Initialize coupled model
model = CoupledFUSERoute(
    fuse_config=dfuse.VIC_CONFIG,
    topology_file=topo_file,
    hru_areas=hru_areas,
    dt=86400.0,              # Daily timestep
    warmup_steps=365,        # 1-year spinup
    spatial_params=True      # Per-HRU parameters
)

# Train
results = train_model(
    model=model,
    forcing=forcing,
    observed=observed,
    n_epochs=500,
    lr=0.01,
    loss_fn='triple',        # NSE + log-NSE + peak-weighted
    spatial_reg=0.0005
)
```

## Loss Functions

DODO provides multiple loss functions for different calibration objectives:

| Loss Function | Description | Best For |
|---------------|-------------|----------|
| `nse` | Nash-Sutcliffe Efficiency | Overall fit, peak-dominated |
| `log_nse` | NSE on log-transformed flows | Low flow periods |
| `kge` | Kling-Gupta Efficiency | Balanced correlation/variability/bias |
| `combined` | α×NSE + (1-α)×log-NSE | Balanced peaks and baseflow |
| `triple` | NSE + log-NSE + peak-weighted | Best overall performance |
| `asymmetric` | NSE with 2x penalty for underestimation | Recovering missed peaks |
| `peak_weighted` | NSE with 3x weight on high flows | Peak capture |

## Data Requirements

### Directory Structure
```
domain_name/
├── forcing/
│   └── FUSE_input/
│       └── domain_input.nc        # P, PET, T, q_obs
├── settings/
│   └── Route/
│       └── topology.nc            # Network topology
└── shapefiles/
    ├── catchment/
    │   └── HRUs_GRUs.shp          # HRU polygons
    └── river_network/
        └── riverNetwork.shp       # River segments
```

### Input Files

**Forcing (NetCDF)**:
- `pr`: Precipitation [mm/day]
- `pet`: Potential evapotranspiration [mm/day]  
- `temp`: Air temperature [°C]
- `q_obs`: Observed streamflow [mm/day over watershed]

**Topology (NetCDF)**:
- `segId`: Segment IDs
- `downSegId`: Downstream segment IDs
- `length`: Reach lengths [m]
- `slope`: Channel slopes [m/m]
- `hruToSegId`: HRU to segment mapping
- `area`: HRU areas [m²]


## Related Repositories

- **[dFUSE](https://github.com/DarriEy/dFUSE)**: Differentiable implementation of FUSE (Clark et al., 2008)
- **[dRoute](https://github.com/DarriEy/dRoute)**: Differentiable Muskingum-Cunge river routing
- **[CONFLUENCE/SYMFLUENCE](https://github.com/DarriEy/SYMFLUENCE)**: Comprehensive hydrological modeling framework
