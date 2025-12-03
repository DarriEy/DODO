"""
Coupled dFUSE + dRoute Differentiable Calibration (v2)

This version properly couples:
- dFUSE: Using new run_fuse_batch_gradient with Enzyme AD
- dRoute: Using native CoDiPack/Enzyme routers

Key features:
1. End-to-end differentiable through both models
2. Proper gradient accumulation for shared parameters
3. NSE/KGE loss functions
4. Warmup period handling
5. Learning rate scheduling and early stopping

Author: Darri Eythorsson
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import xarray as xr
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import warnings

# --- Path Setup ---
CODE_DIR = Path("/Users/darrieythorsson/compHydro/code")
sys.path.append(str(CODE_DIR / "dFUSE/python"))
sys.path.append(str(CODE_DIR / "dRoute/build/python"))

import dfuse
import dfuse_core
import pydmc_route as dmc

# Import the new PyTorch wrapper
from dfuse_torch import DifferentiableFUSEBatch, FUSEModule


# =============================================================================
# 1. LOSS FUNCTIONS
# =============================================================================

def nse_loss(sim: torch.Tensor, obs: torch.Tensor, 
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Nash-Sutcliffe Efficiency loss (1 - NSE)."""
    if mask is None:
        mask = ~torch.isnan(obs)
    
    sim_v, obs_v = sim[mask], obs[mask]
    if len(obs_v) == 0:
        return torch.tensor(float('inf'))
    
    ss_res = torch.sum((sim_v - obs_v) ** 2)
    ss_tot = torch.sum((obs_v - obs_v.mean()) ** 2)
    return ss_res / (ss_tot + 1e-10)


def kge_loss(sim: torch.Tensor, obs: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Kling-Gupta Efficiency loss (1 - KGE)."""
    if mask is None:
        mask = ~torch.isnan(obs)
    
    sim_v, obs_v = sim[mask], obs[mask]
    if len(obs_v) < 2:
        return torch.tensor(float('inf'))
    
    r = torch.corrcoef(torch.stack([sim_v, obs_v]))[0, 1]
    alpha = sim_v.std() / (obs_v.std() + 1e-10)
    beta = sim_v.mean() / (obs_v.mean() + 1e-10)
    
    return torch.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


# =============================================================================
# 2. DIFFERENTIABLE ROUTING LAYER - Using dRoute's Native AD
# =============================================================================

class DifferentiableRouting(torch.autograd.Function):
    """
    PyTorch wrapper for dRoute using fast Enzyme kernels.
    
    Uses dmc.enzyme.EnzymeRouter for fast forward passes and numerical
    gradients computed efficiently via the Enzyme kernels.
    """
    
    @staticmethod
    def forward(ctx, 
                lateral_inflows: torch.Tensor,  # [n_timesteps, n_reaches]
                manning_n: torch.Tensor,         # [n_reaches]
                enzyme_router,                   # dmc.enzyme.EnzymeRouter
                outlet_reach_id: int) -> torch.Tensor:
        """
        Route lateral inflows through river network using fast Enzyme kernels.
        
        Returns outlet discharge [n_timesteps].
        """
        ctx.enzyme_router = enzyme_router
        ctx.outlet_reach_id = outlet_reach_id
        ctx.n_reaches = lateral_inflows.shape[1]
        ctx.n_timesteps = lateral_inflows.shape[0]
        
        inflows_np = lateral_inflows.detach().cpu().numpy().astype(np.float64)
        manning_np = manning_n.detach().cpu().numpy().astype(np.float64)
        
        # Update Manning's n
        enzyme_router.set_manning_n_all(manning_np)
        
        # Fast forward pass using Enzyme simulate
        outlet_Q = dmc.enzyme.simulate(enzyme_router, inflows_np, outlet_reach_id)
        outlet_Q = np.array(outlet_Q)
        
        ctx.save_for_backward(lateral_inflows, manning_n)
        ctx.inflows_np = inflows_np
        ctx.outlet_Q = outlet_Q
        
        return torch.tensor(outlet_Q, dtype=torch.float32, device=lateral_inflows.device)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass using numerical gradients with fast Enzyme kernels.
        """
        lateral_inflows, manning_n = ctx.saved_tensors
        enzyme_router = ctx.enzyme_router
        inflows_np = ctx.inflows_np
        outlet_Q_base = ctx.outlet_Q
        
        grad_np = grad_output.detach().cpu().numpy().astype(np.float64)
        manning_np = manning_n.detach().cpu().numpy().astype(np.float64)
        
        # ====== Manning's n gradients (numerical with fast Enzyme forward) ======
        eps = 0.01
        grad_manning_np = np.zeros(ctx.n_reaches)
        
        for i in range(ctx.n_reaches):
            # Forward perturbation
            mann_pert = manning_np.copy()
            mann_pert[i] = manning_np[i] * (1 + eps)
            enzyme_router.set_manning_n_all(mann_pert)
            Q_plus = np.array(dmc.enzyme.simulate(enzyme_router, inflows_np, ctx.outlet_reach_id))
            
            # Backward perturbation
            mann_pert[i] = manning_np[i] * (1 - eps)
            enzyme_router.set_manning_n_all(mann_pert)
            Q_minus = np.array(dmc.enzyme.simulate(enzyme_router, inflows_np, ctx.outlet_reach_id))
            
            # Gradient via chain rule: dL/dn = sum_t(dL/dQ[t] * dQ[t]/dn)
            dQ_dn = (Q_plus - Q_minus) / (2 * eps * manning_np[i])
            grad_manning_np[i] = np.sum(grad_np * dQ_dn)
        
        # Restore original Manning's n
        enzyme_router.set_manning_n_all(manning_np)
        
        grad_manning = torch.from_numpy(grad_manning_np.astype(np.float32))
        
        # ====== Lateral inflow gradients ======
        grad_lateral = grad_np[:, np.newaxis] * np.ones((1, ctx.n_reaches))
        grad_lateral_t = torch.from_numpy(grad_lateral.astype(np.float32))
        
        return grad_lateral_t, grad_manning, None, None


# =============================================================================
# 3. COUPLED MODEL
# =============================================================================

class CoupledFUSERoute(nn.Module):
    """
    Coupled dFUSE + dRoute model with end-to-end differentiability.
    
    Uses dRoute's fast Enzyme kernels for routing.
    """
    
    def __init__(self,
                 fuse_config,
                 topology_file: str,
                 hru_areas: np.ndarray,
                 dt: float = 86400.0,
                 warmup_steps: int = 30):
        super().__init__()
        
        self.fuse_config = fuse_config
        self.config_dict = fuse_config.to_dict()
        self.dt_seconds = dt
        self.dt_days = dt / 86400.0
        self.warmup_steps = warmup_steps
        
        # Parameter bounds
        self.param_names = list(dfuse.PARAM_NAMES)
        n_params = len(self.param_names)
        
        lowers = torch.tensor([dfuse.PARAM_BOUNDS[n][0] for n in self.param_names])
        uppers = torch.tensor([dfuse.PARAM_BOUNDS[n][1] for n in self.param_names])
        self.register_buffer('param_lower', lowers)
        self.register_buffer('param_upper', uppers)
        
        # FUSE parameters (unconstrained)
        self.fuse_raw_params = nn.Parameter(torch.zeros(n_params))
        
        # Load network
        self.network = self._load_network(topology_file)
        self.n_reaches = self.network.num_reaches()
        
        # Find outlet (reach with downstream_junction_id == -1)
        self.outlet_reach_id = 0
        for i in range(self.n_reaches):
            reach = self.network.get_reach(i)
            if reach.downstream_junction_id < 0:
                self.outlet_reach_id = i
                break
        
        # Create EnzymeRouter (fast Enzyme kernels)
        self.enzyme_router = dmc.enzyme.EnzymeRouter(
            self.network,
            dt=dt,
            num_substeps=4,
            method=0  # 0 = Muskingum-Cunge
        )
        
        # Manning's n (log-space)
        self.log_manning_n = nn.Parameter(torch.full((self.n_reaches,), np.log(0.035)))
        
        # HRU to reach mapping
        self.register_buffer(
            'mapping_matrix',
            self._build_mapping_matrix(topology_file, hru_areas)
        )
        self.n_hrus = self.mapping_matrix.shape[0]
        
        # Number of states
        self.n_states = dfuse_core.get_num_active_states(self.config_dict)
    
    def _load_network(self, topology_file: str) -> 'dmc.Network':
        """Load network from topology file."""
        ds = xr.open_dataset(topology_file)
        
        seg_ids = ds['segId'].values
        down_seg_ids = ds['downSegId'].values
        slopes = ds['slope'].values
        lengths = ds['length'].values
        mann_n = ds['mann_n'].values
        
        n_segs = len(seg_ids)
        seg_id_to_idx = {int(seg_id): i for i, seg_id in enumerate(seg_ids)}
        
        # Build upstream connectivity
        upstream_map = {i: [] for i in range(n_segs)}
        for i, down_id in enumerate(down_seg_ids):
            down_id_int = int(down_id)
            if down_id_int in seg_id_to_idx:
                down_idx = seg_id_to_idx[down_id_int]
                upstream_map[down_idx].append(i)
        
        network = dmc.Network()
        
        # Create reaches
        for i in range(n_segs):
            reach = dmc.Reach()
            reach.id = i
            reach.length = float(lengths[i])
            reach.slope = max(float(slopes[i]), 0.0001)
            reach.manning_n = float(mann_n[i])
            
            reach.geometry.width_coef = 7.2
            reach.geometry.width_exp = 0.5
            reach.geometry.depth_coef = 0.27
            reach.geometry.depth_exp = 0.3
            
            reach.upstream_junction_id = i
            
            down_id = int(down_seg_ids[i])
            if down_id in seg_id_to_idx:
                reach.downstream_junction_id = seg_id_to_idx[down_id]
            else:
                reach.downstream_junction_id = -1
            
            network.add_reach(reach)
        
        # Create junctions
        for i in range(n_segs):
            junc = dmc.Junction()
            junc.id = i
            junc.upstream_reach_ids = upstream_map[i]
            junc.downstream_reach_ids = [i]
            network.add_junction(junc)
        
        network.build_topology()
        ds.close()
        
        return network
    
    def _build_mapping_matrix(self, topology_file: str, 
                              hru_areas: np.ndarray) -> torch.Tensor:
        """Build HRU to reach mapping matrix."""
        ds = xr.open_dataset(topology_file)
        
        hru_ids = ds['hruId'].values
        hru_to_seg = ds['hruToSegId'].values
        seg_ids = ds['segId'].values
        
        seg_id_to_idx = {int(seg_id): i for i, seg_id in enumerate(seg_ids)}
        
        n_hrus = len(hru_ids)
        n_reaches = len(seg_ids)
        
        mapping = torch.zeros(n_hrus, n_reaches)
        
        for h_idx, (hru_id, seg_id) in enumerate(zip(hru_ids, hru_to_seg)):
            seg_id_int = int(seg_id)
            if seg_id_int in seg_id_to_idx:
                reach_idx = seg_id_to_idx[seg_id_int]
                area_m2 = hru_areas[h_idx]
                area_km2 = area_m2 / 1e6
                mm_to_m3s = area_km2 * 1000 / self.dt_seconds
                mapping[h_idx, reach_idx] = mm_to_m3s
        
        ds.close()
        return mapping
    
    def get_physical_params(self) -> torch.Tensor:
        """Transform raw params to physical space."""
        return self.param_lower + (self.param_upper - self.param_lower) * torch.sigmoid(self.fuse_raw_params)
    
    def get_initial_state(self) -> torch.Tensor:
        """Get default initial states."""
        state = torch.zeros(self.n_hrus, self.n_states)
        state[:, 0] = 50.0
        if self.n_states > 1:
            state[:, 1] = 20.0
        if self.n_states > 2:
            state[:, 2] = 200.0
        return state
    
    def forward(self, forcing: torch.Tensor, 
                initial_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run coupled model.
        
        Args:
            forcing: [n_timesteps, n_hrus, 3] Forcing data
            initial_state: [n_hrus, n_states] Initial states
            
        Returns:
            outlet_Q: [n_timesteps] Outlet discharge in m³/s
            runoff: [n_timesteps, n_hrus] Runoff in mm
        """
        if initial_state is None:
            initial_state = self.get_initial_state().to(forcing.device)
        
        # Get physical parameters
        phys_params = self.get_physical_params()
        
        # Run FUSE with gradient flow
        runoff = DifferentiableFUSEBatch.apply(
            phys_params, initial_state, forcing,
            self.config_dict, self.dt_days
        )
        
        # Map to reaches
        lateral_inflows = torch.matmul(runoff, self.mapping_matrix.T)
        
        # Route using EnzymeRouter
        manning_n = torch.exp(self.log_manning_n)
        outlet_Q = DifferentiableRouting.apply(
            lateral_inflows, manning_n, self.enzyme_router, self.outlet_reach_id
        )
        
        return outlet_Q, runoff
    
    def get_param_dict(self) -> Dict[str, float]:
        """Get current parameters."""
        phys = self.get_physical_params()
        return {name: phys[i].item() for i, name in enumerate(self.param_names)}


# =============================================================================
# 4. TRAINING
# =============================================================================

def train_model(model: CoupledFUSERoute,
                forcing: torch.Tensor,
                observed: torch.Tensor,
                n_epochs: int = 100,
                lr: float = 0.01,
                loss_fn: str = 'nse',
                verbose: bool = True) -> Dict:
    """Train the coupled model."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    initial_state = model.get_initial_state()
    warmup = model.warmup_steps
    
    history = {'loss': [], 'nse': [], 'kge': []}
    best_loss = float('inf')
    best_state = None
    
    loss_func = nse_loss if loss_fn == 'nse' else kge_loss
    
    if verbose:
        print(f"\nTraining for {n_epochs} epochs with {loss_fn.upper()} loss")
        print("-" * 60)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        sim_Q, _ = model(forcing, initial_state)
        
        # Apply warmup
        sim_valid = sim_Q[warmup:]
        obs_valid = observed[warmup:]
        mask = ~torch.isnan(obs_valid) & (obs_valid >= 0)
        
        if mask.sum() == 0:
            print("Warning: No valid observations")
            break
        
        loss = loss_func(sim_valid, obs_valid, mask)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss.detach())
        
        # Record
        with torch.no_grad():
            nse_val = 1 - nse_loss(sim_valid, obs_valid, mask).item()
            kge_val = 1 - kge_loss(sim_valid, obs_valid, mask).item()
        
        history['loss'].append(loss.item())
        history['nse'].append(nse_val)
        history['kge'].append(kge_val)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if verbose and (epoch + 1) % 5 == 0:
            mean_n = torch.exp(model.log_manning_n).mean().item()
            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | "
                  f"NSE: {nse_val:.3f} | KGE: {kge_val:.3f} | n̄: {mean_n:.4f}")
    
    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    
    if verbose:
        print("-" * 60)
        print(f"Training complete. Best loss: {best_loss:.4f}")
    
    return {
        'history': history,
        'best_loss': best_loss,
        'final_params': model.get_param_dict(),
        'final_manning_n': torch.exp(model.log_manning_n).detach().numpy()
    }


# =============================================================================
# 5. DATA LOADING
# =============================================================================

def load_data(data_path: Path) -> Tuple[torch.Tensor, torch.Tensor, str, np.ndarray]:
    """Load Bow at Banff data."""
    print("Loading data...")
    
    topo_file = data_path / "settings/mizuRoute/topology.nc"
    topo_ds = xr.open_dataset(topo_file)
    hru_areas = topo_ds['area'].values.astype(np.float64)
    topo_ds.close()
    
    forcing_dir = data_path / "forcing/SUMMA_input"
    forcing_files = sorted(list(forcing_dir.glob("*200205.nc")))
    
    if not forcing_files:
        raise FileNotFoundError(f"No forcing in {forcing_dir}")
    
    ds = xr.open_dataset(forcing_files[0])
    precip = ds['pptrate'].values * 86400.0
    temp = ds['airtemp'].values - 273.15
    pet = ds['SWRadAtm'].values * 0.0864 * 0.408 if 'SWRadAtm' in ds else np.zeros_like(precip)
    ds.close()
    
    forcing = torch.tensor(np.stack([precip, pet, temp], axis=2), dtype=torch.float32)
    print(f"  Forcing: {forcing.shape}")
    
    obs_file = data_path / "observations/streamflow/preprocessed/Bow_at_Banff_distributed_streamflow_processed.csv"
    obs_df = pd.read_csv(obs_file, parse_dates=['datetime'])
    obs = torch.tensor(obs_df['discharge_cms'].values[:len(forcing)], dtype=torch.float32)
    obs[obs < 0] = float('nan')
    print(f"  Obs: {(~torch.isnan(obs)).sum().item()} valid")
    
    return forcing, obs, str(topo_file), hru_areas


# =============================================================================
# 6. MAIN
# =============================================================================

def main():
    DATA_PATH = Path("/Users/darrieythorsson/compHydro/data/CONFLUENCE_data/domain_Bow_at_Banff_distributed")
    
    forcing, observed, topo_file, hru_areas = load_data(DATA_PATH)
    
    print("\nInitializing model...")
    model = CoupledFUSERoute(
        fuse_config=dfuse.VIC_CONFIG,
        topology_file=topo_file,
        hru_areas=hru_areas,
        dt=86400.0,
        warmup_steps=30
    )
    print(f"  FUSE params: {len(model.param_names)}")
    print(f"  Reaches: {model.n_reaches}")
    print(f"  HRUs: {model.n_hrus}")
    
    # Check gradient computation
    print("\nVerifying gradient flow...")
    test_forcing = forcing[:10]
    test_state = model.get_initial_state()
    model.zero_grad()
    Q, _ = model(test_forcing, test_state)
    loss = Q.sum()
    loss.backward()
    
    fuse_grad = model.fuse_raw_params.grad
    manning_grad = model.log_manning_n.grad
    
    if fuse_grad is not None and (fuse_grad != 0).any():
        print("  ✓ FUSE gradients flowing")
    else:
        print("  ✗ FUSE gradients NOT flowing!")
    
    if manning_grad is not None and (manning_grad != 0).any():
        print("  ✓ Manning gradients flowing")
    else:
        print("  ✗ Manning gradients NOT flowing!")
    
    # Train
    results = train_model(
        model=model,
        forcing=forcing,
        observed=observed,
        n_epochs=100,
        lr=0.02,
        loss_fn='nse',
        verbose=True
    )
    
    # Final results
    print("\n" + "=" * 60)
    print("CALIBRATED PARAMETERS")
    print("=" * 60)
    
    print("\nFUSE Parameters:")
    for name, val in results['final_params'].items():
        print(f"  {name}: {val:.4f}")
    
    print(f"\nManning's n: mean={results['final_manning_n'].mean():.4f}, "
          f"range=[{results['final_manning_n'].min():.4f}, {results['final_manning_n'].max():.4f}]")
    
    print(f"\nFinal NSE: {results['history']['nse'][-1]:.3f}")
    print(f"Final KGE: {results['history']['kge'][-1]:.3f}")


if __name__ == "__main__":
    main()