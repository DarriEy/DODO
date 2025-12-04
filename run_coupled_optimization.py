"""
Coupled dFUSE + dRoute Differentiable Calibration (v3 - Enzyme AD)

This version properly couples:
- dFUSE: Using Enzyme AD via DifferentiableFUSEBatch
- dRoute: Using TRUE Enzyme AD via dmc.enzyme.compute_manning_gradients

Key features:
1. End-to-end differentiable through both models
2. True reverse-mode AD for Manning's n (not numerical/CoDiPack)
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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import geopandas as gpd

# --- Path Setup ---
CODE_DIR = Path("/Users/darrieythorsson/compHydro/code")
sys.path.append(str(CODE_DIR / "dFUSE/python"))
sys.path.append(str(CODE_DIR / "dRoute/build"))

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


def log_nse_loss(sim: torch.Tensor, obs: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 epsilon: float = 0.1) -> torch.Tensor:
    """
    NSE on log-transformed flows - better for low flow periods.
    
    Args:
        sim, obs: Simulated and observed flows
        mask: Valid data mask
        epsilon: Small value added before log to handle zeros
    """
    if mask is None:
        mask = ~torch.isnan(obs) & (obs > 0) & (sim > 0)
    
    sim_v, obs_v = sim[mask], obs[mask]
    if len(obs_v) == 0:
        return torch.tensor(float('inf'))
    
    # Log transform with epsilon for numerical stability
    log_sim = torch.log(sim_v + epsilon)
    log_obs = torch.log(obs_v + epsilon)
    
    ss_res = torch.sum((log_sim - log_obs) ** 2)
    ss_tot = torch.sum((log_obs - log_obs.mean()) ** 2)
    return ss_res / (ss_tot + 1e-10)


def combined_nse_loss(sim: torch.Tensor, obs: torch.Tensor,
                      mask: Optional[torch.Tensor] = None,
                      alpha: float = 0.5) -> torch.Tensor:
    """
    Combined NSE + log-NSE loss for balanced peak/baseflow calibration.
    
    Args:
        alpha: Weight for regular NSE (1-alpha for log-NSE)
    """
    nse = nse_loss(sim, obs, mask)
    log_nse = log_nse_loss(sim, obs, mask)
    return alpha * nse + (1 - alpha) * log_nse


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


def flow_duration_loss(sim: torch.Tensor, obs: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Flow duration curve loss - matches distribution of flows.
    Helpful for getting low flows right.
    """
    if mask is None:
        mask = ~torch.isnan(obs)
    
    sim_v, obs_v = sim[mask], obs[mask]
    if len(obs_v) == 0:
        return torch.tensor(float('inf'))
    
    # Sort both
    sim_sorted = torch.sort(sim_v, descending=True)[0]
    obs_sorted = torch.sort(obs_v, descending=True)[0]
    
    # Relative error across the FDC
    rel_err = (sim_sorted - obs_sorted) / (obs_sorted + 1e-6)
    return torch.mean(rel_err ** 2)


def asymmetric_nse_loss(sim: torch.Tensor, obs: torch.Tensor,
                        mask: Optional[torch.Tensor] = None,
                        under_weight: float = 2.0) -> torch.Tensor:
    """
    NSE with asymmetric weighting - penalizes underestimation more.
    
    This helps recover peaks that are being missed.
    
    Args:
        under_weight: Multiplier for underestimation errors (>1 = penalize more)
    """
    if mask is None:
        mask = ~torch.isnan(obs)
    
    sim_v, obs_v = sim[mask], obs[mask]
    if len(obs_v) == 0:
        return torch.tensor(float('inf'))
    
    errors = sim_v - obs_v
    # Weight underestimation (negative errors) more heavily
    weights = torch.where(errors < 0, under_weight, 1.0)
    weighted_ss_res = torch.sum(weights * errors ** 2)
    ss_tot = torch.sum((obs_v - obs_v.mean()) ** 2)
    
    return weighted_ss_res / (ss_tot + 1e-10)


def peak_weighted_nse(sim: torch.Tensor, obs: torch.Tensor,
                      mask: Optional[torch.Tensor] = None,
                      quantile: float = 0.9) -> torch.Tensor:
    """
    NSE with extra weight on high-flow periods.
    
    Args:
        quantile: Flows above this quantile get extra weight
    """
    if mask is None:
        mask = ~torch.isnan(obs)
    
    sim_v, obs_v = sim[mask], obs[mask]
    if len(obs_v) == 0:
        return torch.tensor(float('inf'))
    
    # Find threshold for high flows
    threshold = torch.quantile(obs_v, quantile)
    
    # Weight high flows more
    weights = torch.where(obs_v > threshold, 3.0, 1.0)
    
    weighted_ss_res = torch.sum(weights * (sim_v - obs_v) ** 2)
    weighted_ss_tot = torch.sum(weights * (obs_v - obs_v.mean()) ** 2)
    
    return weighted_ss_res / (weighted_ss_tot + 1e-10)


def triple_objective_loss(sim: torch.Tensor, obs: torch.Tensor,
                          mask: Optional[torch.Tensor] = None,
                          w_nse: float = 0.4,
                          w_log: float = 0.3,
                          w_peak: float = 0.3) -> torch.Tensor:
    """
    Combined loss: NSE + log-NSE + peak-weighted NSE.
    
    Balances overall fit, low flows, and peak capture.
    """
    nse = nse_loss(sim, obs, mask)
    log_nse = log_nse_loss(sim, obs, mask)
    peak_nse = peak_weighted_nse(sim, obs, mask)
    
    return w_nse * nse + w_log * log_nse + w_peak * peak_nse


# =============================================================================
# 2. DIFFERENTIABLE ROUTING LAYERS
# =============================================================================

# Routing method configurations
# Format: (router_class_name, supports_gradients, gradient_method, config_class)
# gradient_method: 'enzyme' = use Enzyme AD, 'native' = use router's built-in AD, 
#                  'sve' = Saint-Venant specific gradient API, 
#                  'sve_enzyme' = Saint-Venant with Enzyme+CVODES adjoint,
#                  None = forward-only
ROUTING_METHOD_INFO = {
    'muskingum_cunge': ('MuskingumCungeRouter', True, 'enzyme', 'RouterConfig'),
    'diffusive_wave': ('DiffusiveWaveRouter', True, 'native', 'RouterConfig'),    # Diffusive wave approx
    'diffusive_wave_ift': ('DiffusiveWaveIFT', True, 'native', 'RouterConfig'),   # Diffusive wave with IFT adjoint
    'irf': ('IRFRouter', True, 'native', 'RouterConfig'),                          # Impulse Response Function
    'kwt_soft': ('SoftGatedKWT', True, 'native', 'RouterConfig'),                  # Differentiable KWT
    'saint_venant': ('SaintVenantRouter', True, 'sve', 'SaintVenantConfig'),       # Full dynamic Saint-Venant (FD gradients)
    'saint_venant_enzyme': ('SaintVenantEnzyme', True, 'sve_enzyme', 'SaintVenantEnzymeConfig'),  # SVE + Enzyme AD
    'lag': ('LagRouter', False, None, 'RouterConfig'),                             # Forward-only
    'kwt': ('KWTRouter', False, None, 'RouterConfig'),                             # Forward-only
}


class DifferentiableRoutingNative(torch.autograd.Function):
    """
    PyTorch wrapper for dRoute routers using their native gradient computation.
    
    Works with: IRFRouter, DiffusiveWaveRouter, DiffusiveWaveIFT, SoftGatedKWT
    
    These routers have built-in gradient computation via:
    - start_recording() / stop_recording()
    - compute_gradients(gauge_reaches, dL_dQ)
    - get_gradients() -> dict
    """
    
    @staticmethod
    def forward(ctx, 
                lateral_inflows: torch.Tensor,  # [n_timesteps, n_reaches]
                manning_n: torch.Tensor,         # [n_reaches]
                router,                          # Any gradient-capable router
                network,                         # dmc.Network
                outlet_reach_id: int,            # Reach ID of outlet
                dt: float,                       # Timestep in seconds
                topo_order_ids: list) -> torch.Tensor:
        """
        Route lateral inflows through river network.
        
        Returns outlet discharge [n_timesteps].
        """
        ctx.router = router
        ctx.network = network
        ctx.outlet_reach_id = outlet_reach_id
        ctx.n_reaches = lateral_inflows.shape[1]
        ctx.n_timesteps = lateral_inflows.shape[0]
        ctx.dt = dt
        ctx.topo_order_ids = topo_order_ids
        
        inflows_np = lateral_inflows.detach().cpu().numpy().astype(np.float64)
        manning_np = manning_n.detach().cpu().numpy().astype(np.float64)
        
        # Update Manning's n on network
        network.set_manning_n_all(manning_np)
        
        # Reset router state and start recording for gradient computation
        router.reset_state()
        router.enable_gradients(True)
        router.start_recording()
        
        # Forward pass
        outlet_Q = []
        for t in range(ctx.n_timesteps):
            for i, rid in enumerate(topo_order_ids):
                router.set_lateral_inflow(int(rid), float(inflows_np[t, i]))
            router.route_timestep()
            outlet_Q.append(router.get_discharge(outlet_reach_id))
        
        router.stop_recording()
        
        outlet_Q = np.array(outlet_Q, dtype=np.float64)
        
        # Save for backward
        ctx.save_for_backward(lateral_inflows, manning_n)
        ctx.inflows_np = inflows_np
        
        return torch.tensor(outlet_Q, dtype=torch.float32, device=lateral_inflows.device)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass using router's native gradient computation.
        """
        lateral_inflows, manning_n = ctx.saved_tensors
        
        grad_np = grad_output.detach().cpu().numpy().astype(np.float64)
        
        # Compute gradients using router's native AD
        # dL_dQ is the gradient of loss w.r.t. each output timestep
        # We sum them to get the total gradient contribution
        ctx.router.compute_gradients([ctx.outlet_reach_id], [float(grad_np.sum())])
        
        # Get gradients from router (dict: 'reach_X_manning_n' -> value)
        grads_dict = ctx.router.get_gradients()
        
        # Map back to tensor in topological order
        grad_manning_np = np.zeros(ctx.n_reaches, dtype=np.float64)
        for i, rid in enumerate(ctx.topo_order_ids):
            key = f"reach_{rid}_manning_n"
            if key in grads_dict:
                grad_manning_np[i] = grads_dict[key]
        
        grad_manning = torch.from_numpy(grad_manning_np.astype(np.float32))
        
        # Lateral inflow gradients (mass conservation approximation)
        grad_lateral = grad_np[:, np.newaxis] * np.ones((1, ctx.n_reaches))
        grad_lateral_t = torch.from_numpy(grad_lateral.astype(np.float32))
        
        # Reset router gradients for next iteration
        ctx.router.reset_gradients()
        
        return grad_lateral_t, grad_manning, None, None, None, None, None


class DifferentiableRoutingSaintVenant(torch.autograd.Function):
    """
    PyTorch wrapper for the full dynamic Saint-Venant router.
    
    The Saint-Venant router solves the full 1D shallow water equations:
    - Continuity: ∂A/∂t + ∂Q/∂x = q_lat
    - Momentum: ∂Q/∂t + ∂(Q²/A)/∂x + gA∂h/∂x = gA(S₀ - Sf)
    
    Uses numerical gradients via finite difference on Manning's n.
    
    NOTE: Saint-Venant is computationally expensive and can be numerically
    challenging for daily timesteps. Consider using 'diffusive_wave' for
    faster, more stable daily routing.
    """
    
    @staticmethod
    def forward(ctx, 
                lateral_inflows: torch.Tensor,  # [n_timesteps, n_reaches]
                manning_n: torch.Tensor,         # [n_reaches]
                router,                          # dmc.SaintVenantRouter
                network,                         # dmc.Network
                outlet_reach_id: int,            # Reach ID of outlet
                dt: float,                       # Timestep in seconds
                topo_order_ids: list) -> torch.Tensor:
        """
        Route lateral inflows through river network using Saint-Venant equations.
        
        Returns outlet discharge [n_timesteps].
        """
        ctx.router = router
        ctx.network = network
        ctx.outlet_reach_id = outlet_reach_id
        ctx.n_reaches = lateral_inflows.shape[1]
        ctx.n_timesteps = lateral_inflows.shape[0]
        ctx.dt = dt
        ctx.topo_order_ids = topo_order_ids
        
        inflows_np = lateral_inflows.detach().cpu().numpy().astype(np.float64)
        manning_np = manning_n.detach().cpu().numpy().astype(np.float64)
        
        # Ensure inflows are non-negative (SVE can struggle with negative values)
        inflows_np = np.maximum(inflows_np, 0.0)
        
        # Update Manning's n on network
        network.set_manning_n_all(manning_np)
        
        # Reset router state
        router.reset_state()
        
        # Start recording for gradient computation
        router.start_recording()
        
        # Forward pass - route and record outputs
        outlet_Q = []
        last_valid_Q = 0.1  # Fallback value
        n_errors = 0
        
        for t in range(ctx.n_timesteps):
            for i, rid in enumerate(topo_order_ids):
                # Ensure minimum inflow to prevent dry bed
                inflow = max(float(inflows_np[t, i]), 1e-6)
                router.set_lateral_inflow(int(rid), inflow)
            
            router.route_timestep()
            router.record_output(outlet_reach_id)
            
            Q = router.get_discharge(outlet_reach_id)
            
            # Check for NaN/Inf and use fallback if needed
            if np.isnan(Q) or np.isinf(Q) or Q < 0:
                Q = last_valid_Q
                n_errors += 1
            else:
                last_valid_Q = max(Q, 0.1)  # Update fallback
            
            outlet_Q.append(Q)
        
        router.stop_recording()
        
        if n_errors > 0:
            print(f"  WARNING: Saint-Venant solver had {n_errors}/{ctx.n_timesteps} "
                  f"timesteps with numerical issues (used fallback values)")
        
        outlet_Q = np.array(outlet_Q, dtype=np.float64)
        
        # Save for backward
        ctx.save_for_backward(lateral_inflows, manning_n)
        ctx.inflows_np = inflows_np
        ctx.outlet_Q = outlet_Q
        
        return torch.tensor(outlet_Q, dtype=torch.float32, device=lateral_inflows.device)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass using Saint-Venant router's gradient computation.
        Uses finite difference internally for Manning's n gradients.
        """
        lateral_inflows, manning_n = ctx.saved_tensors
        
        grad_np = grad_output.detach().cpu().numpy().astype(np.float64)
        
        # Compute gradients via the router's timeseries gradient function
        # This uses finite difference on Manning's n internally
        ctx.router.compute_gradients_timeseries(ctx.outlet_reach_id, grad_np.tolist())
        
        # Get gradients from router
        grads_dict = ctx.router.get_gradients()
        
        # Map back to tensor in topological order
        grad_manning_np = np.zeros(ctx.n_reaches, dtype=np.float64)
        for i, rid in enumerate(ctx.topo_order_ids):
            key = f"reach_{rid}_manning_n"
            if key in grads_dict:
                grad_manning_np[i] = grads_dict[key]
        
        grad_manning = torch.from_numpy(grad_manning_np.astype(np.float32))
        
        # Lateral inflow gradients (mass conservation approximation)
        grad_lateral = grad_np[:, np.newaxis] * np.ones((1, ctx.n_reaches))
        grad_lateral_t = torch.from_numpy(grad_lateral.astype(np.float32))
        
        return grad_lateral_t, grad_manning, None, None, None, None, None


class DifferentiableRoutingSaintVenantEnzyme(torch.autograd.Function):
    """
    PyTorch wrapper for the Enzyme-enabled Saint-Venant router.
    
    Uses CVODES adjoint sensitivity with Enzyme AD for efficient gradients:
    - Forward: CVODES BDF with Enzyme-computed Jacobian
    - Backward: CVODES adjoint with Enzyme-computed J^T λ products
    - Gradients: Accumulated via ∫ λ^T (∂f/∂p) dt
    
    This is significantly faster than finite difference for gradient computation.
    """
    
    @staticmethod
    def forward(ctx, 
                lateral_inflows: torch.Tensor,  # [n_timesteps, n_reaches]
                manning_n: torch.Tensor,         # [n_reaches]
                router,                          # dmc.SaintVenantEnzyme
                network,                         # dmc.Network
                outlet_reach_id: int,            # Reach ID of outlet
                dt: float,                       # Timestep in seconds
                topo_order_ids: list) -> torch.Tensor:
        """
        Route lateral inflows through river network using Saint-Venant + Enzyme.
        
        Returns outlet discharge [n_timesteps].
        """
        ctx.router = router
        ctx.network = network
        ctx.outlet_reach_id = outlet_reach_id
        ctx.n_reaches = lateral_inflows.shape[1]
        ctx.n_timesteps = lateral_inflows.shape[0]
        ctx.dt = dt
        ctx.topo_order_ids = topo_order_ids
        
        inflows_np = lateral_inflows.detach().cpu().numpy().astype(np.float64)
        manning_np = manning_n.detach().cpu().numpy().astype(np.float64)
        
        # Ensure non-negative inflows
        inflows_np = np.maximum(inflows_np, 0.0)
        
        # Update Manning's n on network
        network.set_manning_n_all(manning_np)
        
        # Reset router state and start recording for adjoint
        router.reset_state()
        router.start_recording()
        
        # Forward pass with CVODES checkpointing
        outlet_Q = []
        last_valid_Q = 0.1
        n_errors = 0
        
        for t in range(ctx.n_timesteps):
            for i, rid in enumerate(topo_order_ids):
                inflow = max(float(inflows_np[t, i]), 1e-6)
                router.set_lateral_inflow(int(rid), inflow)
            
            router.route_timestep()
            
            Q = router.get_discharge(outlet_reach_id)
            
            if np.isnan(Q) or np.isinf(Q) or Q < 0:
                Q = last_valid_Q
                n_errors += 1
            else:
                last_valid_Q = max(Q, 0.1)
            
            outlet_Q.append(Q)
        
        router.stop_recording()
        
        if n_errors > 0:
            print(f"  WARNING: SVE-Enzyme solver had {n_errors}/{ctx.n_timesteps} "
                  f"timesteps with numerical issues")
        
        outlet_Q = np.array(outlet_Q, dtype=np.float64)
        
        # Save for backward
        ctx.save_for_backward(lateral_inflows, manning_n)
        ctx.inflows_np = inflows_np
        ctx.outlet_Q = outlet_Q
        
        return torch.tensor(outlet_Q, dtype=torch.float32, device=lateral_inflows.device)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass using CVODES adjoint + Enzyme AD.
        
        This computes exact gradients via:
        1. Backward integration: λ' = -J^T λ (Enzyme provides J^T λ)
        2. Accumulation: dL/dp = ∫ λ^T (∂f/∂p) dt (Enzyme provides ∂f/∂p)
        """
        lateral_inflows, manning_n = ctx.saved_tensors
        
        grad_np = grad_output.detach().cpu().numpy().astype(np.float64)
        
        # Compute gradients via CVODES adjoint + Enzyme
        ctx.router.compute_gradients(ctx.outlet_reach_id, grad_np.tolist())
        
        # Get gradients from router
        grads_dict = ctx.router.get_gradients()
        
        # Map back to tensor in topological order
        grad_manning_np = np.zeros(ctx.n_reaches, dtype=np.float64)
        for i, rid in enumerate(ctx.topo_order_ids):
            key = f"reach_{rid}_manning_n"
            if key in grads_dict:
                grad_manning_np[i] = grads_dict[key]
        
        grad_manning = torch.from_numpy(grad_manning_np.astype(np.float32))
        
        # Lateral inflow gradients (would need additional adjoint for these)
        grad_lateral = grad_np[:, np.newaxis] * np.ones((1, ctx.n_reaches))
        grad_lateral_t = torch.from_numpy(grad_lateral.astype(np.float32))
        
        # Reset for next iteration
        ctx.router.reset_gradients()
        
        return grad_lateral_t, grad_manning, None, None, None, None, None


class DifferentiableRouting(torch.autograd.Function):
    """
    PyTorch wrapper for dRoute using TRUE Enzyme AD gradients.
    
    Forward: Standard MuskingumCungeRouter
    Backward: Enzyme AD via compute_manning_gradients
    """
    
    @staticmethod
    def forward(ctx, 
                lateral_inflows: torch.Tensor,  # [n_timesteps, n_reaches]
                manning_n: torch.Tensor,         # [n_reaches]
                router,                          # dmc.MuskingumCungeRouter
                network,                         # dmc.Network (for topology extraction)
                outlet_reach_id: int,            # Reach ID of outlet
                dt: float) -> torch.Tensor:      # Timestep in seconds
        """
        Route lateral inflows through river network.
        
        Returns outlet discharge [n_timesteps].
        """
        ctx.router = router
        ctx.network = network
        ctx.outlet_reach_id = outlet_reach_id
        ctx.n_reaches = lateral_inflows.shape[1]
        ctx.n_timesteps = lateral_inflows.shape[0]
        ctx.dt = dt
        
        inflows_np = lateral_inflows.detach().cpu().numpy().astype(np.float64)
        manning_np = manning_n.detach().cpu().numpy().astype(np.float64)
        
        # Update Manning's n on network
        network.set_manning_n_all(manning_np)
        
        # Reset router state
        router.reset_state()
        
        # Forward pass using standard router
        topo_order = list(network.topological_order())
        outlet_Q = []
        
        for t in range(ctx.n_timesteps):
            # Set lateral inflows for all reaches
            for i, rid in enumerate(topo_order):
                router.set_lateral_inflow(int(rid), float(inflows_np[t, i]))
            
            router.route_timestep()
            outlet_Q.append(router.get_discharge(outlet_reach_id))
        
        outlet_Q = np.array(outlet_Q, dtype=np.float64)
        
        # Extract topology for backward pass
        n_reaches = ctx.n_reaches
        topo_order_ids = list(network.topological_order())  # Actual reach IDs
        
        # For Enzyme, we need indices 0..n-1 in topo order (which is just 0,1,2,...)
        # since we already ordered data by topo order
        topo_order = np.arange(n_reaches, dtype=np.int32)
        
        # Map reach IDs to indices in topo order
        id_to_idx = {int(rid): i for i, rid in enumerate(topo_order_ids)}
        
        # Build upstream connectivity arrays
        upstream_counts = np.zeros(n_reaches, dtype=np.int32)
        upstream_lists = [[] for _ in range(n_reaches)]
        
        for i, rid in enumerate(topo_order_ids):
            reach = network.get_reach(int(rid))
            if reach.upstream_junction_id >= 0:
                try:
                    junc = network.get_junction(reach.upstream_junction_id)
                    for up_id in junc.upstream_reach_ids:
                        if up_id in id_to_idx:
                            upstream_lists[i].append(id_to_idx[up_id])
                except:
                    pass
        
        for i in range(n_reaches):
            upstream_counts[i] = len(upstream_lists[i])
        
        # Flatten upstream indices with offsets
        upstream_offsets = np.zeros(n_reaches + 1, dtype=np.int32)
        for i in range(n_reaches):
            upstream_offsets[i + 1] = upstream_offsets[i] + upstream_counts[i]
        
        total_upstream = upstream_offsets[n_reaches]
        upstream_indices = np.zeros(max(total_upstream, 1), dtype=np.int32)
        for i in range(n_reaches):
            offset = upstream_offsets[i]
            for j, up_idx in enumerate(upstream_lists[i]):
                upstream_indices[offset + j] = up_idx
        
        # Extract reach properties (in topo order)
        lengths = np.zeros(n_reaches, dtype=np.float64)
        slopes = np.zeros(n_reaches, dtype=np.float64)
        width_coefs = np.zeros(n_reaches, dtype=np.float64)
        width_exps = np.zeros(n_reaches, dtype=np.float64)
        depth_coefs = np.zeros(n_reaches, dtype=np.float64)
        depth_exps = np.zeros(n_reaches, dtype=np.float64)
        
        for i, rid in enumerate(topo_order_ids):
            reach = network.get_reach(int(rid))
            lengths[i] = reach.length
            slopes[i] = max(reach.slope, 0.0001)
            width_coefs[i] = reach.geometry.width_coef
            width_exps[i] = reach.geometry.width_exp
            depth_coefs[i] = reach.geometry.depth_coef
            depth_exps[i] = reach.geometry.depth_exp
        
        # Find outlet index in topo order
        outlet_idx = id_to_idx[outlet_reach_id]
        
        # Save for backward
        ctx.save_for_backward(lateral_inflows, manning_n)
        ctx.inflows_np = inflows_np
        ctx.topo_order = topo_order
        ctx.topo_order_ids = topo_order_ids  # Actual reach IDs for numerical fallback
        ctx.upstream_counts = upstream_counts
        ctx.upstream_offsets = upstream_offsets
        ctx.upstream_indices = upstream_indices
        ctx.lengths = lengths
        ctx.slopes = slopes
        ctx.width_coefs = width_coefs
        ctx.width_exps = width_exps
        ctx.depth_coefs = depth_coefs
        ctx.depth_exps = depth_exps
        ctx.outlet_idx = outlet_idx
        
        return torch.tensor(outlet_Q, dtype=torch.float32, device=lateral_inflows.device)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass using TRUE Enzyme AD.
        """
        lateral_inflows, manning_n = ctx.saved_tensors
        
        grad_np = grad_output.detach().cpu().numpy().astype(np.float64)
        manning_np = manning_n.detach().cpu().numpy().astype(np.float64)
        
        # Use Enzyme AD for Manning's n gradients
        try:
            grad_manning_np = dmc.enzyme.compute_manning_gradients(
                manning_np,
                ctx.inflows_np,
                grad_np,
                ctx.lengths,
                ctx.slopes,
                ctx.width_coefs,
                ctx.width_exps,
                ctx.depth_coefs,
                ctx.depth_exps,
                ctx.topo_order,
                ctx.upstream_counts,
                ctx.upstream_offsets,
                ctx.upstream_indices,
                ctx.outlet_idx,
                ctx.dt
            )
            grad_manning_np = np.array(grad_manning_np)
        except Exception as e:
            print(f"  Enzyme AD failed ({e}), using numerical fallback")
            # Numerical gradient fallback using standard router
            eps = 0.01
            grad_manning_np = np.zeros(ctx.n_reaches)
            router = ctx.router
            network = ctx.network
            topo_order_ids = ctx.topo_order_ids
            
            for i in range(ctx.n_reaches):
                mann_pert = manning_np.copy()
                
                # Plus perturbation
                mann_pert[i] = manning_np[i] * (1 + eps)
                network.set_manning_n_all(mann_pert)
                router.reset_state()
                Q_plus = []
                for t in range(ctx.n_timesteps):
                    for j, rid in enumerate(topo_order_ids):
                        router.set_lateral_inflow(int(rid), float(ctx.inflows_np[t, j]))
                    router.route_timestep()
                    Q_plus.append(router.get_discharge(ctx.outlet_reach_id))
                
                # Minus perturbation
                mann_pert[i] = manning_np[i] * (1 - eps)
                network.set_manning_n_all(mann_pert)
                router.reset_state()
                Q_minus = []
                for t in range(ctx.n_timesteps):
                    for j, rid in enumerate(topo_order_ids):
                        router.set_lateral_inflow(int(rid), float(ctx.inflows_np[t, j]))
                    router.route_timestep()
                    Q_minus.append(router.get_discharge(ctx.outlet_reach_id))
                
                dQ_dn = (np.array(Q_plus) - np.array(Q_minus)) / (2 * eps * manning_np[i])
                grad_manning_np[i] = np.sum(grad_np * dQ_dn)
            
            # Restore original
            network.set_manning_n_all(manning_np)
        
        grad_manning = torch.from_numpy(grad_manning_np.astype(np.float32))
        
        # Lateral inflow gradients (mass conservation approximation)
        grad_lateral = grad_np[:, np.newaxis] * np.ones((1, ctx.n_reaches))
        grad_lateral_t = torch.from_numpy(grad_lateral.astype(np.float32))
        
        return grad_lateral_t, grad_manning, None, None, None, None


# =============================================================================
# 3. COUPLED MODEL WITH SPATIALLY VARYING PARAMETERS
# =============================================================================

class CoupledFUSERoute(nn.Module):
    """
    Coupled dFUSE + dRoute model with end-to-end differentiability.
    Uses TRUE Enzyme AD for both FUSE and routing gradients.
    
    SPATIALLY VARYING: Each HRU has its own FUSE parameters.
    
    Args:
        fuse_config: FUSE model configuration
        topology_file: Path to topology.nc file
        hru_areas: Array of HRU areas in m²
        dt: Timestep in seconds (default: 86400.0 = 1 day)
        warmup_steps: Number of warmup timesteps (default: 30)
        spatial_params: Use spatially varying FUSE params (default: True)
        routing_method: Routing method to use (default: 'muskingum_cunge')
        
            Differentiable methods (Manning's n is optimized):
            - 'muskingum_cunge': Muskingum-Cunge with Enzyme AD (recommended, fastest)
            - 'diffusive_wave': Diffusive wave approximation (analytical gradients)
            - 'diffusive_wave_ift': Diffusive wave with IFT adjoint (exact gradients)
            - 'irf': Impulse Response Function with soft-masked kernel
            - 'kwt_soft': Soft-gated Kinematic Wave Tracking (differentiable parcels)
            - 'saint_venant': Full dynamic Saint-Venant equations (numerical gradients)
            - 'saint_venant_enzyme': Full SVE with Enzyme AD + CVODES adjoint (exact gradients)
            
            Forward-only methods (Manning's n is NOT optimized):
            - 'lag': Simple lag routing
            - 'kwt': Kinematic Wave Tracking (mizuRoute compatible)
    """
    
    def __init__(self,
                 fuse_config,
                 topology_file: str,
                 hru_areas: np.ndarray,
                 dt: float = 86400.0,
                 warmup_steps: int = 30,
                 spatial_params: bool = True,
                 routing_method: str = 'muskingum_cunge'):
        super().__init__()
        
        self.fuse_config = fuse_config
        self.config_dict = fuse_config.to_dict()
        self.dt_seconds = dt
        self.dt_days = dt / 86400.0
        self.warmup_steps = warmup_steps
        self.spatial_params = spatial_params
        
        # Validate and store routing method
        routing_method = routing_method.lower()
        if routing_method not in ROUTING_METHOD_INFO:
            raise ValueError(f"Unknown routing_method '{routing_method}'. "
                           f"Supported: {list(ROUTING_METHOD_INFO.keys())}")
        
        router_class_name, supports_gradients, gradient_method, config_class = ROUTING_METHOD_INFO[routing_method]
        self.routing_method = routing_method
        self.differentiable_routing = supports_gradients
        self.gradient_method = gradient_method  # 'enzyme', 'native', 'sve', or None
        
        # Parameter bounds
        self.param_names = list(dfuse.PARAM_NAMES)
        n_params = len(self.param_names)
        self.n_params = n_params
        
        lowers = torch.tensor([dfuse.PARAM_BOUNDS[n][0] for n in self.param_names])
        uppers = torch.tensor([dfuse.PARAM_BOUNDS[n][1] for n in self.param_names])
        self.register_buffer('param_lower', lowers)
        self.register_buffer('param_upper', uppers)
        
        # Load network first to get n_hrus
        self.network = self._load_network(topology_file)
        self.n_reaches = self.network.num_reaches()
        self.n_hrus = len(hru_areas)
        
        # FUSE parameters - SPATIALLY VARYING [n_hrus, n_params]
        if spatial_params:
            # Initialize with small random variations
            init_params = torch.zeros(self.n_hrus, n_params)
            init_params = init_params + torch.randn(self.n_hrus, n_params) * 0.2
            self.fuse_raw_params = nn.Parameter(init_params)
        else:
            # Shared parameters (original behavior)
            self.fuse_raw_params = nn.Parameter(torch.zeros(n_params))
        
        # Build ID mapping - use topological order
        topo_order = self.network.topological_order()
        self.reach_ids = list(topo_order)
        self.id_to_idx = {rid: i for i, rid in enumerate(topo_order)}
        self.outlet_reach_id = int(topo_order[-1])
        
        # Create router config based on routing method
        if routing_method == 'saint_venant':
            # Full dynamic Saint-Venant uses its own config class
            # NOTE: SVE is designed for sub-daily timesteps. Daily timesteps require
            # relaxed tolerances and more internal steps.
            config = dmc.SaintVenantConfig()
            config.dt = dt
            config.n_nodes = 10  # Spatial nodes per reach
            
            # Robust settings for daily timesteps (86400s is challenging for SVE)
            config.rel_tol = 1e-3           # Relaxed relative tolerance (default: 1e-4)
            config.abs_tol = 1e-4           # Relaxed absolute tolerance (default: 1e-6)
            config.max_steps = 100000       # Many more internal steps (default: 5000)
            
            # Initial conditions - higher values for stability
            config.initial_depth = 1.0      # Initial water depth [m] (default: 0.5)
            config.initial_velocity = 0.5   # Initial velocity [m/s] (default: 0.1)
            
            # Stability - prevent dry bed numerical issues
            config.min_depth = 0.05         # Minimum depth [m] (default: 0.01)
            config.min_area = 0.5           # Minimum area [m²] (default: 0.1)
            
            config.enable_adjoint = True    # Enable gradient computation
            
            print(f"  NOTE: Saint-Venant with daily dt={dt}s requires relaxed tolerances.")
            print(f"        Consider using 'diffusive_wave' for faster daily routing.")
            
        elif routing_method == 'saint_venant_enzyme':
            # Enzyme-enabled Saint-Venant with CVODES adjoint
            config = dmc.SaintVenantEnzymeConfig()
            config.dt = dt
            config.n_nodes = 10  # Spatial nodes per reach
            
            # Tolerances (Enzyme provides exact Jacobian so can be tighter)
            config.rel_tol = 1e-4
            config.abs_tol = 1e-6
            config.max_steps = 50000
            
            # Initial conditions
            config.initial_depth = 1.0
            config.initial_velocity = 0.5
            config.min_depth = 0.05
            config.min_area = 0.5
            
            # Enzyme-specific settings
            config.use_enzyme_jacobian = True   # Enzyme for Jacobian
            config.use_enzyme_adjoint = True    # Enzyme for adjoint RHS
            config.adjoint_checkpoint_steps = 100  # Checkpointing interval
            config.enable_adjoint = True
            
            print(f"  Saint-Venant with Enzyme AD for exact gradients")
            print(f"    - Enzyme Jacobian: {config.use_enzyme_jacobian}")
            print(f"    - Enzyme Adjoint: {config.use_enzyme_adjoint}")
            print(f"    - Checkpoint interval: {config.adjoint_checkpoint_steps}")
        else:
            # Standard RouterConfig for other methods
            config = dmc.RouterConfig()
            config.dt = dt
            config.num_substeps = 4
            config.enable_gradients = supports_gradients and (gradient_method == 'native')
            
            # Configure method-specific settings
            if routing_method in ['diffusive_wave', 'diffusive_wave_ift']:
                config.dw_num_nodes = 10  # Spatial nodes per reach
            elif routing_method == 'irf':
                config.irf_shape_param = 2.5
                config.irf_max_kernel_size = 500
            elif routing_method == 'kwt_soft':
                config.kwt_gate_steepness = 5.0  # Soft gate steepness
        
        # Create the appropriate router
        self.router = self._create_router(routing_method, config)
        
        # Manning's n - trainable for differentiable routing methods
        initial_log_n = torch.full((self.n_reaches,), np.log(0.035))
        initial_log_n = initial_log_n + torch.randn(self.n_reaches) * 0.1  # Add variation
        
        if self.differentiable_routing:
            # Trainable parameter
            self.log_manning_n = nn.Parameter(initial_log_n)
        else:
            # Non-trainable buffer for forward-only methods
            self.register_buffer('log_manning_n', initial_log_n)
            print(f"  NOTE: Using '{routing_method}' routing (forward-only). "
                  f"Manning's n will NOT be optimized.")
        
        # HRU to reach mapping
        self.register_buffer(
            'mapping_matrix',
            self._build_mapping_matrix(topology_file, hru_areas)
        )
        
        # Store HRU areas for unrouted Q calculation
        self.register_buffer('hru_areas', torch.tensor(hru_areas, dtype=torch.float32))
        
        self.n_states = dfuse_core.get_num_active_states(self.config_dict)
    
    def _load_network(self, topology_file: str) -> dmc.Network:
        """Load river network from topology.nc."""
        ds = xr.open_dataset(topology_file)
        
        seg_ids = ds['segId'].values.astype(int)
        down_seg_ids = ds['downSegId'].values.astype(int)
        lengths = ds['length'].values.astype(float)
        slopes = ds['slope'].values.astype(float)
        mann_n = ds['mann_n'].values if 'mann_n' in ds else np.full(len(seg_ids), 0.035)
        
        network = dmc.Network()
        seg_id_set = set(seg_ids)
        
        # Build upstream mapping
        upstream_map = {int(sid): [] for sid in seg_ids}
        for i, down_id in enumerate(down_seg_ids):
            if int(down_id) in seg_id_set:
                upstream_map[int(down_id)].append(int(seg_ids[i]))
        
        # Add reaches
        for i, sid in enumerate(seg_ids):
            reach = dmc.Reach()
            reach.id = int(sid)
            reach.length = float(lengths[i])
            reach.slope = max(float(slopes[i]), 0.0001)
            reach.manning_n = float(mann_n[i])
            reach.geometry.width_coef = 7.2
            reach.geometry.width_exp = 0.5
            reach.geometry.depth_coef = 0.27
            reach.geometry.depth_exp = 0.3
            reach.upstream_junction_id = int(sid)
            down_id = int(down_seg_ids[i])
            reach.downstream_junction_id = down_id if down_id in seg_id_set else -1
            network.add_reach(reach)
        
        # Add junctions
        for i, sid in enumerate(seg_ids):
            junc = dmc.Junction()
            junc.id = int(sid)
            junc.upstream_reach_ids = upstream_map[int(sid)]
            junc.downstream_reach_ids = [int(sid)]
            network.add_junction(junc)
        
        network.build_topology()
        ds.close()
        return network
    
    def _build_mapping_matrix(self, topology_file: str, hru_areas: np.ndarray) -> torch.Tensor:
        """Build HRU to reach mapping (converts mm/timestep to m³/s)."""
        ds = xr.open_dataset(topology_file)
        hru_to_seg = ds['hruToSegId'].values.astype(int)
        ds.close()
        
        # Use topological order
        topo_order = self.network.topological_order()
        id_to_idx = {rid: i for i, rid in enumerate(topo_order)}
        
        n_hrus = len(hru_to_seg)
        n_reaches = len(topo_order)
        mapping = torch.zeros((n_reaches, n_hrus), dtype=torch.float32)
        
        for h_idx, seg_id in enumerate(hru_to_seg):
            if seg_id in id_to_idx:
                r_idx = id_to_idx[seg_id]
                # mm/timestep * m² / 1000 / dt_seconds = m³/s
                conversion = hru_areas[h_idx] / 1000.0 / self.dt_seconds
                mapping[r_idx, h_idx] = conversion
        
        return mapping
    
    def _create_router(self, routing_method: str, config):
        """Create the appropriate router based on routing method."""
        router_map = {
            'muskingum_cunge': dmc.MuskingumCungeRouter,
            'diffusive_wave': dmc.DiffusiveWaveRouter,
            'diffusive_wave_ift': dmc.DiffusiveWaveIFT,
            'irf': dmc.IRFRouter,
            'kwt_soft': dmc.SoftGatedKWT,
            'saint_venant': dmc.SaintVenantRouter,  # Full dynamic SVE
            'saint_venant_enzyme': dmc.SaintVenantEnzyme,  # SVE + Enzyme AD
            'lag': dmc.LagRouter,
            'kwt': dmc.KWTRouter,
        }
        
        router_class = router_map.get(routing_method)
        if router_class is None:
            raise ValueError(f"Unknown routing method: {routing_method}")
        
        return router_class(self.network, config)
    
    def get_physical_params(self, hru_idx: Optional[int] = None) -> torch.Tensor:
        """
        Get physical parameters from unconstrained.
        
        If spatial_params=True:
            Returns [n_hrus, n_params] or [n_params] if hru_idx given
        Else:
            Returns [n_params]
        """
        if self.spatial_params:
            # [n_hrus, n_params]
            params = self.param_lower + (self.param_upper - self.param_lower) * torch.sigmoid(self.fuse_raw_params)
            if hru_idx is not None:
                return params[hru_idx]
            return params
        else:
            return self.param_lower + (self.param_upper - self.param_lower) * torch.sigmoid(self.fuse_raw_params)
    
    def get_initial_state(self) -> torch.Tensor:
        """Get initial state tensor."""
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
        Forward pass with spatially varying FUSE parameters.
        
        Args:
            forcing: [n_timesteps, n_hrus, 3]
            initial_state: [n_hrus, n_states]
            
        Returns:
            outlet_Q: [n_timesteps]
            runoff: [n_timesteps, n_hrus]
        """
        if initial_state is None:
            initial_state = self.get_initial_state().to(forcing.device)
        
        # Get physical params - returns [n_hrus, n_params] if spatial, [n_params] if shared
        # C++ run_fuse_batch handles BOTH cases natively with OpenMP parallelization!
        phys_params = self.get_physical_params()
        
        # Single efficient call - C++ handles per-HRU params internally
        runoff = DifferentiableFUSEBatch.apply(
            phys_params, initial_state, forcing,
            self.config_dict, self.dt_days
        )
        
        # Map to reaches (reorder to topological order)
        lateral_inflows = torch.matmul(runoff, self.mapping_matrix.T)
        
        # Get Manning's n (from log-space)
        manning_n = torch.exp(self.log_manning_n)
        
        # Dispatch to appropriate routing implementation based on gradient method
        if self.gradient_method == 'enzyme':
            # Muskingum-Cunge with Enzyme AD for gradients
            outlet_Q = DifferentiableRouting.apply(
                lateral_inflows, manning_n, 
                self.router, self.network,
                self.outlet_reach_id, self.dt_seconds
            )
        elif self.gradient_method == 'native':
            # Use router's native gradient computation (IRF, DiffusiveWave, etc.)
            outlet_Q = DifferentiableRoutingNative.apply(
                lateral_inflows, manning_n,
                self.router, self.network,
                self.outlet_reach_id, self.dt_seconds,
                self.reach_ids  # topo_order_ids
            )
        elif self.gradient_method == 'sve':
            # Full dynamic Saint-Venant equations with numerical gradients
            outlet_Q = DifferentiableRoutingSaintVenant.apply(
                lateral_inflows, manning_n,
                self.router, self.network,
                self.outlet_reach_id, self.dt_seconds,
                self.reach_ids  # topo_order_ids
            )
        elif self.gradient_method == 'sve_enzyme':
            # Saint-Venant with Enzyme AD + CVODES adjoint for exact gradients
            outlet_Q = DifferentiableRoutingSaintVenantEnzyme.apply(
                lateral_inflows, manning_n,
                self.router, self.network,
                self.outlet_reach_id, self.dt_seconds,
                self.reach_ids  # topo_order_ids
            )
        else:
            # Forward-only routing (no gradients through routing)
            outlet_Q = self._forward_only_routing(lateral_inflows, manning_n)
        
        return outlet_Q, runoff
    
    def _forward_only_routing(self, lateral_inflows: torch.Tensor, 
                              manning_n: torch.Tensor) -> torch.Tensor:
        """
        Forward-only routing without gradient computation.
        
        Used for routing methods that don't support differentiable optimization
        (lag, kwt).
        
        Args:
            lateral_inflows: [n_timesteps, n_reaches] in m³/s
            manning_n: [n_reaches] Manning's n values
            
        Returns:
            outlet_Q: [n_timesteps] Outlet discharge in m³/s
        """
        n_timesteps = lateral_inflows.shape[0]
        
        # Convert to numpy for C++ router
        inflows_np = lateral_inflows.detach().cpu().numpy().astype(np.float64)
        manning_np = manning_n.detach().cpu().numpy().astype(np.float64)
        
        # Update Manning's n on network
        self.network.set_manning_n_all(manning_np)
        
        # Reset router state
        self.router.reset_state()
        
        # Get topological order for setting inflows
        topo_order = list(self.network.topological_order())
        
        # Route timestep by timestep
        outlet_Q = []
        for t in range(n_timesteps):
            # Set lateral inflows for all reaches
            for i, rid in enumerate(topo_order):
                self.router.set_lateral_inflow(int(rid), float(inflows_np[t, i]))
            
            # Route one timestep
            self.router.route_timestep()
            
            # Get outlet discharge
            outlet_Q.append(self.router.get_discharge(self.outlet_reach_id))
        
        # Convert back to torch tensor (no gradients - detached from computational graph)
        outlet_Q = np.array(outlet_Q, dtype=np.float64)
        return torch.tensor(outlet_Q, dtype=torch.float32, device=lateral_inflows.device)
    
    def get_param_dict(self) -> Dict[str, float]:
        """Get current parameters (mean across HRUs if spatial)."""
        if self.spatial_params:
            phys = self.get_physical_params()  # [n_hrus, n_params]
            return {name: phys[:, i].mean().item() for i, name in enumerate(self.param_names)}
        else:
            phys = self.get_physical_params()
            return {name: phys[i].item() for i, name in enumerate(self.param_names)}
    
    def get_param_stats(self) -> Dict[str, Dict[str, float]]:
        """Get parameter statistics (mean, min, max) across HRUs."""
        if self.spatial_params:
            phys = self.get_physical_params()  # [n_hrus, n_params]
            stats = {}
            for i, name in enumerate(self.param_names):
                vals = phys[:, i]
                stats[name] = {
                    'mean': vals.mean().item(),
                    'min': vals.min().item(),
                    'max': vals.max().item(),
                    'std': vals.std().item()
                }
            return stats
        else:
            phys = self.get_physical_params()
            return {name: {'mean': phys[i].item(), 'min': phys[i].item(), 
                          'max': phys[i].item(), 'std': 0.0} 
                    for i, name in enumerate(self.param_names)}


# =============================================================================
# 4. TRAINING
# =============================================================================

def train_model(model: CoupledFUSERoute,
                forcing: torch.Tensor,
                observed: torch.Tensor,
                n_epochs: int = 100,
                lr: float = 0.01,
                loss_fn: str = 'nse',
                alpha: float = 0.5,
                spatial_reg: float = 0.0,
                verbose: bool = True) -> Dict:
    """
    Train the coupled model.
    
    Args:
        model: CoupledFUSERoute model
        forcing: Forcing data [n_timesteps, n_hrus, 3]
        observed: Observed streamflow [n_timesteps]
        n_epochs: Number of training epochs
        lr: Learning rate
        loss_fn: Loss function: 'nse', 'kge', 'log_nse', 'combined', 'fdc'
        alpha: For combined loss, weight of NSE vs log-NSE (0.5 = equal)
        spatial_reg: L2 regularization on spatial parameter variance (0 = none)
        verbose: Print progress
    """
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-5
    )
    
    initial_state = model.get_initial_state()
    warmup = model.warmup_steps
    
    history = {'loss': [], 'nse': [], 'kge': [], 'log_nse': [], 'peak_nse': []}
    best_loss = float('inf')
    best_state = None
    
    # Select loss function
    if loss_fn == 'nse':
        loss_func = nse_loss
    elif loss_fn == 'kge':
        loss_func = kge_loss
    elif loss_fn == 'log_nse':
        loss_func = log_nse_loss
    elif loss_fn == 'combined':
        loss_func = lambda s, o, m: combined_nse_loss(s, o, m, alpha=alpha)
    elif loss_fn == 'fdc':
        loss_func = flow_duration_loss
    elif loss_fn == 'asymmetric':
        loss_func = lambda s, o, m: asymmetric_nse_loss(s, o, m, under_weight=2.0)
    elif loss_fn == 'peak_weighted':
        loss_func = peak_weighted_nse
    elif loss_fn == 'triple':
        loss_func = triple_objective_loss
    else:
        raise ValueError(f"Unknown loss: {loss_fn}")
    
    if verbose:
        loss_desc = f"{loss_fn.upper()}"
        if loss_fn == 'combined':
            loss_desc += f" (α={alpha})"
        if spatial_reg > 0:
            loss_desc += f" + reg={spatial_reg}"
        print(f"\nTraining for {n_epochs} epochs with {loss_desc} loss")
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
        
        # Primary loss
        loss = loss_func(sim_valid, obs_valid, mask)
        
        # Spatial regularization (penalize high variance across HRUs)
        if spatial_reg > 0 and model.spatial_params:
            params = model.get_physical_params()  # [n_hrus, n_params]
            # Normalize by param range to make comparable
            param_range = model.param_upper - model.param_lower
            normalized_params = (params - model.param_lower) / param_range
            # Variance across HRUs for each param
            spatial_var = normalized_params.var(dim=0).mean()
            loss = loss + spatial_reg * spatial_var
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss.detach())
        
        # Record metrics
        with torch.no_grad():
            nse_val = 1 - nse_loss(sim_valid, obs_valid, mask).item()
            kge_val = 1 - kge_loss(sim_valid, obs_valid, mask).item()
            log_nse_val = 1 - log_nse_loss(sim_valid, obs_valid, mask).item()
            peak_nse_val = 1 - peak_weighted_nse(sim_valid, obs_valid, mask).item()
        
        history['loss'].append(loss.item())
        history['nse'].append(nse_val)
        history['kge'].append(kge_val)
        history['log_nse'].append(log_nse_val)
        history['peak_nse'].append(peak_nse_val)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if verbose and (epoch + 1) % 5 == 0:
            mean_n = torch.exp(model.log_manning_n).mean().item()
            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | "
                  f"NSE: {nse_val:.3f} | KGE: {kge_val:.3f} | logNSE: {log_nse_val:.3f} | peakNSE: {peak_nse_val:.3f} | n̄: {mean_n:.4f}")
    
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
    """Load Bow at Banff data from FUSE input file."""
    print("Loading data...")
    
    # Topology file for HRU areas
    topo_file = data_path / "settings/mizuRoute/topology.nc"
    topo_ds = xr.open_dataset(topo_file)
    hru_areas = topo_ds['area'].values.astype(np.float64)  # m²
    topo_ds.close()
    
    # FUSE forcing file (has precip, temp, pet, and q_obs)
    fuse_input = data_path / "forcing/FUSE_input/Bow_at_Banff_distributed_input.nc"
    ds = xr.open_dataset(fuse_input)
    
    # Extract forcing [time, lat(=hru), lon(=1)] -> [time, hru]
    precip = ds['pr'].values[:, :, 0]      # mm/day
    temp = ds['temp'].values[:, :, 0]      # °C
    pet = ds['pet'].values[:, :, 0]        # mm/day
    q_obs_mm = ds['q_obs'].values[:, :, 0] # mm/day (at outlet HRU)
    
    # Get time info
    time_vals = ds['time'].values
    n_timesteps = len(time_vals)
    n_hrus = precip.shape[1]
    
    print(f"  Time range: {n_timesteps} days")
    print(f"  HRUs: {n_hrus}")
    
    ds.close()
    
    # Stack forcing: [time, hru, 3] for (precip, pet, temp)
    forcing = torch.tensor(
        np.stack([precip, pet, temp], axis=2), 
        dtype=torch.float32
    )
    print(f"  Forcing: {forcing.shape}")
    print(f"    Precip: mean={precip.mean():.2f}, max={precip.max():.2f} mm/day")
    print(f"    PET: mean={pet.mean():.2f}, max={pet.max():.2f} mm/day")
    print(f"    Temp: mean={temp.mean():.1f}, range=[{temp.min():.1f}, {temp.max():.1f}] °C")
    
    # Convert q_obs from mm/day to m³/s
    # q_obs appears to be basin-average, so use total watershed area
    total_area = hru_areas.sum()  # m²
    # mm/day * m² / 1000 / 86400 = m³/s
    # But q_obs might be at outlet HRU only - let's use first non-nan column
    # Actually the file has q_obs at all HRUs but it's the same value (outlet)
    q_obs_mm_outlet = q_obs_mm[:, 0]  # Take first HRU (all should be same)
    
    # Convert: mm/day over total area to m³/s
    q_obs_cms = q_obs_mm_outlet * total_area / 1000.0 / 86400.0
    
    # Handle missing values
    q_obs_cms[q_obs_mm_outlet < 0] = float('nan')
    
    obs = torch.tensor(q_obs_cms, dtype=torch.float32)
    valid_count = (~torch.isnan(obs)).sum().item()
    print(f"  Obs: {valid_count} valid, mean={np.nanmean(q_obs_cms):.2f} m³/s")
    
    return forcing, obs, str(topo_file), hru_areas


# =============================================================================
# 6. PLOTTING FUNCTIONS
# =============================================================================

def plot_hydrograph(sim_Q: np.ndarray, obs_Q: np.ndarray, 
                    warmup: int = 30, save_path: Optional[str] = None):
    """Plot observed vs simulated streamflow."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Full time series
    ax1 = axes[0]
    t = np.arange(len(obs_Q))
    ax1.plot(t, obs_Q, 'b-', label='Observed', linewidth=1.5, alpha=0.8)
    ax1.plot(t, sim_Q, 'r-', label='Simulated', linewidth=1.5, alpha=0.8)
    ax1.axvline(warmup, color='gray', linestyle='--', alpha=0.5, label=f'Spinup ({warmup} days)')
    ax1.set_ylabel('Discharge [m³/s]')
    ax1.set_title(f'Streamflow Comparison (metrics computed on days {warmup}-{len(obs_Q)})')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Compute metrics (post-warmup)
    mask = ~np.isnan(obs_Q[warmup:])
    if mask.sum() > 0:
        obs_v = obs_Q[warmup:][mask]
        sim_v = sim_Q[warmup:][mask]
        nse = 1 - np.sum((sim_v - obs_v)**2) / np.sum((obs_v - obs_v.mean())**2)
        rmse = np.sqrt(np.mean((sim_v - obs_v)**2))
        bias = (sim_v.mean() - obs_v.mean()) / obs_v.mean() * 100
        ax1.text(0.02, 0.95, f'NSE: {nse:.3f}\nRMSE: {rmse:.2f} m³/s\nBias: {bias:.1f}%',
                 transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Scatter plot
    ax2 = axes[1]
    obs_valid = obs_Q[warmup:]
    sim_valid = sim_Q[warmup:]
    mask = ~np.isnan(obs_valid)
    ax2.scatter(obs_valid[mask], sim_valid[mask], alpha=0.5, s=10)
    max_val = max(np.nanmax(obs_valid), np.nanmax(sim_valid))
    ax2.plot([0, max_val], [0, max_val], 'k--', label='1:1 line')
    ax2.set_xlabel('Observed [m³/s]')
    ax2.set_ylabel('Simulated [m³/s]')
    ax2.set_title(f'Scatter Plot (days {warmup}-{len(obs_Q)})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved hydrograph to {save_path}")
    plt.show()


def plot_network_manning(data_path: Path, manning_n: np.ndarray, 
                         topo_order: list, save_path: Optional[str] = None):
    """Plot river network with Manning's n values using shapefiles."""
    
    # Load shapefiles
    catchment_shp = data_path / "shapefiles/catchment/Bow_at_Banff_distributed_HRUs_GRUs.shp"
    network_shp = data_path / "shapefiles/river_network/Bow_at_Banff_distributed_riverNetwork_delineate.shp"
    
    if not catchment_shp.exists() or not network_shp.exists():
        print("  Warning: Shapefiles not found, skipping network plot")
        return
    
    gdf_catchment = gpd.read_file(catchment_shp)
    gdf_network = gpd.read_file(network_shp)
    
    # Create mapping from topo_order index to Manning's n
    # topo_order contains reach IDs in topological order
    reach_to_manning = {int(rid): manning_n[i] for i, rid in enumerate(topo_order)}
    
    # Map Manning's n to catchments (GRU_ID matches reach ID)
    gdf_catchment['manning_n'] = gdf_catchment['GRU_ID'].map(reach_to_manning)
    
    # Map Manning's n to network segments (LINKNO or GRU_ID)
    if 'GRU_ID' in gdf_network.columns:
        gdf_network['manning_n'] = gdf_network['GRU_ID'].map(reach_to_manning)
    elif 'LINKNO' in gdf_network.columns:
        gdf_network['manning_n'] = gdf_network['LINKNO'].map(reach_to_manning)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Catchments colored by Manning's n
    ax1 = axes[0]
    gdf_catchment.plot(column='manning_n', cmap='viridis', legend=True,
                       legend_kwds={'label': "Manning's n", 'shrink': 0.8},
                       ax=ax1, edgecolor='black', linewidth=0.5)
    gdf_network.plot(ax=ax1, color='blue', linewidth=1.5, alpha=0.7)
    ax1.set_title(f"Catchments - Manning's n\nmean={manning_n.mean():.4f}")
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    
    # Plot 2: Network colored by Manning's n with stream order
    ax2 = axes[1]
    gdf_catchment.plot(ax=ax2, color='lightgray', edgecolor='darkgray', linewidth=0.3)
    
    # Color network by Manning's n, width by stream order
    if 'strmOrder' in gdf_network.columns:
        for order in sorted(gdf_network['strmOrder'].unique()):
            subset = gdf_network[gdf_network['strmOrder'] == order]
            subset.plot(column='manning_n', cmap='viridis', ax=ax2,
                       linewidth=0.5 + order * 0.8, legend=False,
                       vmin=manning_n.min(), vmax=manning_n.max())
    else:
        gdf_network.plot(column='manning_n', cmap='viridis', ax=ax2,
                        linewidth=2, legend=True)
    
    # Add colorbar for network
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                                norm=mcolors.Normalize(vmin=manning_n.min(), vmax=manning_n.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2, shrink=0.8)
    cbar.set_label("Manning's n")
    
    ax2.set_title(f"River Network - Manning's n\nrange=[{manning_n.min():.4f}, {manning_n.max():.4f}]")
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved network map to {save_path}")
    plt.show()


def plot_catchment_runoff(data_path: Path, runoff: np.ndarray, topo_order: list,
                          timestep: int = -1, save_path: Optional[str] = None):
    """Plot catchment map colored by runoff at a given timestep."""
    
    catchment_shp = data_path / "shapefiles/catchment/Bow_at_Banff_distributed_HRUs_GRUs.shp"
    if not catchment_shp.exists():
        print("  Warning: Catchment shapefile not found")
        return
    
    gdf = gpd.read_file(catchment_shp)
    
    # Map runoff to catchments
    reach_to_runoff = {int(rid): runoff[timestep, i] for i, rid in enumerate(topo_order)}
    gdf['runoff'] = gdf['GRU_ID'].map(reach_to_runoff)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf.plot(column='runoff', cmap='Blues', legend=True,
             legend_kwds={'label': 'Runoff [mm/day]', 'shrink': 0.8},
             ax=ax, edgecolor='black', linewidth=0.5)
    
    ax.set_title(f'Runoff Distribution (timestep {timestep})')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved runoff map to {save_path}")
    plt.show()


def plot_spatial_params(data_path: Path, model, topo_order: list,
                        params_to_plot: List[str] = None,
                        save_path: Optional[str] = None):
    """Plot spatial distribution of FUSE parameters."""
    
    catchment_shp = data_path / "shapefiles/catchment/Bow_at_Banff_distributed_HRUs_GRUs.shp"
    if not catchment_shp.exists():
        print("  Warning: Catchment shapefile not found")
        return
    
    if not model.spatial_params:
        print("  Warning: Model doesn't have spatial params")
        return
    
    gdf = gpd.read_file(catchment_shp)
    phys_params = model.get_physical_params().detach().numpy()  # [n_hrus, n_params]
    
    # Default: plot key hydrologically-significant parameters
    if params_to_plot is None:
        params_to_plot = ['S1_max', 'S2_max', 'ks', 'ki', 'ku', 'f_tens']
    
    # Filter to available params
    params_to_plot = [p for p in params_to_plot if p in model.param_names]
    
    n_plots = len(params_to_plot)
    if n_plots == 0:
        return
    
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, param_name in enumerate(params_to_plot):
        ax = axes[idx]
        param_idx = model.param_names.index(param_name)
        param_vals = phys_params[:, param_idx]
        
        # Map to catchments
        hru_to_param = {int(rid): param_vals[i] for i, rid in enumerate(topo_order)}
        gdf['param_val'] = gdf['GRU_ID'].map(hru_to_param)
        
        gdf.plot(column='param_val', cmap='viridis', legend=True,
                 legend_kwds={'shrink': 0.6},
                 ax=ax, edgecolor='black', linewidth=0.3)
        
        ax.set_title(f'{param_name}\nmean={param_vals.mean():.2f}, std={param_vals.std():.2f}')
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # Hide unused axes
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Spatial Distribution of FUSE Parameters', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved spatial params to {save_path}")
    plt.show()


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training loss and metrics."""
    has_log_nse = 'log_nse' in history and len(history['log_nse']) > 0
    n_cols = 4 if has_log_nse else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 4))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss
    axes[0].semilogy(epochs, history['loss'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (log scale)')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # NSE
    axes[1].plot(epochs, history['nse'], label='NSE')
    if has_log_nse:
        axes[1].plot(epochs, history['log_nse'], label='log-NSE', alpha=0.7)
        axes[1].legend()
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('NSE')
    axes[1].set_title('Nash-Sutcliffe Efficiency')
    axes[1].grid(True, alpha=0.3)
    
    # KGE
    axes[2].plot(epochs, history['kge'])
    axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KGE')
    axes[2].set_title('Kling-Gupta Efficiency')
    axes[2].grid(True, alpha=0.3)
    
    # Log-NSE (if available)
    if has_log_nse:
        axes[3].plot(epochs, history['log_nse'], color='orange')
        axes[3].axhline(0, color='gray', linestyle='--', alpha=0.5)
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('log-NSE')
        axes[3].set_title('Log-Transformed NSE (low flows)')
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved training history to {save_path}")
    plt.show()


def diagnose_model(model, forcing, observed, warmup=30):
    """Run diagnostic checks on model."""
    print("\n" + "="*60)
    print("DIAGNOSTIC CHECKS")
    print("="*60)
    
    # Check forcing statistics
    print("\nForcing statistics:")
    precip = forcing[:, :, 0].numpy()
    pet = forcing[:, :, 1].numpy()
    temp = forcing[:, :, 2].numpy()
    print(f"  Precip: mean={precip.mean():.2f}, max={precip.max():.2f} mm/day")
    print(f"  PET: mean={pet.mean():.2f}, max={pet.max():.2f} mm/day")
    print(f"  Temp: mean={temp.mean():.1f}, range=[{temp.min():.1f}, {temp.max():.1f}] °C")
    
    # Check observed flow
    obs_np = observed.numpy()
    print(f"\nObserved streamflow:")
    print(f"  Mean: {np.nanmean(obs_np):.2f} m³/s")
    print(f"  Range: [{np.nanmin(obs_np):.2f}, {np.nanmax(obs_np):.2f}] m³/s")
    
    # Run forward pass and check intermediate outputs
    print("\nModel forward pass:")
    with torch.no_grad():
        initial_state = model.get_initial_state()
        sim_Q, runoff = model(forcing, initial_state)
        
        runoff_np = runoff.numpy()
        print(f"  Runoff: mean={runoff_np.mean():.4f}, max={runoff_np.max():.4f} mm/day")
        print(f"  Simulated Q: mean={sim_Q.mean():.2f}, max={sim_Q.max():.2f} m³/s")
        
        # Check ratio
        ratio = sim_Q.mean().item() / np.nanmean(obs_np)
        print(f"\n  Sim/Obs ratio: {ratio:.2f}")
        if ratio < 0.1:
            print("  ⚠ Simulated flow is ~10x too LOW!")
        elif ratio > 10:
            print("  ⚠ Simulated flow is ~10x too HIGH!")
        elif 0.5 < ratio < 2:
            print("  ✓ Simulated flow magnitude is reasonable")
    
    # Check parameter values
    print("\nCurrent parameters:")
    if model.spatial_params:
        params = model.get_physical_params()  # [n_hrus, n_params]
        for i, name in enumerate(model.param_names[:5]):  # First 5
            vals = params[:, i]
            print(f"  {name}: mean={vals.mean().item():.4f}, range=[{vals.min().item():.4f}, {vals.max().item():.4f}]")
        print("  ...")
    else:
        params = model.get_physical_params()
        for i, name in enumerate(model.param_names[:5]):  # First 5
            print(f"  {name}: {params[i].item():.4f}")
        print("  ...")
    
    manning = torch.exp(model.log_manning_n).detach().numpy()
    print(f"\nManning's n: mean={manning.mean():.4f}")
    
    return sim_Q.numpy(), runoff_np


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    DATA_PATH = Path("/Users/darrieythorsson/compHydro/data/CONFLUENCE_data/domain_Bow_at_Banff_distributed")
    OUTPUT_DIR = Path("./outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    forcing, observed, topo_file, hru_areas = load_data(DATA_PATH)
    
    # --- Choose routing method ---
    # Differentiable methods (Manning's n is optimized):
    #   'muskingum_cunge'       - Muskingum-Cunge with Enzyme AD (recommended, fastest)
    #   'diffusive_wave'        - Diffusive wave approximation with analytical gradients
    #   'diffusive_wave_ift'    - Diffusive wave with IFT adjoint (exact gradients)
    #   'irf'                   - Impulse Response Function with soft-masked kernel
    #   'kwt_soft'              - Soft-gated Kinematic Wave Tracking
    #   'saint_venant'          - Full dynamic SVE (numerical gradients, slow)
    #   'saint_venant_enzyme'   - Full dynamic SVE with Enzyme AD (exact gradients)
    #
    # Forward-only methods (Manning's n is NOT optimized):
    #   'lag'                   - Simple lag routing
    #   'kwt'                   - Kinematic Wave Tracking (mizuRoute compatible)
    ROUTING_METHOD = 'diffusive_wave'  # Change this to use different routing
    
    print("\nInitializing model...")
    model = CoupledFUSERoute(
        fuse_config=dfuse.VIC_CONFIG,
        topology_file=topo_file,
        hru_areas=hru_areas,
        dt=86400.0,
        warmup_steps=365,  # 1 year spinup
        spatial_params=True,  # Spatially varying FUSE parameters
        routing_method=ROUTING_METHOD
    )
    
    # Count total parameters
    n_fuse_params = model.n_hrus * len(model.param_names) if model.spatial_params else len(model.param_names)
    n_manning_params = model.n_reaches if model.differentiable_routing else 0
    n_total = n_fuse_params + n_manning_params
    
    print(f"  Routing method: {model.routing_method}")
    print(f"  Differentiable routing: {model.differentiable_routing}")
    if model.differentiable_routing:
        gradient_desc = {
            'enzyme': 'Enzyme AD (reverse-mode)',
            'native': 'Native router AD',
            'sve': 'Saint-Venant numerical gradients (finite difference)',
            'sve_enzyme': 'Saint-Venant Enzyme AD + CVODES adjoint (exact)'
        }
        print(f"  Gradient method: {gradient_desc.get(model.gradient_method, model.gradient_method)}")
    print(f"  FUSE param types: {len(model.param_names)}")
    print(f"  Reaches: {model.n_reaches}")
    print(f"  HRUs: {model.n_hrus}")
    print(f"  Spatially varying: {model.spatial_params}")
    print(f"  Total optimizable parameters: {n_total}")
    print(f"    - FUSE: {n_fuse_params} ({model.n_hrus} HRUs × {len(model.param_names)} params)")
    if model.differentiable_routing:
        print(f"    - Manning's n: {n_manning_params} (trainable)")
    else:
        print(f"    - Manning's n: {model.n_reaches} (fixed, not optimized)")
        print(f"  Routing is forward-only (no gradient computation)")
    if model.spatial_params:
        print(f"  ⚠ NOTE: Spatial params = slower training (~{model.n_hrus}x per epoch)")
    manning_init = torch.exp(model.log_manning_n).detach().numpy()
    print(f"  Initial Manning's n: mean={manning_init.mean():.4f}, range=[{manning_init.min():.4f}, {manning_init.max():.4f}]")
    
    # Print parameter bounds
    print("\nParameter bounds (from dfuse.PARAM_BOUNDS):")
    for i, name in enumerate(model.param_names):
        lo, hi = model.param_lower[i].item(), model.param_upper[i].item()
        print(f"  {name}: [{lo:.2f}, {hi:.2f}]")
    
    # Run diagnostics BEFORE training
    sim_Q_init, _ = diagnose_model(model, forcing, observed, warmup=model.warmup_steps)
    
    # Check gradient computation
    print("\nVerifying gradient flow...")
    test_forcing = forcing[:10]
    test_state = model.get_initial_state()
    model.zero_grad()
    Q, _ = model(test_forcing, test_state)
    loss = Q.sum()
    loss.backward()
    
    fuse_grad = model.fuse_raw_params.grad
    
    if fuse_grad is not None and (fuse_grad != 0).any():
        print("  ✓ FUSE gradients flowing")
        if model.spatial_params:
            print(f"    shape: {list(fuse_grad.shape)}, nonzero: {(fuse_grad != 0).sum().item()}/{fuse_grad.numel()}")
    else:
        print("  ✗ FUSE gradients NOT flowing!")
    
    if model.differentiable_routing:
        manning_grad = model.log_manning_n.grad
        if manning_grad is not None and (manning_grad != 0).any():
            grad_method_str = "Enzyme AD" if model.gradient_method == 'enzyme' else "Native router AD"
            print(f"  ✓ Manning gradients flowing ({grad_method_str})")
            print(f"    grad sum: {manning_grad.sum().item():.6f}, max: {manning_grad.abs().max().item():.6f}")
        else:
            print("  ✗ Manning gradients NOT flowing!")
    else:
        print(f"  ℹ Manning's n not optimized (using {model.routing_method} forward-only routing)")
    
    # Plot initial state
    print("\n" + "="*60)
    print("INITIAL STATE (before training)")
    print("="*60)
    plot_hydrograph(sim_Q_init, observed.numpy(), warmup=model.warmup_steps, 
                    save_path=str(OUTPUT_DIR / "hydrograph_initial.png"))
    
    # Train with combined loss (NSE + log-NSE) - BEST RESULT SO FAR
    # Single phase with α=0.6 achieved NSE=0.727, KGE=0.777, log-NSE=0.786
    # Now trying triple objective to improve peak capture
    results = train_model(
        model=model,
        forcing=forcing,
        observed=observed,
        n_epochs=600,           # More epochs for complex loss
        lr=0.03,                # Learning rate
        loss_fn='triple',       # NSE + log-NSE + peak-weighted
        alpha=0.6,              # Not used for triple, but kept for reference
        spatial_reg=0.0005,     # Slightly less regularization
        verbose=True
    )
    
    # Final results
    print("\n" + "=" * 60)
    print("CALIBRATED PARAMETERS")
    print("=" * 60)
    
    if model.spatial_params:
        print("\nFUSE Parameters (spatial mean ± std, [min, max]):")
        param_stats = model.get_param_stats()
        for name, stats in param_stats.items():
            if stats['std'] > 0.001:
                print(f"  {name}: {stats['mean']:.4f} ± {stats['std']:.4f} [{stats['min']:.4f}, {stats['max']:.4f}]")
            else:
                print(f"  {name}: {stats['mean']:.4f}")
    else:
        print("\nFUSE Parameters:")
        for name, val in results['final_params'].items():
            print(f"  {name}: {val:.4f}")
    
    print(f"\nManning's n: mean={results['final_manning_n'].mean():.4f}, "
          f"range=[{results['final_manning_n'].min():.4f}, {results['final_manning_n'].max():.4f}]")
    
    print(f"\nFinal NSE: {results['history']['nse'][-1]:.3f}")
    print(f"Final KGE: {results['history']['kge'][-1]:.3f}")
    if 'log_nse' in results['history']:
        print(f"Final log-NSE: {results['history']['log_nse'][-1]:.3f}")
    if 'peak_nse' in results['history']:
        print(f"Final peak-NSE: {results['history']['peak_nse'][-1]:.3f}")
    
    # Generate final simulation
    print("\n" + "="*60)
    print("FINAL PLOTS")
    print("="*60)
    
    with torch.no_grad():
        sim_Q_final, runoff_final = model(forcing, model.get_initial_state())
        sim_Q_final = sim_Q_final.numpy()
        runoff_final = runoff_final.numpy()
    
    # DIAGNOSTIC: Compare routed vs unrouted (instantaneous sum)
    print("\n" + "="*60)
    print("ROUTING DIAGNOSTIC")
    print("="*60)
    
    # Compute "unrouted" Q = sum of all HRU runoff converted to m³/s
    # runoff_final is [time, hru] in mm/day
    # Need to convert: mm/day * m² / 1000 / 86400 = m³/s
    hru_areas_np = model.hru_areas.numpy()  # [n_hrus] in m²
    
    unrouted_Q = np.zeros(len(runoff_final))
    for t in range(len(runoff_final)):
        # Sum (runoff_mm * area_m2) across HRUs, convert to m³/s
        unrouted_Q[t] = np.sum(runoff_final[t, :] * hru_areas_np) / 1000.0 / 86400.0
    
    obs_np = observed.numpy()
    warmup = model.warmup_steps
    mask = ~np.isnan(obs_np[warmup:])
    
    # NSE for routed
    obs_v = obs_np[warmup:][mask]
    sim_routed = sim_Q_final[warmup:][mask]
    nse_routed = 1 - np.sum((sim_routed - obs_v)**2) / np.sum((obs_v - obs_v.mean())**2)
    
    # NSE for unrouted
    sim_unrouted = unrouted_Q[warmup:][mask]
    nse_unrouted = 1 - np.sum((sim_unrouted - obs_v)**2) / np.sum((obs_v - obs_v.mean())**2)
    
    print(f"  Routed Q:   mean={sim_routed.mean():.2f} m³/s, NSE={nse_routed:.3f}")
    print(f"  Unrouted Q: mean={sim_unrouted.mean():.2f} m³/s, NSE={nse_unrouted:.3f}")
    print(f"  Observed:   mean={obs_v.mean():.2f} m³/s")
    
    if nse_unrouted > nse_routed + 0.05:
        print("  ⚠ Routing is DEGRADING performance - consider simpler routing or check parameters")
    elif nse_routed > nse_unrouted + 0.05:
        print("  ✓ Routing is IMPROVING performance")
    else:
        print("  ~ Routing has minimal effect")
    
    # Check which parameters are at bounds
    print("\n" + "="*60)
    print("PARAMETERS AT BOUNDS")
    print("="*60)
    
    if model.spatial_params:
        params = model.get_physical_params()  # [n_hrus, n_params]
        params_mean = params.mean(dim=0)  # [n_params]
    else:
        params_mean = model.get_physical_params()
    
    at_lower = []
    at_upper = []
    for i, name in enumerate(model.param_names):
        val = params_mean[i].item()
        lo, hi = model.param_lower[i].item(), model.param_upper[i].item()
        # Check if within 1% of bound
        range_val = hi - lo
        if val < lo + 0.01 * range_val:
            at_lower.append(f"{name} ({val:.4f} ≈ {lo:.2f})")
        elif val > hi - 0.01 * range_val:
            at_upper.append(f"{name} ({val:.4f} ≈ {hi:.2f})")
    
    if at_lower:
        print(f"  At LOWER bound: {', '.join(at_lower)}")
    if at_upper:
        print(f"  At UPPER bound: {', '.join(at_upper)}")
    if not at_lower and not at_upper:
        print("  ✓ No parameters at bounds")
    
    # Plot final hydrograph
    plot_hydrograph(sim_Q_final, observed.numpy(), warmup=model.warmup_steps,
                    save_path=str(OUTPUT_DIR / "hydrograph_final.png"))
    
    # Plot routed vs unrouted comparison
    fig, ax = plt.subplots(figsize=(14, 5))
    t = np.arange(len(obs_np))
    ax.plot(t, obs_np, 'b-', label='Observed', linewidth=1.5, alpha=0.8)
    ax.plot(t, sim_Q_final, 'r-', label=f'Routed (NSE={nse_routed:.3f})', linewidth=1.5, alpha=0.8)
    ax.plot(t, unrouted_Q, 'g--', label=f'Unrouted (NSE={nse_unrouted:.3f})', linewidth=1.5, alpha=0.7)
    ax.axvline(warmup, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Day')
    ax.set_ylabel('Discharge [m³/s]')
    ax.set_title('Routed vs Unrouted Streamflow Comparison')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "routed_vs_unrouted.png"), dpi=150, bbox_inches='tight')
    print(f"  Saved routed vs unrouted to outputs/routed_vs_unrouted.png")
    plt.show()
    
    # Plot training history
    plot_training_history(results['history'],
                          save_path=str(OUTPUT_DIR / "training_history.png"))
    
    # Plot network with Manning's n (using shapefiles)
    topo_order = list(model.network.topological_order())
    plot_network_manning(DATA_PATH, results['final_manning_n'], topo_order,
                         save_path=str(OUTPUT_DIR / "network_manning.png"))
    
    # Plot catchment runoff at peak flow timestep (AFTER warmup)
    warmup = model.warmup_steps
    peak_timestep_post_warmup = np.argmax(sim_Q_final[warmup:])
    peak_timestep = warmup + peak_timestep_post_warmup
    plot_catchment_runoff(DATA_PATH, runoff_final, topo_order, 
                          timestep=peak_timestep,
                          save_path=str(OUTPUT_DIR / "catchment_runoff_peak.png"))
    
    # Plot spatial FUSE parameters (if spatial)
    if model.spatial_params:
        plot_spatial_params(DATA_PATH, model, topo_order,
                           params_to_plot=['S1_max', 'S2_max', 'ks', 'ku', 'f_tens', 'melt_rate'],
                           save_path=str(OUTPUT_DIR / "spatial_fuse_params.png"))
    
    print("\n" + "="*60)
    print(f"Outputs saved to {OUTPUT_DIR.absolute()}")
    print("="*60)


if __name__ == "__main__":
    main()