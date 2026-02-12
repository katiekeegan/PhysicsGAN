"""
PDF Parameter Inference Plotting Driver with Permutation-Invariant NPE

This script reloads the permutation-invariant NPE posterior from permutation_invariant_npe.py
and generates comprehensive comparison plots against SBI methods and other baselines.

Key Features:
- Uses permutation-invariant NPE as the primary inference method
- Simplified architecture (no PointNetPMA or head networks needed)
- Direct posterior sampling from NPE
- Comparison against SBI methods (NPE, MCABC, Wasserstein MCABC)
- Histogram-based uncertainty quantification

Problem Types Supported:
- simplified_dis: Simplified DIS problem with 1D PDF inputs (x only)
- mceg4dis: Monte Carlo Event Generator for DIS with 2D PDF inputs (x, Q2)

Usage:
    python plotting_driver.py --problem simplified_dis --num_events 10000

    # With custom true parameters:
    python plotting_driver.py --problem mceg4dis --num_events 10000 --true_params -0.71 3.48 1.34 2.33
"""

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sbi.utils.torchutils import BoxUniform
from simulator import MCEGSimulator, SimplifiedDIS

np.random.seed(42)
torch.manual_seed(42)

# Set up matplotlib for high-quality plots
plt.style.use("default")
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{libertine}\usepackage{zi4}\usepackage{newtxmath}"
)
import matplotlib.font_manager as _fm

# Choose a preferred Times-family serif font when available,
# otherwise fall back to a commonly available serif.
_preferred_serif = [
    "Times New Roman",
    "Times",
    "Times-Roman",
    "TeX Gyre Termes",
    "DejaVu Serif",
]
_available_names = {_f.name for _f in _fm.fontManager.ttflist}
_chosen_serif = next((p for p in _preferred_serif if p in _available_names), None)
if _chosen_serif is None:
    # pick a default serif if none of the preferred names are present
    _chosen_serif = "DejaVu Serif" if "DejaVu Serif" in _available_names else (
        next(iter(_available_names), None)
    )
    print(f"⚠️  Warning: preferred Times-family fonts not found. Using '{_chosen_serif}' instead.")

plt.rcParams.update(
    {
        "font.size": 22,
        "axes.labelsize": 22,
        "axes.titlesize": 24,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "text.usetex": False,
        # Set the global family directly to the chosen serif for consistent rendering
        "font.family": _chosen_serif if _chosen_serif is not None else "serif",
        "font.serif": [_chosen_serif] if _chosen_serif is not None else [],
        # Use a custom mathtext font to match the chosen serif for math expressions
        "mathtext.fontset": "custom",
        "mathtext.rm": _chosen_serif,
        "mathtext.it": _chosen_serif + ":italic",
        "mathtext.bf": _chosen_serif + ":bold",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "axes.axisbelow": True,
    }
)


def get_prior_for_problem(problem: str, device: str = "cpu") -> BoxUniform:
    device = torch.device(device)
    if problem == "simplified_dis":
        return BoxUniform(
            low=torch.tensor([-1.0, 0.0, -1.0, 0.0], device=device),
            high=torch.tensor([0.0, 5.0, 0.0, 5.0], device=device),
        )
    if problem in {"mceg", "mceg4dis"}:
        return BoxUniform(
            low=torch.tensor([-1.0, 0.0, -10.0, -10.0], device=device),
            high=torch.tensor([10.0, 10.0, 10.0, 10.0], device=device),
        )
    raise ValueError(f"Unknown problem: {problem}")

def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_npe_posterior(problem="simplified_dis", device="cpu"):
    """
    Load the permutation-invariant NPE posterior from saved checkpoint.
    
    If checkpoint doesn't exist, builds and saves it.
    
    Returns:
        posterior: The NPE posterior object for sampling
        cfg: Dict with config info (posterior, prior, num_events, etc.)
    """
    import pickle
    import os
    
    import pickle
    import os
    save_dir = f"npe_checkpoints/{problem}"

    # Support for sequential NPE
    use_sequential = getattr(load_npe_posterior, "use_sequential", False)
    if use_sequential:
        posterior_path = os.path.join(save_dir, "snpe_c_posterior.pkl")
        config_path = os.path.join(save_dir, "snpe_c_config.pkl")
    else:
        posterior_path = os.path.join(save_dir, "posterior.pkl")
        config_path = os.path.join(save_dir, "config.pkl")

    # Try to load from checkpoint
    if os.path.exists(posterior_path) and os.path.exists(config_path):
        print(f"🔍 Loading {'sequential ' if use_sequential else ''}permutation-invariant NPE posterior from {posterior_path}...")
        try:
            with open(posterior_path, "rb") as f:
                posterior = pickle.load(f)
            with open(config_path, "rb") as f:
                cfg = pickle.load(f)
            print(f"✓ NPE posterior loaded successfully from checkpoint")
            return posterior, cfg
        except Exception as e:
            print(f"⚠️  Could not load from checkpoint: {e}. Building new posterior...")

    # Build and save if not found (only for standard NPE)
    if use_sequential:
        raise FileNotFoundError("Sequential NPE checkpoint not found. Please run sequential_permutation_invariant_npe.py first.")

    print(f"🔍 Building permutation-invariant NPE posterior for {problem}...")
    from permutation_invariant_npe import build_permutation_invariant_posterior

    posterior, cfg_obj = build_permutation_invariant_posterior(
        problem=problem,
        num_thetas=10000,
        num_events=10000,
        latent_dim=512,
        training_device=torch.device(device),
        simulation_device=torch.device(device),
        storage_device=torch.device("cpu"),
        training_batch_size=256,
        chunk_size=1024,
        show_progress_bars=True,
    )

    # Return as dict for consistency with loaded version
    cfg = {
        'posterior': posterior,
        'prior': cfg_obj.prior,
        'num_events': cfg_obj.num_events,
        'x_dim': cfg_obj.x_dim,
        'training_device': cfg_obj.training_device,
        'simulation_device': cfg_obj.simulation_device,
        'problem': problem,
    }

    print(f"✓ NPE posterior built and saved successfully")
    return posterior, cfg


def sample_from_npe(posterior, cfg, x_obs, num_samples=1000, device="cpu"):
    """
    Sample from the NPE posterior given observation x_obs.
    
    Args:
        posterior: NPE posterior object
        cfg: Config dict with simulator info
        x_obs: Observed data tensor [1, num_events, x_dim]
        num_samples: Number of posterior samples to draw
        device: Device to use
        
    Returns:
        samples: Parameter samples [num_samples, param_dim]
    """
    x_obs = x_obs.to(device)
    with torch.no_grad():
        samples = posterior.sample((num_samples,), x=x_obs)
    return samples


def generate_observation(cfg, true_params, num_events, device="cpu"):
    """
    Generate observation data from simulator.
    
    Args:
        cfg: Config dict with problem type info
        true_params: True parameter values [param_dim]
        num_events: Number of events to simulate
        device: Device to use
        
    Returns:
        x_obs: Observation tensor [1, num_events, x_dim]
    """
    problem = cfg.get('problem', 'simplified_dis')
    
    # Recreate simulator from problem type
    if problem == "simplified_dis":
        from simulator import SimplifiedDIS
        simulator = SimplifiedDIS(device=torch.device("cpu"))
    elif problem in ["mceg", "mceg4dis"]:
        from simulator import MCEGSimulator
        simulator = MCEGSimulator(device=torch.device("cpu"))
    else:
        raise ValueError(f"Unknown problem: {problem}")
    
    true_params_cpu = true_params.detach().cpu()
    x_raw = simulator.sample(true_params_cpu, n_events=num_events)
    if isinstance(x_raw, torch.Tensor):
        x_tensor = x_raw.float().to(device)
    else:
        x_tensor = torch.from_numpy(x_raw).float().to(device)
    
    # Apply log transform (same as in permutation_invariant_npe.py)
    x_obs = torch.log1p(x_tensor.clamp_min(1e-8))
    
    # Ensure shape [1, num_events, x_dim]
    if x_obs.dim() == 1:
        x_obs = x_obs.unsqueeze(0)
    if x_obs.dim() == 2:
        x_obs = x_obs.unsqueeze(0)
        
    return x_obs


def plot_params_distribution_npe(
    posterior,
    cfg,
    x_obs,
    true_params,
    device="cpu",
    num_samples=1000,
    sbi_samples=None,
    sbi_labels=None,
    save_path=None,
):
    """
    Plot parameter distributions from NPE posterior and compare with SBI methods.
    
    Args:
        posterior: NPE posterior
        cfg: PermutationInvariantConfig
        x_obs: Observed data
        true_params: Ground truth parameters
        device: Device to use
        num_samples: Number of posterior samples
        sbi_samples: List of SBI sample arrays for comparison
        sbi_labels: Labels for SBI methods
        save_path: Path to save figure
    """
    # Sample from NPE
    npe_samples = sample_from_npe(posterior, cfg, x_obs, num_samples, device)
    npe_samples = npe_samples.detach().cpu().numpy()
    
    # Create figure (1x4 layout for parameter plots)
    param_dim = true_params.shape[0]
    ncols = max(4, param_dim)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    # ensure axes is a flat array
    if not isinstance(axes, (list, np.ndarray)):
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    param_names = ["$a_u$", "$b_u$", "$a_d$", "$b_d$"][:param_dim]
    
    for i in range(param_dim):
        ax = axes[i]
        
        # Plot NPE posterior
        ax.hist(npe_samples[:, i], bins=40, alpha=0.6, label="NPE", density=True)
        
        # Plot SBI methods if provided
        if sbi_samples is not None:
            for samples, label in zip(sbi_samples, sbi_labels):
                ax.hist(samples[:, i], bins=40, alpha=0.4, label=label, density=True)
        
        # Plot true parameters
        ax.axvline(true_params[i].item(), color="red", linestyle="--", linewidth=2, label="True")
        
        ax.set_xlabel(param_names[i])
        ax.set_ylabel("Density")
        ax.set_title(f"Parameter {i}: {param_names[i]}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved parameter distribution plot to {save_path}")
    return fig


def _eval_pdf_u_minus_ub_mceg(theta_arr, x_vals, q2_val=10.0):
    """
    Evaluate u(x,Q2) - ub(x,Q2) for mceg4dis problem using collaborator PDF.
    
    Args:
        theta_arr: Parameter array to evaluate
        x_vals: Array of x values to evaluate
        q2_val: Q2 value (fixed for slice)
        
    Returns:
        Array of u - ub values at x_vals
    """
    try:
        from mceg4dis.pdf import PDF
        from mceg4dis.mellin import MELLIN
        from mceg4dis.alphaS import ALPHAS
    except Exception:
        raise ImportError("Could not import mceg4dis PDF components")
    
    pdf_temp = PDF(MELLIN(npts=8), ALPHAS())
    cpar = pdf_temp.get_current_par_array()[::]
    
    arr = np.asarray(theta_arr)
    try:
        cpar[4:4+arr.shape[0]] = arr
    except Exception:
        try:
            cpar[4:8] = arr
        except Exception:
            cpar[4:4+arr.shape[0]] = arr
    
    try:
        pdf_temp.setup(cpar)
    except Exception:
        raise ValueError(f"Could not setup PDF with parameters {arr}")
    
    vals = []
    for x_val in x_vals:
        try:
            u = pdf_temp.get_xF(float(x_val), float(q2_val), "u", evolve=True)
            ub = pdf_temp.get_xF(float(x_val), float(q2_val), "ub", evolve=True)
            
            uval = float(u[0]) if hasattr(u, "__len__") else float(u)
            ubval = float(ub[0]) if hasattr(ub, "__len__") else float(ub)
            vals.append(uval - ubval)
        except Exception:
            vals.append(np.nan)
    
    return np.asarray(vals)


def plot_function_posterior_npe(
    posterior,
    cfg,
    x_obs,
    true_params,
    device="cpu",
    num_samples=100,
    problem="simplified_dis",
    save_path=None,
    q2_values=None,
):
    """
    Plot function-space posterior (PDF predictions with uncertainty bands).
    
    Args:
        posterior: NPE posterior
        cfg: PermutationInvariantConfig
        x_obs: Observed data
        true_params: Ground truth parameters
        device: Device to use
        num_samples: Number of posterior samples for uncertainty
        problem: Problem type
        save_path: Path to save figure
    """
    from simulator import SimplifiedDIS, MCEGSimulator
    
    if problem == "simplified_dis":
        sim = SimplifiedDIS(device=torch.device("cpu"))
        x_grid = torch.linspace(0.01, 0.99, 100)
    elif problem in ["mceg", "mceg4dis"]:
        sim = MCEGSimulator(device=torch.device("cpu"))
        x_grid = np.linspace(0.001, 0.99, 100)
    else:
        raise ValueError(f"Unknown problem: {problem}")
    
    # Sample posterior
    npe_samples = sample_from_npe(posterior, cfg, x_obs, num_samples, device)
    npe_samples = npe_samples.detach().cpu()
    
    # Evaluate PDF for each sample
    if problem == "simplified_dis":
        pdfs_up = []
        pdfs_down = []
        
        x_grid_tensor = x_grid.to(torch.device("cpu"))
        
        for j in range(num_samples):
            theta = npe_samples[j]
            pdf_dict = sim.f(x_grid_tensor, theta)
            pdfs_up.append(pdf_dict["up"].numpy())
            pdfs_down.append(pdf_dict["down"].numpy())
        
        pdfs_up = np.array(pdfs_up)
        pdfs_down = np.array(pdfs_down)
        
        # Compute true PDF
        true_params_cpu = true_params.detach().cpu()
        true_pdf_dict = sim.f(x_grid_tensor, true_params_cpu)
        true_pdf_up = true_pdf_dict["up"].numpy()
        true_pdf_down = true_pdf_dict["down"].numpy()
        
        x_grid_np = x_grid.numpy()
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for ax, pdfs, true_pdf, label in zip(
            axes,
            [pdfs_up, pdfs_down],
            [true_pdf_up, true_pdf_down],
            ["Up quark PDF", "Down quark PDF"],
        ):
            # Plot median and IQR
            median = np.median(pdfs, axis=0)
            p25, p75 = np.percentile(pdfs, [25, 75], axis=0)
            
            ax.plot(x_grid_np, median, "b-", linewidth=2, label="NPE median")
            ax.fill_between(x_grid_np, p25, p75, alpha=0.3, label="NPE IQR")
            ax.plot(x_grid_np, true_pdf, "r--", linewidth=2, label="True PDF")
            
            ax.set_xlabel("x")
            ax.set_ylabel("PDF value")
            ax.set_title(label)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    else:  # mceg4dis
        # Allow plotting for multiple Q^2 values
        if q2_values is None:
            q2_values = np.linspace(1.0, 10.0, 5)
        n_q2 = len(q2_values)
        fig, axes = plt.subplots(1, n_q2, figsize=(6 * n_q2, 5), squeeze=False)
        axes = axes[0]
        true_params_np = true_params.detach().cpu().numpy()
        # Use a color map for distinct colors
        import matplotlib.cm as cm
        color_map = cm.get_cmap('tab10', n_q2)
        for idx, q2_val in enumerate(q2_values):
            color = color_map(idx)
            pdfs_u_ub = []
            for j in range(num_samples):
                theta = npe_samples[j].numpy()
                try:
                    pdf_vals = _eval_pdf_u_minus_ub_mceg(theta, x_grid, q2_val=q2_val)
                    pdfs_u_ub.append(pdf_vals)
                except Exception as e:
                    print(f"⚠️  Warning: Could not evaluate PDF for sample {j} at Q²={q2_val}: {e}")
                    continue
            ax = axes[idx]
            if len(pdfs_u_ub) == 0:
                ax.text(0.5, 0.5, f"No PDFs for Q²={q2_val}", ha="center", va="center")
                continue
            pdfs_u_ub = np.array(pdfs_u_ub)
            # Evaluate true PDF
            try:
                true_pdf = _eval_pdf_u_minus_ub_mceg(true_params_np, x_grid, q2_val=q2_val)
            except Exception as e:
                print(f"⚠️  Warning: Could not evaluate true PDF at Q²={q2_val}: {e}")
                true_pdf = np.full_like(x_grid, np.nan)
            # Plot median and IQR
            median = np.median(pdfs_u_ub, axis=0)
            p25, p75 = np.percentile(pdfs_u_ub, [25, 75], axis=0)
            ax.fill_between(x_grid, p25, p75, alpha=0.3, color=color)
            ax.plot(x_grid, median, '-', linewidth=2, color=color, label="Median")
            ax.plot(x_grid, true_pdf, '--', linewidth=2, color=color, label="True")
            ax.set_xlabel("x")
            ax.set_ylabel("$u(x,Q^2) - \\bar{u}(x,Q^2)$")
            ax.set_title(f"$Q^2={q2_val:.2f}$")
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            # Only show legend for first plot
            if idx == 0:
                ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Saved function posterior plot to {save_path}")
        return fig


def plot_parameter_errors_npe(
    posterior,
    cfg,
    device="cpu",
    num_samples=100,
    problem="simplified_dis",
    save_path=None,
):
    """
    Plot parameter prediction errors across random samples.
    
    Args:
        posterior: NPE posterior
        cfg: PermutationInvariantConfig
        device: Device to use
        num_samples: Number of test samples
        problem: Problem type
        save_path: Path to save figure
    """
    errors = []
    
    for _ in range(num_samples):
        # Sample true parameters from prior
        prior = cfg.get("prior", None)
        if prior is None:
            prior = get_prior_for_problem(problem, device=device)
        true_params = prior.sample((1,))
        true_params = true_params.to(device)
        
        # Generate observation
        x_obs = generate_observation(cfg, true_params[0], cfg['num_events'], device)
        
        # Predict parameters (using posterior mean)
        posterior_samples = sample_from_npe(posterior, cfg, x_obs, 100, device)
        pred_params = posterior_samples.mean(dim=0)

        # Compute relative error: |pred - true| / max(|true|, eps)
        eps = 1e-8
        true_vec = true_params[0].to(device)
        denom = true_vec.abs().clamp_min(eps)
        error = (pred_params - true_vec).abs() / denom
        error = error.cpu().numpy()
        errors.append(error)
    
    errors = np.array(errors)
    
    # Create figure (1x4 layout for parameter error plots)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    # ensure axes is a flat array
    if not isinstance(axes, (list, np.ndarray)):
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    param_names = ["$a_u$", "$b_u$", "$a_d$", "$b_d$"]

    for i in range(4):
        ax = axes[i]
        ax.hist(errors[:, i], bins=30, alpha=0.7, edgecolor="black")
        ax.set_xlabel(f"Relative error in {param_names[i]}")
        ax.set_ylabel("Count")
        ax.set_title(f"Parameter {i} Relative Error")
        ax.grid(True, alpha=0.3)

        mean_err = np.mean(errors[:, i])
        ax.axvline(mean_err, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_err:.4f}")
        ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved parameter error plot to {save_path}")
    return fig


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate plots comparing permutation-invariant NPE with SBI methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Load sequential NPE checkpoints instead of standard NPE.",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="simplified_dis",
        help="Problem type: simplified_dis or mceg4dis",
    )
    parser.add_argument(
        "--num_events", type=int, default=10000, help="Number of events for observation"
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="Number of posterior samples"
    )
    parser.add_argument(
        "--true_params",
        type=float,
        nargs="+",
        default=None,
        help="True parameter values for plotting",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots_npe",
        help="Directory to save plots",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    
    # Create output directory
    _ensure_dir(args.output_dir)
    
    # Set default true parameters based on problem
    if args.true_params is not None:
        true_params = torch.tensor(args.true_params, dtype=torch.float32)
    else:
        if args.problem in ["mceg", "mceg4dis"]:
            true_params = torch.tensor(
                [-7.10000000e-01, 3.48000000e00, 1.34000000e00, 2.33000000],
                dtype=torch.float32,
            )
        elif args.problem == "simplified_dis":
            true_params = torch.tensor([-0.5, 1.09375, -0.5, 4.0], dtype=torch.float32)
        else:
            raise ValueError(f"Unknown problem: {args.problem}")
    
    print(f"\n🎯 PDF Parameter Inference Plotting with Permutation-Invariant NPE")
    print(f"   Problem: {args.problem}")
    print(f"   True parameters: {true_params.tolist()}")
    print(f"   Device: {device}")
    print()
    
    # Load NPE posterior (optionally sequential)
    if args.sequential:
        # Set a flag on the function to trigger sequential loading
        load_npe_posterior.use_sequential = True
    else:
        load_npe_posterior.use_sequential = False
    posterior, cfg = load_npe_posterior(args.problem, device=args.device)
    
    # Generate observation using the number of events the NPE was trained with
    num_events_for_obs = cfg['num_events']
    print(f"📊 Generating observation with {num_events_for_obs} events...")
    x_obs = generate_observation(cfg, true_params, num_events_for_obs, device)
    print(f"   Observation shape: {x_obs.shape}")
    
    # Load SBI samples for comparison (if available)
    sbi_samples = None
    sbi_labels = None
    try:
        if args.problem in ["mceg", "mceg4dis"]:
            npe_file = "samples_NPE_mceg.txt"
            wass_file = "samples_wasserstein_mceg.txt"
            mmd_file = "samples_mmd_mceg.txt"
        else:
            npe_file = "samples_npe.txt"
            wass_file = "samples_wasserstein.txt"
            mmd_file = "samples_mmd.txt"
        
        if os.path.exists(npe_file):
            samples_npe = torch.tensor(np.loadtxt(npe_file), dtype=torch.float32)
            samples_wass = torch.tensor(np.loadtxt(wass_file), dtype=torch.float32)
            samples_mmd = torch.tensor(np.loadtxt(mmd_file), dtype=torch.float32)
            sbi_samples = [samples_npe, samples_wass, samples_mmd]
            sbi_labels = ["SBI-NPE", "SBI-Wasserstein", "SBI-MCABC"]
            print(f"✓ Loaded SBI comparison samples")
    except Exception as e:
        print(f"⚠️  Could not load SBI samples: {e}")
    
    # Generate plots
    print(f"\n📈 Generating plots...")
    
    # Parameter distribution
    print(f"  • Parameter distributions...")
    plot_params_distribution_npe(
        posterior,
        cfg,
        x_obs,
        true_params,
        device=args.device,
        num_samples=args.num_samples,
        sbi_samples=sbi_samples,
        sbi_labels=sbi_labels,
        save_path=os.path.join(args.output_dir, "params_distribution.png"),
    )
    
    # Function posterior
    print(f"  • Function-space posterior...")
    # For mceg4dis, plot multiple Q^2 values
    q2_values = None
    if args.problem in ["mceg", "mceg4dis"]:
        q2_values = np.linspace(1.0, 10.0, 5)
    plot_function_posterior_npe(
        posterior,
        cfg,
        x_obs,
        true_params,
        device=args.device,
        num_samples=100,
        problem=args.problem,
        save_path=os.path.join(args.output_dir, "function_posterior.png"),
        q2_values=q2_values,
    )
    
    # Parameter errors
    print(f"  • Parameter prediction errors...")
    plot_parameter_errors_npe(
        posterior,
        cfg,
        device=args.device,
        num_samples=50,
        problem=args.problem,
        save_path=os.path.join(args.output_dir, "param_errors.png"),
    )
    
    print(f"\n✅ All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
