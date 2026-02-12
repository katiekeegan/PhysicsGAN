"""
Comparison of Permutation-Invariant NPE vs Histogram-Based NPE

This script compares two NPE approaches for PDF parameter inference:
1. Permutation-invariant NPE: Operates directly on raw event sets
2. Histogram-based NPE: Converts events to fixed histogram summaries

Key Differences:
- Permutation-invariant: Uses PermutationInvariantEmbedding, handles variable event counts
- Histogram-based: Uses histogram_summary to convert events → fixed-size vector, then standard NPE

Usage:
    python histogrammed_npe.py --problem simplified_dis --num_events 10000
"""

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sbi.inference import NPE, simulate_for_sbi
from sbi.utils.torchutils import BoxUniform

from simulator import MCEGSimulator, SimplifiedDIS

np.random.seed(42)
torch.manual_seed(42)

# Plotting setup
plt.style.use("default")
plt.rcParams.update(
    {
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "text.usetex": False,
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "axes.axisbelow": True,
    }
)


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def histogram_summary(x, nbins_per_side=16, min_abs=1e-6, max_abs=10.0, density=True):
    """
    Convert events to histogram summary statistics.
    Creates symmetric log-spaced bins for data that may be negative, zero, or positive.
    
    Args:
        x: [N, D] tensor of events
        nbins_per_side: Number of bins on each side of zero
        min_abs: Minimum absolute value for binning
        max_abs: Maximum absolute value for binning
        density: Whether to normalize histograms
        
    Returns:
        [D * (2*nbins_per_side)] histogram summary vector
    """
    x = x.detach()
    D = x.shape[1]
    device, dtype = x.device, x.dtype

    # Create log-spaced bins
    pos_edges = torch.logspace(
        torch.log10(torch.tensor(min_abs, device=device, dtype=dtype)),
        torch.log10(torch.tensor(max_abs, device=device, dtype=dtype)),
        steps=nbins_per_side + 1,
        device=device,
        dtype=dtype,
    )
    neg_edges = -torch.logspace(
        torch.log10(torch.tensor(max_abs, device=device, dtype=dtype)),
        torch.log10(torch.tensor(min_abs, device=device, dtype=dtype)),
        steps=nbins_per_side + 1,
        device=device,
        dtype=dtype,
    )
    edges = torch.cat([neg_edges, pos_edges])

    summaries = []
    for d in range(D):
        # Compute histogram for dimension d
        idx = torch.bucketize(x[:, d], edges, right=False) - 1
        B = edges.numel() - 1
        idx = idx.clamp(min=0, max=B - 1)
        counts = torch.bincount(idx, minlength=B)
        widths = edges[1:] - edges[:-1]
        
        h = counts.to(dtype)
        if density:
            total = counts.sum().clamp_min(1)
            h = h / total / widths
            Z = (h * widths).sum().clamp_min(1e-12)
            h = h / Z
        else:
            h = h / counts.sum().clamp_min(1)
        summaries.append(h)
    
    return torch.cat(summaries, dim=0)


def get_prior_for_problem(problem):
    """Get prior distribution for the problem."""
    if problem == "simplified_dis":
        return BoxUniform(
            low=torch.tensor([-1.0, 0.0, -1.0, 0.0]),
            high=torch.tensor([0.0, 5.0, 0.0, 5.0]),
        )
    elif problem in ["mceg", "mceg4dis"]:
        return BoxUniform(
            low=torch.tensor([-2.0, 0.5, -2.0, 0.5]),
            high=torch.tensor([0.0, 7.0, 3.0, 5.0]),
        )
    else:
        raise ValueError(f"Unknown problem: {problem}")


def get_simulator_for_problem(problem):
    """Get simulator for the problem."""
    if problem == "simplified_dis":
        return SimplifiedDIS(device=torch.device("cpu"))
    elif problem in ["mceg", "mceg4dis"]:
        return MCEGSimulator(device=torch.device("cpu"))
    else:
        raise ValueError(f"Unknown problem: {problem}")


def build_histogram_npe(problem, num_simulations=10000, num_events=10000, nbins=16, device="cpu"):
    """
    Build NPE using histogram summaries instead of raw events.
    
    Args:
        problem: Problem type
        num_simulations: Number of simulation runs for training
        num_events: Number of events per simulation
        nbins: Number of histogram bins per side
        device: Device to use
        
    Returns:
        posterior: Trained NPE posterior
        prior: Prior distribution
    """
    print(f"\n🔨 Building histogram-based NPE for {problem}...")
    print(f"   Simulations: {num_simulations}, Events per sim: {num_events}, Bins: {nbins}")
    
    prior = get_prior_for_problem(problem)
    simulator = get_simulator_for_problem(problem)
    
    # Define simulator that returns histogram summaries
    def sim_fn(theta_batch):
        summaries = []
        for theta in theta_batch:
            # Generate events
            x_raw = simulator.sample(theta.detach().cpu(), n_events=num_events)
            x_tensor = torch.from_numpy(x_raw).float() if not isinstance(x_raw, torch.Tensor) else x_raw.float()
            
            # Convert to histogram summary
            summary = histogram_summary(x_tensor, nbins_per_side=nbins)
            summaries.append(summary)
        
        return torch.stack(summaries)
    
    # Generate training data
    print("   Generating training data...")
    theta, x = simulate_for_sbi(sim_fn, prior, num_simulations=num_simulations)
    
    # Train NPE
    print("   Training NPE on histogram summaries...")
    inference = NPE(prior, show_progress_bars=True, device=device)
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)
    
    print("   ✓ Histogram-based NPE training complete")
    return posterior, prior


def load_or_build_histogram_npe(problem, num_simulations=10000, num_events=10000, nbins=16, device="cpu"):
    """Load histogram NPE from cache or build if not found."""
    save_dir = f"npe_checkpoints/{problem}"
    posterior_path = os.path.join(save_dir, "histogram_npe_posterior.pkl")
    prior_path = os.path.join(save_dir, "histogram_npe_prior.pkl")
    
    if os.path.exists(posterior_path) and os.path.exists(prior_path):
        print(f"🔍 Loading histogram NPE from {posterior_path}...")
        try:
            with open(posterior_path, "rb") as f:
                posterior = pickle.load(f)
            with open(prior_path, "rb") as f:
                prior = pickle.load(f)
            print("✓ Histogram NPE loaded from checkpoint")
            return posterior, prior
        except Exception as e:
            print(f"⚠️  Could not load checkpoint: {e}. Building new...")
    
    # Build and save
    posterior, prior = build_histogram_npe(problem, num_simulations, num_events, nbins, device)
    
    _ensure_dir(save_dir)
    with open(posterior_path, "wb") as f:
        pickle.dump(posterior, f)
    with open(prior_path, "wb") as f:
        pickle.dump(prior, f)
    print(f"✓ Saved histogram NPE to {save_dir}")
    
    return posterior, prior


def load_permutation_invariant_npe(problem, device="cpu"):
    """Load permutation-invariant NPE from saved checkpoint."""
    save_dir = f"npe_checkpoints/{problem}"
    posterior_path = os.path.join(save_dir, "posterior.pkl")
    config_path = os.path.join(save_dir, "config.pkl")
    
    if not os.path.exists(posterior_path) or not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Permutation-invariant NPE not found. Run plotting_driver.py first to generate it."
        )
    
    print(f"🔍 Loading permutation-invariant NPE from {posterior_path}...")
    with open(posterior_path, "rb") as f:
        posterior = pickle.load(f)
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    print("✓ Permutation-invariant NPE loaded")
    
    return posterior, cfg


def generate_observation(problem, true_params, num_events, nbins=16, device="cpu"):
    """
    Generate observation for both NPE types.
    
    Returns:
        x_raw: Raw events for permutation-invariant NPE [1, num_events, x_dim]
        x_hist: Histogram summary for histogram-based NPE [summary_dim]
    """
    simulator = get_simulator_for_problem(problem)
    
    true_params_cpu = true_params.detach().cpu()
    x_raw_np = simulator.sample(true_params_cpu, n_events=num_events)
    
    if isinstance(x_raw_np, torch.Tensor):
        x_tensor = x_raw_np.float().to(device)
    else:
        x_tensor = torch.from_numpy(x_raw_np).float().to(device)
    
    # For permutation-invariant NPE: apply log transform and reshape
    x_raw = torch.log1p(x_tensor.clamp_min(1e-8))
    if x_raw.dim() == 2:
        x_raw = x_raw.unsqueeze(0)
    
    # For histogram NPE: create histogram summary
    x_hist = histogram_summary(x_tensor, nbins_per_side=nbins)
    
    return x_raw, x_hist


def compare_parameter_distributions(
    pi_samples, hist_samples, true_params, problem="simplified_dis", save_path=None
):
    """
    Compare parameter distributions from both NPE approaches.
    
    Args:
        pi_samples: Samples from permutation-invariant NPE [N, param_dim]
        hist_samples: Samples from histogram-based NPE [N, param_dim]
        true_params: Ground truth parameters
        problem: Problem type
        save_path: Path to save figure
    """
    param_dim = true_params.shape[0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    param_names = ["$a_u$", "$b_u$", "$a_d$", "$b_d$"][:param_dim]
    
    for i in range(param_dim):
        ax = axes[i]
        
        # Plot both posteriors
        ax.hist(
            pi_samples[:, i],
            bins=40,
            alpha=0.5,
            label="Permutation-Invariant NPE",
            density=True,
            color="blue",
        )
        ax.hist(
            hist_samples[:, i],
            bins=40,
            alpha=0.5,
            label="Histogram-Based NPE",
            density=True,
            color="orange",
        )
        
        # Plot true parameters
        ax.axvline(
            true_params[i].item(),
            color="red",
            linestyle="--",
            linewidth=2,
            label="True",
        )
        
        ax.set_xlabel(param_names[i])
        ax.set_ylabel("Density")
        ax.set_title(f"Parameter {i}: {param_names[i]}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved parameter comparison to {save_path}")
    return fig


def compare_function_posteriors(
    pi_samples, hist_samples, true_params, problem="simplified_dis", device="cpu", save_path=None
):
    """
    Compare PDF function posteriors from both NPE approaches.
    """
    if problem == "simplified_dis":
        from simulator import SimplifiedDIS
        sim = SimplifiedDIS(device=torch.device("cpu"))
        x_grid = torch.linspace(0.01, 0.99, 100)
        
        # Evaluate PDFs for permutation-invariant samples
        pi_pdfs_up, pi_pdfs_down = [], []
        for j in range(min(100, pi_samples.shape[0])):
            theta = pi_samples[j]
            pdf_dict = sim.f(x_grid, theta)
            pi_pdfs_up.append(pdf_dict["up"].numpy())
            pi_pdfs_down.append(pdf_dict["down"].numpy())
        
        # Evaluate PDFs for histogram samples
        hist_pdfs_up, hist_pdfs_down = [], []
        for j in range(min(100, hist_samples.shape[0])):
            theta = torch.tensor(hist_samples[j], dtype=torch.float32)
            pdf_dict = sim.f(x_grid, theta)
            hist_pdfs_up.append(pdf_dict["up"].numpy())
            hist_pdfs_down.append(pdf_dict["down"].numpy())
        
        pi_pdfs_up = np.array(pi_pdfs_up)
        pi_pdfs_down = np.array(pi_pdfs_down)
        hist_pdfs_up = np.array(hist_pdfs_up)
        hist_pdfs_down = np.array(hist_pdfs_down)
        
        # Compute true PDF
        true_params_cpu = true_params.detach().cpu()
        true_pdf_dict = sim.f(x_grid, true_params_cpu)
        true_pdf_up = true_pdf_dict["up"].numpy()
        true_pdf_down = true_pdf_dict["down"].numpy()
        
        x_grid_np = x_grid.numpy()
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for ax, pi_pdfs, hist_pdfs, true_pdf, label in zip(
            axes,
            [pi_pdfs_up, pi_pdfs_down],
            [hist_pdfs_up, hist_pdfs_down],
            [true_pdf_up, true_pdf_down],
            ["Up quark PDF", "Down quark PDF"],
        ):
            # Permutation-invariant
            pi_median = np.median(pi_pdfs, axis=0)
            pi_p25, pi_p75 = np.percentile(pi_pdfs, [25, 75], axis=0)
            
            # Histogram-based
            hist_median = np.median(hist_pdfs, axis=0)
            hist_p25, hist_p75 = np.percentile(hist_pdfs, [25, 75], axis=0)
            
            # Plot
            ax.plot(x_grid_np, pi_median, "b-", linewidth=2, label="Perm-Inv median")
            ax.fill_between(x_grid_np, pi_p25, pi_p75, alpha=0.3, color="blue", label="Perm-Inv IQR")
            
            ax.plot(x_grid_np, hist_median, color="orange", linewidth=2, label="Hist median")
            ax.fill_between(x_grid_np, hist_p25, hist_p75, alpha=0.3, color="orange", label="Hist IQR")
            
            ax.plot(x_grid_np, true_pdf, "r--", linewidth=2, label="True PDF")
            
            ax.set_xlabel("x")
            ax.set_ylabel("PDF value")
            ax.set_title(label)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    else:  # mceg4dis
        print("⚠️  Function posterior comparison for mceg4dis not yet implemented")
        return None
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved function comparison to {save_path}")
    return fig


def compute_metrics(samples, true_params):
    """Compute summary metrics for posterior samples."""
    median = np.median(samples, axis=0)
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    
    mae = np.abs(median - true_params.cpu().numpy()).mean()
    
    return {
        "median": median,
        "mean": mean,
        "std": std,
        "mae": mae,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare Permutation-Invariant NPE vs Histogram-Based NPE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="simplified_dis",
        help="Problem type: simplified_dis or mceg4dis",
    )
    parser.add_argument(
        "--num_events",
        type=int,
        default=10000,
        help="Number of events for observation and histogram NPE training",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=10000,
        help="Number of simulations for histogram NPE training",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of posterior samples to draw",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=16,
        help="Number of histogram bins per side",
    )
    parser.add_argument(
        "--true_params",
        type=float,
        nargs="+",
        default=None,
        help="True parameter values",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots_npe_comparison",
        help="Directory to save plots",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    _ensure_dir(args.output_dir)
    
    # Set default true parameters
    if args.true_params is not None:
        true_params = torch.tensor(args.true_params, dtype=torch.float32)
    else:
        if args.problem in ["mceg", "mceg4dis"]:
            true_params = torch.tensor(
                [-0.71, 3.48, 1.34, 2.33], dtype=torch.float32
            )
        elif args.problem == "simplified_dis":
            true_params = torch.tensor([-0.5, 1.09375, -0.5, 4.0], dtype=torch.float32)
        else:
            raise ValueError(f"Unknown problem: {args.problem}")
    
    print(f"\n🎯 NPE Comparison: Permutation-Invariant vs Histogram-Based")
    print(f"   Problem: {args.problem}")
    print(f"   True parameters: {true_params.tolist()}")
    print(f"   Device: {device}")
    print()
    
    # Load permutation-invariant NPE
    pi_posterior, pi_cfg = load_permutation_invariant_npe(args.problem, device=args.device)
    
    # Load or build histogram-based NPE
    hist_posterior, hist_prior = load_or_build_histogram_npe(
        args.problem,
        num_simulations=args.num_simulations,
        num_events=args.num_events,
        nbins=args.nbins,
        device=args.device,
    )
    
    # Generate observation
    print(f"\n📊 Generating observation with {args.num_events} events...")
    x_raw, x_hist = generate_observation(
        args.problem, true_params, args.num_events, nbins=args.nbins, device=device
    )
    print(f"   Raw events shape: {x_raw.shape}")
    print(f"   Histogram summary shape: {x_hist.shape}")
    
    # Sample from both posteriors
    print(f"\n🎲 Sampling {args.num_samples} from each posterior...")
    
    with torch.no_grad():
        pi_samples = pi_posterior.sample((args.num_samples,), x=x_raw.to(device))
        pi_samples = pi_samples.detach().cpu().numpy()
    
    with torch.no_grad():
        hist_samples = hist_posterior.sample((args.num_samples,), x=x_hist.unsqueeze(0).to(device))
        hist_samples = hist_samples.detach().cpu().numpy()
    
    print("✓ Sampling complete")
    
    # Compute metrics
    print(f"\n📈 Computing metrics...")
    pi_metrics = compute_metrics(pi_samples, true_params)
    hist_metrics = compute_metrics(hist_samples, true_params)
    
    print("\nPermutation-Invariant NPE:")
    print(f"  MAE: {pi_metrics['mae']:.4f}")
    print(f"  Std: {pi_metrics['std']}")
    
    print("\nHistogram-Based NPE:")
    print(f"  MAE: {hist_metrics['mae']:.4f}")
    print(f"  Std: {hist_metrics['std']}")
    
    # Generate comparison plots
    print(f"\n📊 Generating comparison plots...")
    
    compare_parameter_distributions(
        pi_samples,
        hist_samples,
        true_params,
        problem=args.problem,
        save_path=os.path.join(args.output_dir, "parameter_comparison.png"),
    )
    
    if args.problem == "simplified_dis":
        compare_function_posteriors(
            pi_samples,
            hist_samples,
            true_params,
            problem=args.problem,
            device=device,
            save_path=os.path.join(args.output_dir, "function_comparison.png"),
        )
    
    print(f"\n✅ All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
