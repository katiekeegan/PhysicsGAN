"""
Sequential Permutation-Invariant NPE / SNPE-C / SNPE-A for DIS simulators (GPU-friendly, chunked).

Methods (CLI: --method):
- npe:    single-round (prior-only) training of q(theta|x) (i.e., non-sequential).
- snpe_c: sequential SNPE-C (SBI class: SNPE). Requires proposal correction via append_simulations(..., proposal=...).
- snpe_a: sequential SNPE-A (SBI class: SNPE_A). Also uses proposal information.

IMPORTANT:
Proper sequential SNPE requires a fixed observed dataset x_o.
Provide it via --x_obs_path (a torch .pt file containing a tensor shaped (num_events, x_dim) or (1, num_events, x_dim)).
If not provided, the script falls back to a synthetic reference observation for sampling, which is NOT proper SNPE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple, Optional
import os
import pickle

import torch

# SBI imports
from sbi.inference import SNPE
try:
    from sbi.inference import SNPE_A
except Exception:
    SNPE_A = None  # older installs may not ship it; in 0.25.0 it should exist.

from sbi.neural_nets import posterior_nn
from sbi.neural_nets.embedding_nets import FCEmbedding, PermutationInvariantEmbedding
from sbi.utils.torchutils import BoxUniform

from simulator import MCEGSimulator, SimplifiedDIS

def _default_true_params(problem: str) -> torch.Tensor:
    if problem in {"mceg", "mceg4dis"}:
        return torch.tensor([-7.10000000e-01, 3.48000000e00, 1.34000000e00, 2.33000000], dtype=torch.float32)
    if problem == "simplified_dis":
        return torch.tensor([-0.5, 1.09375, -0.5, 4.0], dtype=torch.float32)
    raise ValueError(f"Unknown problem: {problem}")


@torch.no_grad()
def _make_x_obs_from_true_params(
    simulator_fn: Callable[[torch.Tensor, int], torch.Tensor],
    true_params: torch.Tensor,
    *,
    num_events: int,
    simulation_device: torch.device,
    training_device: torch.device,
) -> torch.Tensor:
    # simulator_fn expects a single theta vector (theta_dim,)
    theta_star = true_params.to(simulation_device)
    x_o = simulator_fn(theta_star, n_events=num_events)  # (num_events, x_dim)
    return x_o.unsqueeze(0).to(training_device)          # (1, num_events, x_dim)


@dataclass
class SequentialPermutationInvariantConfig:
    posterior: object
    prior: BoxUniform
    num_events: int
    x_dim: int
    training_device: torch.device
    simulation_device: torch.device
    num_rounds: int
    simulations_per_round: int
    method: str


def _log_transform(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(x.clamp_min(1e-8))


def _simulate_simplified_dis(simulator, theta: torch.Tensor, n_events: int) -> torch.Tensor:
    raw = simulator.sample(theta, n_events=n_events)
    return _log_transform(raw)


def _simulate_mceg(simulator, theta: torch.Tensor, n_events: int) -> torch.Tensor:
    raw = simulator.sample(theta, n_events=n_events)
    return _log_transform(raw)


def _get_simulator_and_prior(
    problem: str,
    simulation_device: torch.device,
    prior_device: torch.device,
) -> Tuple[Callable[[torch.Tensor, int], torch.Tensor], BoxUniform, int]:
    if problem == "simplified_dis":
        sim = SimplifiedDIS(device=simulation_device)
        prior = BoxUniform(
            low=torch.tensor([-1.0, 0.0, -1.0, 0.0], device=prior_device),
            high=torch.tensor([0.0, 5.0, 0.0, 5.0], device=prior_device),
        )
        x_dim = 2
        simulate_fn = lambda theta, n_events: _simulate_simplified_dis(sim, theta, n_events)
        return simulate_fn, prior, x_dim

    if problem in {"mceg", "mceg4dis"}:
        sim = MCEGSimulator(device=simulation_device)
        prior = BoxUniform(
            low=torch.tensor([-1.0, 0.0, -10.0, -10.0], device=prior_device),
            high=torch.tensor([10.0, 10.0, 10.0, 10.0], device=prior_device),
        )
        x_dim = 2
        simulate_fn = lambda theta, n_events: _simulate_mceg(sim, theta, n_events)
        return simulate_fn, prior, x_dim

    raise ValueError(f"Unsupported problem '{problem}'. Choose 'simplified_dis' or 'mceg4dis'.")


def _build_embedding(x_dim: int, latent_dim: int = 32) -> PermutationInvariantEmbedding:
    trial_net = FCEmbedding(
        input_dim=x_dim,
        num_hiddens=64,
        num_layers=2,
        output_dim=latent_dim,
    )
    return PermutationInvariantEmbedding(
        trial_net=trial_net,
        trial_net_output_dim=latent_dim,
        aggregation_fn="sum",
        num_layers=1,
        num_hiddens=64,
        output_dim=latent_dim,
        aggregation_dim=1,
    )


def _load_x_obs(x_obs_path: Optional[str], *, training_device: torch.device) -> Optional[torch.Tensor]:
    if not x_obs_path:
        return None
    x = torch.load(x_obs_path, map_location="cpu")
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"--x_obs_path must load a torch.Tensor, got {type(x)}")
    if x.ndim == 2:
        x = x.unsqueeze(0)  # (1, num_events, x_dim)
    if x.ndim != 3:
        raise ValueError(f"x_o must have shape (num_events, x_dim) or (1, num_events, x_dim). Got {tuple(x.shape)}")
    return x.to(training_device)


@torch.no_grad()
def _append_simulations_from_prior(
    inference,
    simulator: Callable[[torch.Tensor, int], torch.Tensor],
    prior: BoxUniform,
    *,
    num_simulations: int,
    num_events: int,
    x_dim: int,
    simulation_device: torch.device,
    storage_device: torch.device,
    chunk_size: int,
    show_progress_bars: bool,
) -> None:
    n_done = 0
    while n_done < num_simulations:
        n_cur = min(chunk_size, num_simulations - n_done)

        theta = prior.sample((n_cur,))
        theta_sim = theta.to(simulation_device, non_blocking=True)

        xs = torch.empty((n_cur, num_events, x_dim), device=simulation_device)
        for i in range(n_cur):
            xs[i] = simulator(theta_sim[i], n_events=num_events)

        inference.append_simulations(
            theta.to(storage_device),
            xs.to(storage_device),
            exclude_invalid_x=True,
        )

        n_done += n_cur
        if show_progress_bars:
            print(f"  Appended {n_done}/{num_simulations} simulations from prior", end="\r")
    if show_progress_bars:
        print()


@torch.no_grad()
def _append_simulations_from_proposal(
    inference,
    simulator: Callable[[torch.Tensor, int], torch.Tensor],
    *,
    proposal,
    num_simulations: int,
    num_events: int,
    x_dim: int,
    simulation_device: torch.device,
    storage_device: torch.device,
    chunk_size: int,
    show_progress_bars: bool,
) -> None:
    if getattr(proposal, "default_x", None) is None:
        raise ValueError("proposal.default_x is None. Call proposal.set_default_x(x_o) before using it as a proposal.")

    n_done = 0
    while n_done < num_simulations:
        n_cur = min(chunk_size, num_simulations - n_done)

        theta = proposal.sample((n_cur,))  # uses proposal.default_x

        theta_sim = theta.to(simulation_device, non_blocking=True)

        xs = torch.empty((n_cur, num_events, x_dim), device=simulation_device)
        for i in range(n_cur):
            xs[i] = simulator(theta_sim[i], n_events=num_events)

        inference.append_simulations(
            theta.to(storage_device),
            xs.to(storage_device),
            proposal=proposal,  # proposal correction
            exclude_invalid_x=True,
        )

        n_done += n_cur
        if show_progress_bars:
            print(f"  Appended {n_done}/{num_simulations} simulations from proposal", end="\r")
    if show_progress_bars:
        print()


def build_sequential_permutation_invariant_posterior(
    *,
    problem: str,
    method: str = "snpe_c",
    x_obs_path: Optional[str] = None,
    true_params: Optional[torch.Tensor] = None,   # <-- NEW
    num_simulations_per_round: int = 1000,
    num_rounds: int = 3,
    num_events: int = 256,
    latent_dim: int = 32,
    training_device: Optional[torch.device] = None,
    simulation_device: Optional[torch.device] = None,
    storage_device: Optional[torch.device] = None,
    training_batch_size: int = 256,
    chunk_size: int = 1024,
    show_progress_bars: bool = False,
) -> Tuple[object, SequentialPermutationInvariantConfig]:

    method = method.lower()
    if method not in {"npe", "snpe_c", "snpe_a"}:
        raise ValueError(f"--method must be one of: npe, snpe_c, snpe_a. Got: {method}")

    if method == "snpe_a" and SNPE_A is None:
        raise RuntimeError("SNPE_A is not available in this sbi install. Upgrade sbi or use --method snpe_c.")

    if training_device is None:
        training_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if simulation_device is None:
        simulation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if storage_device is None:
        storage_device = torch.device("cpu")

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    simulator_fn, prior, x_dim = _get_simulator_and_prior(
        problem,
        simulation_device=simulation_device,
        prior_device=training_device,
    )

    embedding_net = _build_embedding(x_dim=x_dim, latent_dim=latent_dim)

    density_estimator = posterior_nn(
        "mdn",
        embedding_net=embedding_net,
        z_score_x="none",
        z_score_theta="independent",
    )

    InferenceCls = SNPE if method in {"npe", "snpe_c"} else SNPE_A
    inference = InferenceCls(
        prior=prior,
        density_estimator=density_estimator,
        show_progress_bars=show_progress_bars,
        device=training_device,
    )

        # Fixed single observation x_o for proper sequential SNPE
    x_o = _load_x_obs(x_obs_path, training_device=training_device)

    if method != "npe":
        if x_o is None:
            if true_params is None:
                true_params = _default_true_params(problem)
            x_o = _make_x_obs_from_true_params(
                simulator_fn,
                true_params=true_params,
                num_events=num_events,
                simulation_device=simulation_device,
                training_device=training_device,
            )
            if show_progress_bars:
                print(f"[info] Built x_o from true_params={true_params.tolist()} with shape {tuple(x_o.shape)}")

    if method == "npe":
        num_rounds_effective = 1
    else:
        num_rounds_effective = num_rounds

    posterior = None

    for round_num in range(num_rounds_effective):
        print(f"\n🔄 Round {round_num + 1}/{num_rounds_effective}   (method={method})")

        if round_num == 0:
            print("   Sampling from prior...")
            _append_simulations_from_prior(
                inference,
                simulator_fn,
                prior,
                num_simulations=num_simulations_per_round,
                num_events=num_events,
                x_dim=x_dim,
                simulation_device=simulation_device,
                storage_device=storage_device,
                chunk_size=chunk_size,
                show_progress_bars=show_progress_bars,
            )
        else:
            if x_o is None:
                raise ValueError(
                    "Sequential SNPE requires a fixed observation x_o. "
                    "Provide --x_obs_path or use --true_params to generate x_o."
                )

            proposal = posterior
            if proposal is None:
                raise RuntimeError("Internal error: posterior is None at round>0.")

            print("   Sampling from proposal (previous posterior, conditioned on x_o)...")
            _append_simulations_from_proposal(
                inference,
                simulator_fn,
                proposal=proposal,
                num_simulations=num_simulations_per_round,
                num_events=num_events,
                x_dim=x_dim,
                simulation_device=simulation_device,
                storage_device=storage_device,
                chunk_size=chunk_size,
                show_progress_bars=show_progress_bars,
            )

        print("   Training density estimator...")
        inference.train(training_batch_size=training_batch_size, max_num_epochs=100)
        posterior = inference.build_posterior()

        if method != "npe":
            # Critical: set default_x so posterior can be used as proposal next round
            posterior = posterior.set_default_x(x_o)

        total_sims = num_simulations_per_round * (round_num + 1)
        print(f"   ✓ Round {round_num + 1} complete ({total_sims} total sims)")

    cfg = SequentialPermutationInvariantConfig(
        posterior=posterior,
        prior=prior,
        num_events=num_events,
        x_dim=x_dim,
        training_device=training_device,
        simulation_device=simulation_device,
        num_rounds=num_rounds_effective,
        simulations_per_round=num_simulations_per_round,
        method=method,
    )

    # Save posterior and config
    save_dir = f"npe_checkpoints/{problem}"
    os.makedirs(save_dir, exist_ok=True)

    posterior_path = os.path.join(save_dir, f"{method}_posterior.pkl")
    config_path = os.path.join(save_dir, f"{method}_config.pkl")

    with open(posterior_path, "wb") as f:
        pickle.dump(posterior, f)
    with open(config_path, "wb") as f:
        pickle.dump(
            {
                "method": method,
                "problem": problem,
                "posterior_path": posterior_path,
                "num_events": num_events,
                "x_dim": x_dim,
                "num_rounds": num_rounds_effective,
                "simulations_per_round": num_simulations_per_round,
                "training_device": str(training_device),
                "simulation_device": str(simulation_device),
                "x_obs_path": x_obs_path,
            },
            f,
        )

    print(f"\n✓ Saved posterior to {posterior_path}")
    print(f"✓ Saved config to {config_path}")
    return posterior, cfg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sequential Permutation-Invariant NPE/SNPE-C/SNPE-A for DIS simulators (GPU-friendly)"
    )
    parser.add_argument("--problem", choices=["simplified_dis", "mceg4dis"], default="simplified_dis")
    parser.add_argument(
        "--method",
        choices=["npe", "snpe_c", "snpe_a"],
        default="snpe_c",
        help="npe=single-round prior-only; snpe_c=SNPE (proposal-corrected); snpe_a=SNPE_A.",
    )
    parser.add_argument(
        "--x_obs_path",
        type=str,
        default=None,
        help="Path to torch-saved observed x_o tensor for proper sequential SNPE. Shape (num_events, x_dim) or (1, num_events, x_dim).",
    )
    parser.add_argument("--num_simulations_per_round", type=int, default=200)
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--num_events", type=int, default=10000)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--training_batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--cpu_sim", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    training_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simulation_device = torch.device("cpu") if args.cpu_sim else training_device

    posterior, cfg = build_sequential_permutation_invariant_posterior(
        problem=args.problem,
        method=args.method,
        x_obs_path=args.x_obs_path,
        num_simulations_per_round=args.num_simulations_per_round,
        num_rounds=args.num_rounds,
        num_events=args.num_events,
        latent_dim=args.latent_dim,
        training_batch_size=args.training_batch_size,
        chunk_size=args.chunk_size,
        training_device=training_device,
        simulation_device=simulation_device,
        storage_device=torch.device("cpu"),
        show_progress_bars=not args.no_progress,
    )

    print("\nFinished.")
    print(f"Method: {cfg.method}")
    print(f"Total sims: {args.num_simulations_per_round * cfg.num_rounds}")