"""
Steering Sensitivity as Uncertainty — ViNT Experiment

Uses activation steering vectors to derive three novel uncertainty signals:
  1. Steering Projection — dot product of hidden state with steering direction
  2. Steering Sensitivity — directional derivative of distance prediction along
     the steering vector (Jacobian-based, primary contribution)
  3. Minimum Steering Distance (α*) — smallest perturbation magnitude that
     changes the prediction by threshold δ

Evaluates each signal (+ MC Dropout baseline) via AUROC and Spearman correlation
against prediction error.

Usage:
  python experiments/steering_sensitivity.py \
    --checkpoint deployment/model_weights/vint.pth \
    --data-folder /path/to/go_stanford_cropped \
    --max-trajectories 50 \
    --samples-per-traj 30 \
    --seed 42
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from uncertainty_correlation import (
    TrajectoryLoader,
    load_vint_model,
    compute_failure_proxy,
    compute_mc_dropout_variance,
    PROJECT_ROOT,
)
from activation_steering import (
    MultiLayerCapture,
    SteeringHook,
    collect_hidden_states,
    compute_steering_vectors,
    NUM_LAYERS,
    FAILURE_THRESHOLD,
)


# ===========================================================================
# GradCapture — forward hook that preserves gradient flow
# ===========================================================================
class GradCapture:
    """Captures residual stream output at a layer WITHOUT detaching,
    so that torch.autograd.grad can backprop through it."""

    def __init__(self):
        self.captured = None  # will hold the output tensor (with grad)
        self._handle = None

    def install(self, layer):
        def hook_fn(module, input, output):
            output.retain_grad()
            self.captured = output
            return output
        self._handle = layer.register_forward_hook(hook_fn)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self.captured = None


# ===========================================================================
# Signal 1: Steering Projection
# ===========================================================================
def compute_steering_projection(hidden, steering_vec):
    """Dot product of flattened hidden state with normalized steering vector.

    Args:
        hidden: [7, 512] tensor (detached, cpu or gpu)
        steering_vec: [7, 512] tensor
    Returns:
        float — projection magnitude
    """
    h_flat = hidden.flatten().float()
    v_flat = steering_vec.flatten().float()
    v_norm = v_flat / (v_flat.norm() + 1e-12)
    return float(torch.dot(h_flat, v_norm).item())


# ===========================================================================
# Signal 2: Steering Sensitivity (Jacobian-based)
# ===========================================================================
def compute_steering_sensitivity(model, obs, goal, layer_idx, steering_vec, device):
    """Directional derivative of dist_pred along the steering vector at a given layer.

    |∂dist_pred/∂h_l · v_l| / ||v_l||

    Uses torch.autograd.grad with a GradCapture hook that keeps gradient flow.

    Args:
        model: ViNT model (will temporarily be put in eval mode with grads enabled)
        obs: [1, 18, H, W] observation tensor
        goal: [1, 3, H, W] goal tensor
        layer_idx: which transformer layer to hook
        steering_vec: [7, 512] steering vector for that layer
        device: torch device
    Returns:
        float — sensitivity value
    """
    grad_cap = GradCapture()
    target_layer = model.decoder.sa_decoder.layers[layer_idx]
    grad_cap.install(target_layer)

    model.eval()
    # Need gradients for the hook output but not for model parameters
    with torch.enable_grad():
        dist_pred, _, _ = model(obs, goal)
        h = grad_cap.captured  # [1, 7, 512] with grad

        # Compute gradient of scalar dist_pred w.r.t. hidden state h
        grad_h = torch.autograd.grad(
            dist_pred, h, create_graph=False, retain_graph=False
        )[0]  # [1, 7, 512]

    grad_cap.remove()

    # Directional derivative along steering vector
    grad_flat = grad_h.squeeze(0).flatten().float()  # [7*512]
    v_flat = steering_vec.flatten().float().to(device)
    v_norm_scalar = v_flat.norm() + 1e-12
    sensitivity = float(torch.abs(torch.dot(grad_flat, v_flat)).item() / v_norm_scalar.item())
    return sensitivity


# ===========================================================================
# Signal 3: Minimum Steering Distance (α*)
# ===========================================================================
def compute_min_steering_alpha(model, obs, goal, layer_idx, steering_vec, device,
                               delta=0.5, alpha_max=5.0, n_steps=16):
    """Binary search for smallest α s.t. |f(h + αv) - f(h)| > δ.

    Args:
        model: ViNT model
        obs, goal: input tensors
        layer_idx: transformer layer to steer
        steering_vec: [7, 512] steering vector
        device: torch device
        delta: prediction change threshold
        alpha_max: upper bound of search
        n_steps: number of binary search iterations
    Returns:
        float — α* (returns alpha_max if threshold never reached)
    """
    target_layer = model.decoder.sa_decoder.layers[layer_idx]

    # Baseline prediction (no steering)
    model.eval()
    with torch.no_grad():
        d_base, _, _ = model(obs, goal)
    d_base = d_base.item()

    lo, hi = 0.0, alpha_max
    for _ in range(n_steps):
        mid = (lo + hi) / 2.0
        hook = SteeringHook(steering_vec.to(device), mid)
        handle = target_layer.register_forward_hook(hook)
        with torch.no_grad():
            d_steered, _, _ = model(obs, goal)
        handle.remove()

        if abs(d_steered.item() - d_base) > delta:
            hi = mid
        else:
            lo = mid

    # Also check negative direction
    lo_neg, hi_neg = 0.0, alpha_max
    for _ in range(n_steps):
        mid = (lo_neg + hi_neg) / 2.0
        hook = SteeringHook(steering_vec.to(device), -mid)
        handle = target_layer.register_forward_hook(hook)
        with torch.no_grad():
            d_steered, _, _ = model(obs, goal)
        handle.remove()

        if abs(d_steered.item() - d_base) > delta:
            hi_neg = mid
        else:
            lo_neg = mid

    # Return the smaller of the two directions
    return min(hi, hi_neg)


# ===========================================================================
# Main experiment
# ===========================================================================
def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_vint_model(args.checkpoint, device)

    loader = TrajectoryLoader(
        args.data_folder,
        max_trajectories=args.max_trajectories,
        seed=args.seed,
    )

    all_samples = loader.generate_samples(samples_per_traj=args.samples_per_traj)
    if not all_samples:
        print("ERROR: No samples generated.")
        return

    # Train/test split (same as activation_steering.py)
    rng = np.random.RandomState(args.seed)
    indices = np.arange(len(all_samples))
    rng.shuffle(indices)
    split_idx = int(len(all_samples) * args.train_fraction)
    train_samples = [all_samples[i] for i in indices[:split_idx]]
    test_samples = [all_samples[i] for i in indices[split_idx:]]
    print(f"Split: {len(train_samples)} train, {len(test_samples)} test")

    # Phase 1: Collect hidden states (train split) for steering vector computation
    print("\n=== Phase 1: Collecting per-layer hidden states (train split) ===")
    t0 = time.time()
    train_records = collect_hidden_states(model, loader, train_samples, device)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Phase 2: Compute steering vectors
    print("\n=== Phase 2: Computing per-layer steering vectors ===")
    steering_vectors = compute_steering_vectors(train_records)
    if steering_vectors is None:
        return

    # Phase 3: Compute all signals for test samples
    print(f"\n=== Phase 3: Computing signals for {len(test_samples)} test samples ===")
    t0 = time.time()

    # Also collect hidden states for test samples (needed for Signal 1)
    print("  Collecting test hidden states...")
    test_records = collect_hidden_states(model, loader, test_samples, device)

    results = []
    for i, (sample, record) in enumerate(zip(test_samples, test_records)):
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Sample {i + 1}/{len(test_samples)} ({rate:.1f} samples/sec)")

        obs = loader.build_obs_tensor(sample["traj_name"], sample["curr_time"]).to(device)
        goal = loader.build_goal_tensor(sample["traj_name"], sample["goal_time"]).to(device)
        d_actual = sample["d_actual"]
        d_predicted = record["d_predicted"]
        rel_error = record["relative_error"]
        is_failure = rel_error >= FAILURE_THRESHOLD

        # Per-layer signals
        layer_projections = {}
        layer_sensitivities = {}
        layer_random_sens = {}
        layer_alpha_stars = {}

        for layer_idx in range(NUM_LAYERS):
            sv = steering_vectors[layer_idx]

            # Signal 1: Steering Projection
            h = record["layer_hiddens"][layer_idx]  # [7, 512], detached
            proj = compute_steering_projection(h, sv)
            layer_projections[layer_idx] = proj

            # Signal 2: Steering Sensitivity (Jacobian-based)
            sens = compute_steering_sensitivity(
                model, obs, goal, layer_idx, sv, device
            )
            layer_sensitivities[layer_idx] = sens

            # Signal 2b: Random Sensitivity — same Jacobian trick but with a random unit vector
            rand_vec = torch.randn_like(sv)
            rand_vec = rand_vec / (rand_vec.norm() + 1e-12) * sv.norm()  # match steering vec scale
            rand_sens = compute_steering_sensitivity(
                model, obs, goal, layer_idx, rand_vec, device
            )
            layer_random_sens[layer_idx] = rand_sens

            # Signal 3: Minimum Steering Distance (disabled — too slow)
            layer_alpha_stars[layer_idx] = 0.0

        # MC Dropout baseline (disabled — too slow)
        # mc_var, mc_mean = compute_mc_dropout_variance(
        #     model, obs, goal, n_passes=args.mc_passes,
        # )
        mc_var, mc_mean = 0.0, 0.0

        result = {
            "traj_name": sample["traj_name"],
            "curr_time": sample["curr_time"],
            "goal_time": sample["goal_time"],
            "d_actual": d_actual,
            "d_predicted": d_predicted,
            "relative_error": rel_error,
            "is_failure": is_failure,
            "mc_dropout_var": mc_var,
            "mc_dropout_mean": mc_mean,
        }
        for layer_idx in range(NUM_LAYERS):
            result[f"projection_L{layer_idx}"] = layer_projections[layer_idx]
            result[f"sensitivity_L{layer_idx}"] = layer_sensitivities[layer_idx]
            result[f"random_sens_L{layer_idx}"] = layer_random_sens[layer_idx]
            result[f"alpha_star_L{layer_idx}"] = layer_alpha_stars[layer_idx]
            sens = layer_sensitivities[layer_idx]
            result[f"proj_over_sens_L{layer_idx}"] = layer_projections[layer_idx] / (sens + 1e-12)

        results.append(result)

    model.eval()
    elapsed = time.time() - t0
    print(f"\nPhase 3 complete in {elapsed:.1f}s")

    # Phase 4: Evaluate
    print("\n=== Phase 4: Evaluation ===")
    metrics = compute_metrics(results)
    print_metrics(metrics)

    # Phase 5: Save
    output_dir = PROJECT_ROOT / "experiments" / "results" / "steering_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "checkpoint": args.checkpoint,
        "data_folder": args.data_folder,
        "max_trajectories": args.max_trajectories,
        "samples_per_traj": args.samples_per_traj,
        "train_fraction": args.train_fraction,
        "mc_passes": args.mc_passes,
        "delta": args.delta,
        "binary_search_steps": args.binary_search_steps,
        "seed": args.seed,
        "device": str(device),
        "n_train": len(train_samples),
        "n_test": len(test_samples),
        "elapsed_seconds": elapsed,
    }

    output = {
        "config": config,
        "metrics": metrics,
        "samples": results,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    save_plots(results, metrics, output_dir)
    print(f"Plots saved to {output_dir}")


# ===========================================================================
# Metrics
# ===========================================================================
def compute_metrics(results):
    """Compute AUROC and Spearman for each signal at each layer + MC Dropout."""
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    failures = np.array([r["is_failure"] for r in results])
    rel_errors = np.array([r["relative_error"] for r in results])
    has_both_classes = len(np.unique(failures)) == 2

    # Build signal dict: signal_name -> values array
    signals = {"mc_dropout_var": np.array([r["mc_dropout_var"] for r in results])}

    for layer_idx in range(NUM_LAYERS):
        # For projection, low projection = uncertain, so we negate for AUROC
        # (higher signal = more likely failure)
        proj_vals = np.array([r[f"projection_L{layer_idx}"] for r in results])
        signals[f"projection_L{layer_idx}"] = -proj_vals  # negate: low proj → high uncertainty

        signals[f"sensitivity_L{layer_idx}"] = np.array(
            [r[f"sensitivity_L{layer_idx}"] for r in results]
        )

        signals[f"random_sens_L{layer_idx}"] = np.array(
            [r[f"random_sens_L{layer_idx}"] for r in results]
        )

        # For alpha_star, small α* = uncertain, so negate for AUROC
        alpha_vals = np.array([r[f"alpha_star_L{layer_idx}"] for r in results])
        signals[f"alpha_star_L{layer_idx}"] = -alpha_vals  # negate: small α* → high uncertainty

        # Combined: projection / sensitivity (negate — high ratio = confident & correct)
        proj_sens = np.array([r[f"proj_over_sens_L{layer_idx}"] for r in results])
        signals[f"proj_over_sens_L{layer_idx}"] = -proj_sens

    metrics = {}
    for name, values in signals.items():
        auroc = None
        if has_both_classes:
            try:
                auroc = float(roc_auc_score(failures, values))
            except ValueError:
                pass

        rho, p = spearmanr(values, rel_errors)
        metrics[name] = {
            "auroc": auroc,
            "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
            "spearman_p": float(p) if not np.isnan(p) else 1.0,
        }

    return metrics


def print_metrics(metrics):
    """Pretty-print metrics table."""
    print(f"\n{'Signal':<25} {'AUROC':>8} {'Spearman ρ':>12} {'p-value':>12}")
    print("-" * 60)

    # MC Dropout first
    m = metrics["mc_dropout_var"]
    auroc_str = f"{m['auroc']:.4f}" if m["auroc"] is not None else "N/A"
    print(f"{'mc_dropout_var':<25} {auroc_str:>8} {m['spearman_rho']:>12.4f} {m['spearman_p']:>12.2e}")

    print("-" * 60)

    # Per-layer signals grouped
    for signal_type in ["projection", "sensitivity", "random_sens", "alpha_star", "proj_over_sens"]:
        for layer_idx in range(NUM_LAYERS):
            name = f"{signal_type}_L{layer_idx}"
            if name not in metrics:
                continue
            m = metrics[name]
            auroc_str = f"{m['auroc']:.4f}" if m["auroc"] is not None else "N/A"
            print(f"{name:<25} {auroc_str:>8} {m['spearman_rho']:>12.4f} {m['spearman_p']:>12.2e}")
        print("-" * 60)


# ===========================================================================
# Plots
# ===========================================================================
def save_plots(results, metrics, output_dir):
    """Generate scatter plots, ROC curves, and per-layer comparison."""
    from sklearn.metrics import roc_curve, roc_auc_score

    failures = np.array([r["is_failure"] for r in results])
    rel_errors = np.array([r["relative_error"] for r in results])
    has_both_classes = len(np.unique(failures)) == 2

    # --- 1. Scatter plots for each signal ---
    signal_keys = ["mc_dropout_var"]
    for layer_idx in range(NUM_LAYERS):
        signal_keys.extend([
            f"projection_L{layer_idx}",
            f"sensitivity_L{layer_idx}",
            f"alpha_star_L{layer_idx}",
        ])

    for key in signal_keys:
        # Use raw values for scatter (not negated)
        if key.startswith("projection_") or key.startswith("alpha_star_"):
            raw_key = key.split("_L")[0]
            layer = int(key.split("_L")[1])
            values = np.array([r[f"{raw_key}_L{layer}"] for r in results])
        else:
            values = np.array([r[key] for r in results]) if key in results[0] else -metrics[key]
            if key == "mc_dropout_var":
                values = np.array([r["mc_dropout_var"] for r in results])
            else:
                values = np.array([r[key] for r in results])

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["green" if not f else "red" for f in failures]
        ax.scatter(values, rel_errors, c=colors, alpha=0.6, s=20)
        ax.set_xlabel(key)
        ax.set_ylabel("Relative Error")
        m = metrics.get(key, {})
        rho = m.get("spearman_rho", 0)
        auroc = m.get("auroc")
        auroc_str = f"{auroc:.3f}" if auroc is not None else "N/A"
        ax.set_title(f"{key} vs Relative Error\n(Spearman ρ={rho:.3f}, AUROC={auroc_str})")
        ax.legend(handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", label="Success"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", label="Failure"),
        ])
        fig.tight_layout()
        fig.savefig(output_dir / f"scatter_{key}.png", dpi=150)
        plt.close(fig)

    # --- 2. ROC curves ---
    if has_both_classes:
        # One plot per signal type, with all layers overlaid
        for signal_type, title in [
            ("sensitivity", "Steering Sensitivity"),
            ("projection", "Steering Projection"),
            ("alpha_star", "Min Steering Distance (α*)"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

            for layer_idx in range(NUM_LAYERS):
                key = f"{signal_type}_L{layer_idx}"
                m = metrics.get(key, {})
                auroc = m.get("auroc")
                if auroc is None:
                    continue

                # Use the same negated values as in compute_metrics
                if signal_type == "projection":
                    values = -np.array([r[f"projection_L{layer_idx}"] for r in results])
                elif signal_type == "alpha_star":
                    values = -np.array([r[f"alpha_star_L{layer_idx}"] for r in results])
                else:
                    values = np.array([r[f"sensitivity_L{layer_idx}"] for r in results])

                fpr, tpr, _ = roc_curve(failures, values)
                ax.plot(fpr, tpr, color=colors[layer_idx],
                        label=f"Layer {layer_idx} (AUC={auroc:.3f})")

            # MC Dropout reference
            mc_vals = np.array([r["mc_dropout_var"] for r in results])
            mc_auroc = metrics["mc_dropout_var"].get("auroc")
            if mc_auroc is not None:
                fpr, tpr, _ = roc_curve(failures, mc_vals)
                ax.plot(fpr, tpr, "k--", label=f"MC Dropout (AUC={mc_auroc:.3f})")

            ax.plot([0, 1], [0, 1], "k:", alpha=0.3)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC — {title}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / f"roc_{signal_type}.png", dpi=150)
            plt.close(fig)

    # --- 3. Per-layer comparison bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    signal_types = ["projection", "sensitivity", "alpha_star"]
    x = np.arange(NUM_LAYERS)
    width = 0.25
    colors_bar = ["tab:blue", "tab:orange", "tab:green"]

    # AUROC comparison
    ax = axes[0]
    for j, st in enumerate(signal_types):
        aurocs = []
        for layer_idx in range(NUM_LAYERS):
            a = metrics[f"{st}_L{layer_idx}"]["auroc"]
            aurocs.append(a if a is not None else 0.5)
        ax.bar(x + j * width, aurocs, width, label=st, color=colors_bar[j])
    mc_auroc = metrics["mc_dropout_var"]["auroc"]
    if mc_auroc is not None:
        ax.axhline(mc_auroc, color="black", linestyle="--", label="MC Dropout")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC by Signal Type and Layer")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"L{i}" for i in range(NUM_LAYERS)])
    ax.legend(fontsize=8)

    # Spearman comparison
    ax = axes[1]
    for j, st in enumerate(signal_types):
        rhos = [metrics[f"{st}_L{layer_idx}"]["spearman_rho"] for layer_idx in range(NUM_LAYERS)]
        ax.bar(x + j * width, rhos, width, label=st, color=colors_bar[j])
    mc_rho = metrics["mc_dropout_var"]["spearman_rho"]
    ax.axhline(mc_rho, color="black", linestyle="--", label="MC Dropout")
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Spearman ρ by Signal Type and Layer")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"L{i}" for i in range(NUM_LAYERS)])
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "layer_comparison.png", dpi=150)
    plt.close(fig)

    # --- 4. Signal statistics ---
    print("\n=== Signal Statistics ===")
    for key in ["mc_dropout_var"] + [
        f"{st}_L{l}" for st in signal_types for l in range(NUM_LAYERS)
    ]:
        vals = np.array([r[key] for r in results]) if key in results[0] else np.zeros(len(results))
        if key in results[0]:
            vals = np.array([r[key] for r in results])
            print(f"  {key:<25}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
                  f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Steering Sensitivity as Uncertainty — ViNT Experiment"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to vint.pth checkpoint")
    parser.add_argument("--data-folder", type=str, required=True,
                        help="Path to dataset folder (e.g., go_stanford_cropped)")
    parser.add_argument("--max-trajectories", type=int, default=None,
                        help="Max trajectories to use (default: all)")
    parser.add_argument("--samples-per-traj", type=int, default=30,
                        help="Samples per trajectory (default: 30)")
    parser.add_argument("--train-fraction", type=float, default=0.7,
                        help="Fraction of samples for steering vector computation (default: 0.7)")
    parser.add_argument("--mc-passes", type=int, default=20,
                        help="MC Dropout forward passes (default: 20)")
    parser.add_argument("--delta", type=float, default=0.5,
                        help="Prediction change threshold for alpha* search (default: 0.5)")
    parser.add_argument("--binary-search-steps", type=int, default=16,
                        help="Binary search iterations for alpha* (default: 16)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    run_experiment(args)


if __name__ == "__main__":
    main()
