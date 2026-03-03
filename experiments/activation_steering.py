"""
Activation Steering Experiment for ViNT — Per-Layer Residual Stream

Computes a steering vector at each of the 4 transformer layer residual streams
and evaluates which layer is most effective for improving distance prediction.

Architecture:
  tokens [B, 7, 512]
    → positional_encoding
    → sa_decoder.layers[0]  ← residual stream layer 0
    → sa_decoder.layers[1]  ← residual stream layer 1
    → sa_decoder.layers[2]  ← residual stream layer 2
    → sa_decoder.layers[3]  ← residual stream layer 3
    → flatten → MLP (512→256→128→64→32) → dist_predictor, action_predictor

Each TransformerEncoderLayer (norm_first=True) does:
    x = x + self_attn(norm1(x))   # residual after attention
    x = x + ffn(norm2(x))         # residual after FFN
Hooking layer[i] captures the residual stream output at that boundary.

Usage:
  python experiments/activation_steering.py \
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
    PROJECT_ROOT,
)

ALPHA_VALUES = [0, -2.0, -1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0, 2.0]
FAILURE_THRESHOLD = 0.3
NUM_LAYERS = 4  # ViNT has 4 transformer layers


# ===========================================================================
# Hook helpers
# ===========================================================================
class MultiLayerCapture:
    """Captures residual stream output at each transformer layer."""

    def __init__(self):
        self.values = {}  # layer_idx -> tensor
        self._handles = []

    def install(self, model):
        for i, layer in enumerate(model.decoder.sa_decoder.layers):
            capture = self
            layer_idx = i

            def make_hook(idx):
                def hook_fn(module, input, output):
                    capture.values[idx] = output.detach().clone()
                return hook_fn

            h = layer.register_forward_hook(make_hook(i))
            self._handles.append(h)

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.values.clear()


class SteeringHook:
    """Forward hook that adds alpha * steering_vector to a layer's output."""

    def __init__(self, steering_vector: torch.Tensor, alpha: float):
        self.steering_vector = steering_vector  # [7, 512]
        self.alpha = alpha

    def __call__(self, module, input, output):
        return output + self.alpha * self.steering_vector


# ===========================================================================
# Phase 1: Collect per-layer hidden states
# ===========================================================================
def collect_hidden_states(model, loader, samples, device):
    """Capture residual stream at all 4 layers for each sample."""
    capture = MultiLayerCapture()
    capture.install(model)

    records = []
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(samples):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Collecting {i + 1}/{len(samples)}")

            obs = loader.build_obs_tensor(sample["traj_name"], sample["curr_time"]).to(device)
            goal = loader.build_goal_tensor(sample["traj_name"], sample["goal_time"]).to(device)

            dist_pred, _, _ = model(obs, goal)
            d_predicted = dist_pred.item()
            d_actual = sample["d_actual"]
            _, rel_error = compute_failure_proxy(d_predicted, d_actual, threshold=FAILURE_THRESHOLD)

            # Store hidden state from each layer
            layer_hiddens = {}
            for layer_idx, val in capture.values.items():
                layer_hiddens[layer_idx] = val.cpu().squeeze(0)  # [7, 512]

            records.append({
                "layer_hiddens": layer_hiddens,
                "d_predicted": d_predicted,
                "d_actual": d_actual,
                "relative_error": rel_error,
                "is_positive": rel_error < FAILURE_THRESHOLD,
                **sample,
            })

    capture.remove()
    return records


# ===========================================================================
# Phase 2: Compute per-layer steering vectors
# ===========================================================================
def compute_steering_vectors(records):
    """Compute a unit steering vector per layer."""
    pos_records = [r for r in records if r["is_positive"]]
    neg_records = [r for r in records if not r["is_positive"]]

    print(f"\n=== Steering Vector Stats ===")
    print(f"  Positive samples (rel_error < {FAILURE_THRESHOLD}): {len(pos_records)}")
    print(f"  Negative samples (rel_error >= {FAILURE_THRESHOLD}): {len(neg_records)}")

    if not pos_records or not neg_records:
        print("  ERROR: Need both positive and negative samples.")
        return None

    steering_vectors = {}
    for layer_idx in range(NUM_LAYERS):
        pos_h = torch.stack([r["layer_hiddens"][layer_idx] for r in pos_records])
        neg_h = torch.stack([r["layer_hiddens"][layer_idx] for r in neg_records])

        pos_mean = pos_h.mean(dim=0)
        neg_mean = neg_h.mean(dim=0)

        raw_vec = pos_mean - neg_mean  # unnormalized steering vector
        raw_norm = raw_vec.norm().item()

        cos_sim = torch.nn.functional.cosine_similarity(
            pos_mean.flatten().unsqueeze(0), neg_mean.flatten().unsqueeze(0)
        ).item()

        steering_vectors[layer_idx] = raw_vec  # keep raw norm; alpha scales it

        print(f"  Layer {layer_idx}: raw_norm={raw_norm:.2f}, cos_sim={cos_sim:.4f}")

    return steering_vectors


# ===========================================================================
# Phase 3: Evaluate per-layer steering
# ===========================================================================
def evaluate_steering(model, loader, test_samples, steering_vectors, device):
    """For each layer and alpha, steer and measure."""
    all_results = {}  # (layer_idx, alpha) -> [sample_results]

    for layer_idx in range(NUM_LAYERS):
        sv = steering_vectors[layer_idx].to(device)
        target_layer = model.decoder.sa_decoder.layers[layer_idx]
        print(f"\n  --- Layer {layer_idx} ---")

        for alpha in ALPHA_VALUES:
            hook = SteeringHook(sv, alpha)
            handle = target_layer.register_forward_hook(hook)

            sample_results = []
            model.eval()
            with torch.no_grad():
                for sample in test_samples:
                    obs = loader.build_obs_tensor(sample["traj_name"], sample["curr_time"]).to(device)
                    goal = loader.build_goal_tensor(sample["traj_name"], sample["goal_time"]).to(device)

                    dist_pred, _, _ = model(obs, goal)
                    d_predicted = dist_pred.item()
                    d_actual = sample["d_actual"]
                    is_failure, rel_error = compute_failure_proxy(
                        d_predicted, d_actual, threshold=FAILURE_THRESHOLD
                    )

                    sample_results.append({
                        "traj_name": sample["traj_name"],
                        "curr_time": sample["curr_time"],
                        "goal_time": sample["goal_time"],
                        "d_actual": d_actual,
                        "d_predicted": d_predicted,
                        "relative_error": rel_error,
                        "is_failure": is_failure,
                    })

            handle.remove()
            all_results[(layer_idx, alpha)] = sample_results
            mean_err = np.mean([r["relative_error"] for r in sample_results])
            fail_rate = np.mean([r["is_failure"] for r in sample_results])
            if alpha in (0, ALPHA_VALUES[1], ALPHA_VALUES[-1]):
                print(f"    alpha={alpha:<6} | err={mean_err:.4f} | fail={fail_rate:.4f}")

    return all_results


# ===========================================================================
# Phase 4: Output
# ===========================================================================
def print_summary_table(all_results):
    """Print per-layer summary table."""
    baseline_err = np.mean([r["relative_error"] for r in all_results[(0, 0)]])
    baseline_fail = np.mean([r["is_failure"] for r in all_results[(0, 0)]])

    print(f"\nBaseline: mean_rel_error={baseline_err:.4f}, failure_rate={baseline_fail:.4f}")
    print(f"\n{'layer':>5} | {'alpha':>6} | {'mean_rel_err':>12} | {'failure_rate':>12} | {'d_err':>8} | {'d_fail':>8}")
    print("-" * 70)

    for layer_idx in range(NUM_LAYERS):
        for alpha in ALPHA_VALUES:
            if alpha == 0 and layer_idx > 0:
                continue  # baseline is same for all layers
            samples = all_results[(layer_idx, alpha)]
            mean_err = np.mean([r["relative_error"] for r in samples])
            fail_rate = np.mean([r["is_failure"] for r in samples])
            d_err = mean_err - baseline_err
            d_fail = fail_rate - baseline_fail
            print(f"{layer_idx:>5} | {alpha:>6} | {mean_err:>12.4f} | {fail_rate:>12.4f} | {d_err:>+8.4f} | {d_fail:>+8.4f}")
        if layer_idx < NUM_LAYERS - 1:
            print("-" * 70)


def find_best_config(all_results):
    """Find best (layer, alpha) by lowest failure rate, then lowest error."""
    baseline_fail = np.mean([r["is_failure"] for r in all_results[(0, 0)]])
    baseline_err = np.mean([r["relative_error"] for r in all_results[(0, 0)]])

    best_key = (0, 0)
    best_fail = baseline_fail
    best_err = baseline_err

    for (layer_idx, alpha), samples in all_results.items():
        fail_rate = np.mean([r["is_failure"] for r in samples])
        mean_err = np.mean([r["relative_error"] for r in samples])
        if fail_rate < best_fail or (fail_rate == best_fail and mean_err < best_err):
            best_key = (layer_idx, alpha)
            best_fail = fail_rate
            best_err = mean_err

    print(f"\nBest config: layer={best_key[0]}, alpha={best_key[1]}")
    print(f"  failure_rate: {baseline_fail:.4f} -> {best_fail:.4f} ({best_fail - baseline_fail:+.4f})")
    print(f"  mean_rel_err: {baseline_err:.4f} -> {best_err:.4f} ({best_err - baseline_err:+.4f})")
    return best_key


def save_plots(all_results, output_dir):
    """Generate per-layer comparison plots."""
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    # Plot 1: alpha vs failure rate, one line per layer
    fig, ax = plt.subplots(figsize=(10, 6))
    for layer_idx in range(NUM_LAYERS):
        fail_rates = []
        for alpha in ALPHA_VALUES:
            samples = all_results[(layer_idx, alpha)]
            fail_rates.append(np.mean([r["is_failure"] for r in samples]))
        ax.plot(ALPHA_VALUES, fail_rates, "o-", color=colors[layer_idx],
                label=f"Layer {layer_idx}", markersize=4)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Failure Rate (rel_error >= 30%)")
    ax.set_title("Per-Layer Residual Stream Steering: Alpha vs Failure Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "layer_alpha_vs_failure_rate.png", dpi=150)
    plt.close(fig)

    # Plot 2: alpha vs mean relative error, one line per layer
    fig, ax = plt.subplots(figsize=(10, 6))
    for layer_idx in range(NUM_LAYERS):
        mean_errs = []
        for alpha in ALPHA_VALUES:
            samples = all_results[(layer_idx, alpha)]
            mean_errs.append(np.mean([r["relative_error"] for r in samples]))
        ax.plot(ALPHA_VALUES, mean_errs, "o-", color=colors[layer_idx],
                label=f"Layer {layer_idx}", markersize=4)
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Mean Relative Error")
    ax.set_title("Per-Layer Residual Stream Steering: Alpha vs Mean Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "layer_alpha_vs_mean_error.png", dpi=150)
    plt.close(fig)

    # Plot 3: per-sample scatter for best config
    # Find best config
    best_key = (0, 0)
    best_fail = 1.0
    for (layer_idx, alpha), samples in all_results.items():
        fail_rate = np.mean([r["is_failure"] for r in samples])
        if fail_rate < best_fail:
            best_key = (layer_idx, alpha)
            best_fail = fail_rate

    baseline_errors = [r["relative_error"] for r in all_results[(0, 0)]]
    steered_errors = [r["relative_error"] for r in all_results[best_key]]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(baseline_errors, steered_errors, alpha=0.5, s=15, color="tab:purple")
    max_val = max(max(baseline_errors), max(steered_errors)) * 1.05
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="y=x (no change)")
    ax.set_xlabel("Baseline Error (alpha=0)")
    ax.set_ylabel(f"Steered Error (layer={best_key[0]}, alpha={best_key[1]})")
    ax.set_title(f"Per-Sample: Baseline vs Best Steered (layer {best_key[0]}, a={best_key[1]})")
    ax.legend()
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_vs_best_steered.png", dpi=150)
    plt.close(fig)

    # Plot 4: heatmap — layer x alpha → delta failure rate
    baseline_fail = np.mean([r["is_failure"] for r in all_results[(0, 0)]])
    heatmap = np.zeros((NUM_LAYERS, len(ALPHA_VALUES)))
    for i, layer_idx in enumerate(range(NUM_LAYERS)):
        for j, alpha in enumerate(ALPHA_VALUES):
            fail = np.mean([r["is_failure"] for r in all_results[(layer_idx, alpha)]])
            heatmap[i, j] = fail - baseline_fail

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(heatmap, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(ALPHA_VALUES)))
    ax.set_xticklabels([str(a) for a in ALPHA_VALUES])
    ax.set_yticks(range(NUM_LAYERS))
    ax.set_yticklabels([f"Layer {i}" for i in range(NUM_LAYERS)])
    ax.set_xlabel("Alpha")
    ax.set_title("Delta Failure Rate vs Baseline (green = improvement)")
    for i in range(NUM_LAYERS):
        for j in range(len(ALPHA_VALUES)):
            ax.text(j, i, f"{heatmap[i, j]:+.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "heatmap_layer_alpha.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {output_dir}")


def save_results_json(all_results, config, output_dir):
    """Save results to JSON."""
    baseline_err = np.mean([r["relative_error"] for r in all_results[(0, 0)]])
    baseline_fail = np.mean([r["is_failure"] for r in all_results[(0, 0)]])

    summary = []
    for layer_idx in range(NUM_LAYERS):
        for alpha in ALPHA_VALUES:
            samples = all_results[(layer_idx, alpha)]
            mean_err = np.mean([r["relative_error"] for r in samples])
            fail_rate = np.mean([r["is_failure"] for r in samples])
            summary.append({
                "layer": layer_idx,
                "alpha": alpha,
                "mean_relative_error": float(mean_err),
                "failure_rate": float(fail_rate),
                "delta_error": float(mean_err - baseline_err),
                "delta_failure": float(fail_rate - baseline_fail),
            })

    output = {
        "config": config,
        "summary": summary,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {results_path}")


# ===========================================================================
# Main
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

    # Deterministic train/test split
    rng = np.random.RandomState(args.seed)
    indices = np.arange(len(all_samples))
    rng.shuffle(indices)
    split_idx = int(len(all_samples) * args.train_fraction)
    train_samples = [all_samples[i] for i in indices[:split_idx]]
    test_samples = [all_samples[i] for i in indices[split_idx:]]
    print(f"Split: {len(train_samples)} train, {len(test_samples)} test")

    # Phase 1: Collect per-layer hidden states
    print("\n=== Phase 1: Collecting per-layer hidden states (train split) ===")
    t0 = time.time()
    train_records = collect_hidden_states(model, loader, train_samples, device)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Phase 2: Compute per-layer steering vectors
    print("\n=== Phase 2: Computing per-layer steering vectors ===")
    steering_vectors = compute_steering_vectors(train_records)
    if steering_vectors is None:
        return

    # Phase 3: Evaluate per-layer steering
    print("\n=== Phase 3: Evaluating per-layer steering (test split) ===")
    t0 = time.time()
    all_results = evaluate_steering(model, loader, test_samples, steering_vectors, device)
    print(f"\n  Done in {time.time() - t0:.1f}s")

    # Phase 4: Output
    print("\n=== Phase 4: Results ===")
    print_summary_table(all_results)
    best = find_best_config(all_results)

    output_dir = PROJECT_ROOT / "experiments" / "results" / "activation_steering"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "checkpoint": args.checkpoint,
        "data_folder": args.data_folder,
        "max_trajectories": args.max_trajectories,
        "samples_per_traj": args.samples_per_traj,
        "train_fraction": args.train_fraction,
        "seed": args.seed,
        "device": str(device),
        "n_train": len(train_samples),
        "n_test": len(test_samples),
        "hook_point": "model.decoder.sa_decoder.layers[i] (per-layer residual stream)",
        "hidden_shape": "[7, 512]",
        "alpha_values": ALPHA_VALUES,
        "num_layers": NUM_LAYERS,
        "failure_threshold": FAILURE_THRESHOLD,
    }

    save_results_json(all_results, config, output_dir)
    save_plots(all_results, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Activation Steering Experiment for ViNT (per-layer residual stream)"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-folder", type=str, required=True)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument("--samples-per-traj", type=int, default=30)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    run_experiment(args)


if __name__ == "__main__":
    main()
