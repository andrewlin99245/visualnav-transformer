"""Evaluate steering injection using total loss (dist + action).

Partition by total loss, evaluate by total loss.
No action_mask filter — all samples included for vector computation.
"""
import sys, os, time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
from vint_train.models.gnm.modified_mobilenetv2 import MobileNetEncoder
from vint_train.models.vint.vint import ViNT

sys.path.insert(0, os.path.dirname(__file__))
from steering_injection import (
    load_vint_model, SequentialTrajectoryLoader, SteeringInjector,
    MultiLayerCapture, TRANSFORM, compute_per_sample_loss
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results" / "wp1_steering"


def run_baseline(model, dataset, indices, device, num_layers, hook, label=""):
    """Collect all samples (no mask filter), compute total loss, collect hiddens."""
    np.random.seed(42)
    t0 = time.time()

    total_losses = []
    dist_losses = []
    action_losses = []
    all_hiddens = {i: [] for i in range(num_layers)}

    with torch.no_grad():
        for count, idx in enumerate(indices):
            if (count + 1) % 2000 == 0 or count == 0:
                elapsed = time.time() - t0
                rate = (count + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{label}] {count + 1}/{len(indices)} ({rate:.1f}/sec)")

            try:
                sample = dataset[int(idx)]
            except (TypeError, FileNotFoundError, OSError):
                continue
            if sample is None:
                continue

            if len(sample) == 7:
                obs_image, goal_image, action_label, dist_label, goal_pos, _, action_mask = sample
            else:
                obs_image, goal_image, action_label, dist_label, goal_pos, action_mask = sample

            obs_images = torch.split(obs_image.unsqueeze(0), 3, dim=1)
            obs_images = [TRANSFORM(img).to(device) for img in obs_images]
            obs_image_t = torch.cat(obs_images, dim=1)
            goal_image_t = TRANSFORM(goal_image.unsqueeze(0)).to(device)

            dist_pred, action_pred, _ = model(obs_image_t, goal_image_t)

            # Total loss = dist_loss + action_loss (same as training alpha=0.5)
            dist_loss = F.mse_loss(dist_pred.cpu(), dist_label.unsqueeze(0).float()).item()
            action_loss = F.mse_loss(action_pred.cpu(), action_label.unsqueeze(0)).item()
            total_loss = 0.005 * dist_loss + 0.5 * action_loss

            total_losses.append(total_loss)
            dist_losses.append(dist_loss)
            action_losses.append(action_loss)

            for li in range(num_layers):
                h = hook.captured[li].reshape(-1).cpu().half()
                all_hiddens[li].append(h)

    elapsed = time.time() - t0
    n = len(total_losses)
    print(f"  [{label}] {n} samples in {elapsed:.1f}s")
    print(f"  [{label}] TotalLoss={np.mean(total_losses):.6f}, "
          f"DistLoss={np.mean(dist_losses):.6f}, "
          f"ActionLoss={np.mean(action_losses):.6f}")

    hiddens = {}
    for li in range(num_layers):
        hiddens[li] = torch.stack(all_hiddens[li]).numpy()

    return {
        "total_losses": np.array(total_losses),
        "hiddens": hiddens,
        "total_loss": np.mean(total_losses),
        "dist_loss": np.mean(dist_losses),
        "action_loss": np.mean(action_losses),
    }


def compute_vectors(losses, hiddens, num_layers, partition_frac=0.25):
    """Compute steering vectors by partitioning on total loss."""
    n = len(losses)
    q = max(1, int(n * partition_frac))
    sorted_idx = np.argsort(losses)
    good_idx = sorted_idx[:q]
    bad_idx = sorted_idx[-q:]

    vectors = {}
    h_mean_norms = {}
    for li in range(num_layers):
        h = torch.from_numpy(hiddens[li]).float()
        good_mean = h[good_idx].mean(dim=0)
        bad_mean = h[bad_idx].mean(dim=0)
        v = good_mean - bad_mean
        cos = F.cosine_similarity(good_mean.unsqueeze(0), bad_mean.unsqueeze(0)).item()
        print(f"  Layer {li}: norm={v.norm().item():.2f}, cos(good,bad)={cos:.4f}")
        vectors[li] = v

        h_reshaped = h.reshape(h.shape[0], 7, 512)
        h_mean_norms[li] = h_reshaped.norm(dim=-1).mean().item()

    print(f"  Partitioned: {q} good, {q} bad (from {n} total)")
    return vectors, h_mean_norms


def compute_spearman(vectors, hiddens, losses, num_layers, label=""):
    """Compute Spearman correlation between cos_sim(h, v) and losses."""
    print(f"  Spearman correlations ({label}):")
    for li in range(num_layers):
        h = torch.from_numpy(hiddens[li]).float()
        v = vectors[li].unsqueeze(0)
        cos_sims = F.cosine_similarity(h, v, dim=1).numpy()
        rho, pval = spearmanr(cos_sims, losses)
        print(f"    Layer {li}: ρ={rho:+.4f} (p={pval:.2e})")


def run_inject(model, dataset, indices, device, label=""):
    """Injection pass: compute total loss on all samples."""
    np.random.seed(42)
    t0 = time.time()

    total_losses = []
    dist_losses = []
    action_losses = []

    with torch.no_grad():
        for count, idx in enumerate(indices):
            if (count + 1) % 2000 == 0 or count == 0:
                elapsed = time.time() - t0
                rate = (count + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{label}] {count + 1}/{len(indices)} ({rate:.1f}/sec)")

            try:
                sample = dataset[int(idx)]
            except (TypeError, FileNotFoundError, OSError):
                continue
            if sample is None:
                continue

            if len(sample) == 7:
                obs_image, goal_image, action_label, dist_label, goal_pos, _, action_mask = sample
            else:
                obs_image, goal_image, action_label, dist_label, goal_pos, action_mask = sample

            obs_images = torch.split(obs_image.unsqueeze(0), 3, dim=1)
            obs_images = [TRANSFORM(img).to(device) for img in obs_images]
            obs_image_t = torch.cat(obs_images, dim=1)
            goal_image_t = TRANSFORM(goal_image.unsqueeze(0)).to(device)

            dist_pred, action_pred, _ = model(obs_image_t, goal_image_t)

            dist_loss = F.mse_loss(dist_pred.cpu(), dist_label.unsqueeze(0).float()).item()
            action_loss = F.mse_loss(action_pred.cpu(), action_label.unsqueeze(0)).item()
            total_loss = 0.005 * dist_loss + 0.5 * action_loss

            total_losses.append(total_loss)
            dist_losses.append(dist_loss)
            action_losses.append(action_loss)

    elapsed = time.time() - t0
    n = len(total_losses)
    print(f"  [{label}] {n} samples in {elapsed:.1f}s")
    print(f"  [{label}] TotalLoss={np.mean(total_losses):.6f}, "
          f"DistLoss={np.mean(dist_losses):.6f}, "
          f"ActionLoss={np.mean(action_losses):.6f}")

    return {
        "total_loss": np.mean(total_losses),
        "dist_loss": np.mean(dist_losses),
        "action_loss": np.mean(action_losses),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="deployment/model_weights/vint.pth")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--partition-frac", type=float, default=0.25)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    parser.add_argument("--gs-data-folder", default=str(PROJECT_ROOT / "datasets" / "go_stanford_cropped" / "go_stanford"))
    parser.add_argument("--gs-split-folder", default=str(PROJECT_ROOT / "train" / "vint_train" / "data" / "data_splits" / "go_stanford" / "train"))
    parser.add_argument("--scand-data-folder", default=str(PROJECT_ROOT / "datasets" / "scand"))
    parser.add_argument("--scand-split-folder", default=str(PROJECT_ROOT / "train" / "vint_train" / "data" / "data_splits" / "scand" / "train"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_vint_model(args.checkpoint, device)
    model.eval()
    num_layers = len(model.decoder.sa_decoder.layers)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use sequential loader (full trajectory inference)
    gs_dataset = SequentialTrajectoryLoader(args.gs_data_folder, args.gs_split_folder, "go_stanford")
    sc_dataset = SequentialTrajectoryLoader(args.scand_data_folder, args.scand_split_folder, "scand")

    rng = np.random.RandomState(args.seed)
    gs_n, sc_n = len(gs_dataset), len(sc_dataset)
    gs_indices = rng.choice(gs_n, min(args.max_samples or gs_n, gs_n), replace=False) if args.max_samples else np.arange(gs_n)
    sc_indices = rng.choice(sc_n, min(args.max_samples or sc_n, sc_n), replace=False) if args.max_samples else np.arange(sc_n)
    print(f"go_stanford: {len(gs_indices)} samples, scand: {len(sc_indices)} samples")

    # Phase 1: Baselines with hidden state collection
    print("\n=== Phase 1: Baseline go_stanford ===")
    hook = MultiLayerCapture()
    hook.install(model.decoder.sa_decoder.layers)
    gs_base = run_baseline(model, gs_dataset, gs_indices, device, num_layers, hook, label="gs")
    hook.remove()

    print("\n=== Phase 2: Baseline scand ===")
    hook = MultiLayerCapture()
    hook.install(model.decoder.sa_decoder.layers)
    sc_base = run_baseline(model, sc_dataset, sc_indices, device, num_layers, hook, label="sc")
    hook.remove()

    # Phase 2: Compute steering vectors (partitioned by total loss)
    print("\n=== Phase 3: Computing steering vectors (by total loss) ===")
    print("\ngo_stanford:")
    gs_vectors, gs_h_norms = compute_vectors(gs_base["total_losses"], gs_base["hiddens"], num_layers, args.partition_frac)
    print("\nscand:")
    sc_vectors, sc_h_norms = compute_vectors(sc_base["total_losses"], sc_base["hiddens"], num_layers, args.partition_frac)

    # Spearman: cos_sim(h, v) vs total_loss for all 4 combinations
    print("\n=== Spearman correlations ===")
    for vec_name, vectors in [("gs_vec", gs_vectors), ("sc_vec", sc_vectors)]:
        for data_name, base in [("gs_data", gs_base), ("sc_data", sc_base)]:
            compute_spearman(vectors, base["hiddens"], base["total_losses"], num_layers,
                           label=f"{vec_name} → {data_name}")

    del gs_base["hiddens"], sc_base["hiddens"]

    # Phase 3: Injection runs
    results = {"gs_baseline": gs_base, "sc_baseline": sc_base}

    configs = [
        ("gs_on_gs",    gs_vectors, gs_dataset, gs_indices, gs_h_norms),
        ("gs_on_scand", gs_vectors, sc_dataset, sc_indices, sc_h_norms),
        ("scand_on_gs", sc_vectors, gs_dataset, gs_indices, gs_h_norms),
        ("scand_on_scand", sc_vectors, sc_dataset, sc_indices, sc_h_norms),
    ]

    for name, vectors, dataset, indices, h_norms_dict in configs:
        print(f"\n=== Injection: {name} (alpha={args.alpha}) ===")
        injector = SteeringInjector(vectors, alpha=args.alpha, h_mean_norms=h_norms_dict)
        injector.install(model.decoder.sa_decoder.layers)
        results[name] = run_inject(model, dataset, indices, device, label=name)
        injector.remove()

    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTS — total loss partition & eval (alpha={args.alpha})")
    print(f"{'='*70}")
    print(f"{'Setting':<25} {'TotalLoss':<12} {'DistLoss':<12} {'ActionLoss':<12}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12}")

    for key in ["gs_baseline", "sc_baseline", "gs_on_gs", "gs_on_scand", "scand_on_gs", "scand_on_scand"]:
        r = results[key]
        print(f"{key:<25} {r['total_loss']:<12.6f} {r['dist_loss']:<12.6f} {r['action_loss']:<12.6f}")

    print(f"\n{'Setting':<25} {'Total Chg':<14} {'Dist Chg':<14} {'Action Chg':<14}")
    print(f"{'-'*25} {'-'*14} {'-'*14} {'-'*14}")
    comparisons = [
        ("gs_on_gs", "gs_baseline"),
        ("gs_on_scand", "sc_baseline"),
        ("scand_on_gs", "gs_baseline"),
        ("scand_on_scand", "sc_baseline"),
    ]
    for inj, base in comparisons:
        ri, rb = results[inj], results[base]
        total_chg = (ri["total_loss"] / rb["total_loss"] - 1) * 100
        dist_chg = (ri["dist_loss"] / rb["dist_loss"] - 1) * 100
        action_chg = (ri["action_loss"] / rb["action_loss"] - 1) * 100
        print(f"{inj:<25} {total_chg:+.1f}%{'':<8} {dist_chg:+.1f}%{'':<8} {action_chg:+.1f}%")


if __name__ == "__main__":
    main()
