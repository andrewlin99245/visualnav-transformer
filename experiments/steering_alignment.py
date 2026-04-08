"""
Steering Alignment Analysis

Measures how well steering vector alignment (cosine similarity) predicts
per-sample training loss. Supports cross-dataset evaluation: compute steering
vectors from one dataset, evaluate on another (e.g., go_stanford → SCAND).

Usage:
  # Same dataset (ID):
  python experiments/steering_alignment.py \
    --checkpoint deployment/model_weights/vint.pth \
    --data-folder /path/to/go_stanford_cropped/go_stanford \
    --max-samples 5000

  # Cross-dataset (OOD): compute vectors from go_stanford, evaluate on SCAND:
  python experiments/steering_alignment.py \
    --checkpoint deployment/model_weights/vint.pth \
    --data-folder /path/to/go_stanford_cropped/go_stanford \
    --eval-data-folder /path/to/scand \
    --eval-data-split-folder train/vint_train/data/data_splits/scand/train \
    --eval-dataset-name scand \
    --max-samples 5000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from scipy.stats import spearmanr, kendalltau

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "train"))

from vint_train.models.vint.vint import ViNT
from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.data.data_utils import VISUALIZATION_IMAGE_SIZE

from uncertainty_correlation import load_vint_model


# ===========================================================================
# Hook to capture residual stream at all layers
# ===========================================================================
class MultiLayerCapture:
    def __init__(self):
        self.captured = {}
        self._handles = []

    def install(self, layers):
        for i, layer in enumerate(layers):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    self.captured[idx] = output.detach()
                    return output
                return hook_fn
            self._handles.append(layer.register_forward_hook(make_hook(i)))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.captured.clear()


# ===========================================================================
# Per-sample loss (mirrors _compute_losses exactly)
# ===========================================================================
def compute_per_sample_loss(dist_pred, action_pred, dist_label, action_label,
                            action_mask, alpha=0.5):
    dist_loss = (dist_pred.squeeze(-1) - dist_label.float()) ** 2
    action_loss = F.mse_loss(action_pred, action_label, reduction="none")
    while action_loss.dim() > 1:
        action_loss = action_loss.mean(dim=-1)
    action_loss = action_loss * action_mask
    total_loss = alpha * 1e-2 * dist_loss + (1 - alpha) * action_loss
    return total_loss, dist_loss, action_loss


# ===========================================================================
# Forward pass: collect losses and hidden states
# ===========================================================================
def collect_losses_and_hiddens(model, dataset, subset_indices, transform,
                               device, num_layers, hook, label=""):
    t0 = time.time()
    all_total_losses = []
    all_dist_losses = []
    all_action_losses = []
    all_hiddens = {i: [] for i in range(num_layers)}

    skipped = 0
    with torch.no_grad():
        for count, idx in enumerate(subset_indices):
            if (count + 1) % 500 == 0 or count == 0:
                elapsed = time.time() - t0
                rate = (count + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{label}] Sample {count + 1}/{len(subset_indices)} ({rate:.1f}/sec)")

            try:
                (obs_image, goal_image, action_label, dist_label,
                 goal_pos, dataset_index, action_mask) = dataset[int(idx)]
            except (TypeError, FileNotFoundError, OSError) as e:
                skipped += 1
                if skipped <= 3:
                    print(f"    Skip sample {idx}: {e}")
                continue

            if obs_image is None or goal_image is None:
                skipped += 1
                continue

            obs_images = torch.split(obs_image.unsqueeze(0), 3, dim=1)
            obs_images = [transform(img).to(device) for img in obs_images]
            obs_image_t = torch.cat(obs_images, dim=1)
            goal_image_t = transform(goal_image.unsqueeze(0)).to(device)

            dist_label_t = dist_label.unsqueeze(0).to(device)
            action_label_t = action_label.unsqueeze(0).to(device)
            action_mask_t = action_mask.unsqueeze(0).float().to(device)

            dist_pred, action_pred, _ = model(obs_image_t, goal_image_t)

            total_loss, dist_loss, action_loss = compute_per_sample_loss(
                dist_pred.cpu(), action_pred.cpu(),
                dist_label_t.cpu(), action_label_t.cpu(),
                action_mask_t.cpu(), alpha=0.5,
            )

            all_total_losses.append(total_loss.item())
            all_dist_losses.append(dist_loss.item())
            all_action_losses.append(action_loss.item())

            for layer_idx in range(num_layers):
                h = hook.captured[layer_idx]
                h_flat = h.reshape(-1).cpu()
                all_hiddens[layer_idx].append(h_flat)

    if skipped > 0:
        print(f"  [{label}] Skipped {skipped} samples")

    elapsed = time.time() - t0
    all_total_losses = np.array(all_total_losses)
    all_dist_losses = np.array(all_dist_losses)
    all_action_losses = np.array(all_action_losses)

    print(f"  [{label}] Done in {elapsed:.1f}s, {len(all_total_losses)} samples")
    print(f"  Total loss: mean={all_total_losses.mean():.4f}, std={all_total_losses.std():.4f}")
    print(f"  Dist loss:  mean={all_dist_losses.mean():.4f}, std={all_dist_losses.std():.4f}")
    print(f"  Action loss: mean={all_action_losses.mean():.4f}, std={all_action_losses.std():.4f}")
    da_rho, da_p = spearmanr(all_dist_losses, all_action_losses)
    print(f"  Dist-Action correlation: Spearman ρ={da_rho:.4f} (p={da_p:.2e})")

    return all_total_losses, all_dist_losses, all_action_losses, all_hiddens


def make_dataset(data_folder, data_split_folder, dataset_name):
    return ViNT_Dataset(
        data_folder=data_folder,
        data_split_folder=data_split_folder,
        dataset_name=dataset_name,
        image_size=(85, 64),
        waypoint_spacing=1,
        min_dist_cat=0,
        max_dist_cat=20,
        min_action_distance=0,
        max_action_distance=10,
        negative_mining=True,
        len_traj_pred=5,
        learn_angle=True,
        context_size=5,
        context_type="temporal",
        end_slack=0,
        goals_per_obs=2,
        normalize=True,
    )


def compute_correlations(hiddens, steering_vectors, total_losses, dist_losses,
                         action_losses, num_layers, label=""):
    print(f"\n=== Correlations ({label}) ===")
    results_per_layer = {}
    layer_cos_sims = {}

    for layer_idx in range(num_layers):
        h = torch.stack(hiddens[layer_idx])
        v = steering_vectors[layer_idx].unsqueeze(0)
        cos_sims = F.cosine_similarity(h, v, dim=1).numpy()
        layer_cos_sims[layer_idx] = cos_sims

        sp_rho, sp_p = spearmanr(cos_sims, total_losses)
        kt_tau, kt_p = kendalltau(cos_sims, total_losses)
        sp_dist, _ = spearmanr(cos_sims, dist_losses)
        sp_action, _ = spearmanr(cos_sims, action_losses)
        kt_dist, _ = kendalltau(cos_sims, dist_losses)
        kt_action, _ = kendalltau(cos_sims, action_losses)

        results_per_layer[layer_idx] = {
            "spearman_rho": float(sp_rho),
            "spearman_p": float(sp_p),
            "kendall_tau": float(kt_tau),
            "kendall_p": float(kt_p),
            "spearman_dist_only": float(sp_dist),
            "spearman_action_only": float(sp_action),
            "kendall_dist_only": float(kt_dist),
            "kendall_action_only": float(kt_action),
            "cos_sim_mean": float(cos_sims.mean()),
            "cos_sim_std": float(cos_sims.std()),
        }

        print(f"  Layer {layer_idx}: Spearman ρ={sp_rho:.4f} (p={sp_p:.2e}), "
              f"Kendall τ={kt_tau:.4f} (p={kt_p:.2e})")
        print(f"           dist_only ρ={sp_dist:.4f} τ={kt_dist:.4f}, "
              f"action_only ρ={sp_action:.4f} τ={kt_action:.4f}")

    # Cross-layer correlation
    print(f"\n  Cross-layer Spearman ρ ({label}):")
    cross_layer = {}
    for i in range(num_layers):
        for j in range(i + 1, num_layers):
            rho, _ = spearmanr(layer_cos_sims[i], layer_cos_sims[j])
            cross_layer[f"L{i}_L{j}"] = float(rho)
            print(f"    L{i} vs L{j}: ρ={rho:.4f}")

    return results_per_layer, cross_layer, layer_cos_sims


# ===========================================================================
# Main
# ===========================================================================
def run_experiment(args):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = load_vint_model(args.checkpoint, device)
    model.eval()

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    num_layers = len(model.decoder.sa_decoder.layers)
    hook = MultiLayerCapture()
    hook.install(model.decoder.sa_decoder.layers)

    rng = np.random.RandomState(args.seed)

    # --- Phase 1: Source dataset (for steering vector computation) ---
    print(f"\n=== Phase 1: Source dataset ({args.dataset_name}) ===")
    dataset = make_dataset(args.data_folder, args.data_split_folder, args.dataset_name)
    print(f"  Dataset size: {len(dataset)}")

    max_src = min(args.max_samples, len(dataset))
    src_indices = rng.choice(len(dataset), max_src, replace=False) if max_src < len(dataset) else np.arange(len(dataset))
    print(f"  Using {len(src_indices)} samples")

    src_total, src_dist, src_action, src_hiddens = collect_losses_and_hiddens(
        model, dataset, src_indices, transform, device, num_layers, hook,
        label=args.dataset_name,
    )

    # --- Phase 2: Compute steering vectors from source ---
    print(f"\n=== Phase 2: Computing steering vectors from {args.dataset_name} ===")
    n = len(src_total)
    sorted_indices = np.argsort(src_total)
    n_partition = max(1, int(n * args.partition_frac))
    pos_indices = sorted_indices[:n_partition]
    neg_indices = sorted_indices[-n_partition:]
    print(f"  Partition: {args.partition_frac:.0%} ({n_partition} samples each)")
    print(f"  Positive loss range: [{src_total[pos_indices].min():.6f}, {src_total[pos_indices].max():.6f}]")
    print(f"  Negative loss range: [{src_total[neg_indices].min():.6f}, {src_total[neg_indices].max():.6f}]")

    steering_vectors = {}
    for layer_idx in range(num_layers):
        hiddens = torch.stack(src_hiddens[layer_idx])
        pos_mean = hiddens[pos_indices].mean(dim=0)
        neg_mean = hiddens[neg_indices].mean(dim=0)
        v = pos_mean - neg_mean
        v_norm = v / (v.norm() + 1e-12)
        steering_vectors[layer_idx] = v_norm

        raw_norm = v.norm().item()
        cos = F.cosine_similarity(pos_mean.unsqueeze(0), neg_mean.unsqueeze(0)).item()
        print(f"  Layer {layer_idx}: raw_norm={raw_norm:.2f}, cos_sim(pos,neg)={cos:.4f}")

    # --- Phase 3: Correlations on source (ID) ---
    src_results, src_cross, _ = compute_correlations(
        src_hiddens, steering_vectors, src_total, src_dist, src_action,
        num_layers, label=f"ID: {args.dataset_name}",
    )

    # --- Phase 4: Eval dataset (OOD) if provided ---
    eval_results = None
    eval_cross = None
    if args.eval_data_folder:
        print(f"\n=== Phase 4: Eval dataset ({args.eval_dataset_name}) ===")
        eval_dataset = make_dataset(
            args.eval_data_folder, args.eval_data_split_folder,
            args.eval_dataset_name,
        )
        print(f"  Dataset size: {len(eval_dataset)}")

        max_eval = min(args.max_samples, len(eval_dataset))
        eval_indices = rng.choice(len(eval_dataset), max_eval, replace=False) if max_eval < len(eval_dataset) else np.arange(len(eval_dataset))
        print(f"  Using {len(eval_indices)} samples")

        eval_total, eval_dist, eval_action, eval_hiddens = collect_losses_and_hiddens(
            model, eval_dataset, eval_indices, transform, device, num_layers, hook,
            label=args.eval_dataset_name,
        )

        eval_results, eval_cross, _ = compute_correlations(
            eval_hiddens, steering_vectors, eval_total, eval_dist, eval_action,
            num_layers, label=f"OOD: {args.eval_dataset_name}",
        )

    hook.remove()

    # --- Save results ---
    print("\n=== Saving results ===")
    output_dir = PROJECT_ROOT / "experiments" / "results" / "steering_alignment"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "checkpoint": args.checkpoint,
            "source_data": args.dataset_name,
            "eval_data": args.eval_dataset_name if args.eval_data_folder else None,
            "max_samples": args.max_samples,
            "partition_frac": args.partition_frac,
            "seed": args.seed,
        },
        "source": {
            "per_layer": src_results,
            "cross_layer": src_cross,
            "loss_stats": {
                "total": {"mean": float(src_total.mean()), "std": float(src_total.std())},
                "dist": {"mean": float(src_dist.mean()), "std": float(src_dist.std())},
                "action": {"mean": float(src_action.mean()), "std": float(src_action.std())},
            },
        },
    }
    if eval_results:
        output["eval"] = {
            "per_layer": eval_results,
            "cross_layer": eval_cross,
        }

    with open(output_dir / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {output_dir / 'results.json'}")
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(description="Steering Alignment Analysis")
    parser.add_argument("--checkpoint", type=str,
                        default=str(PROJECT_ROOT / "deployment" / "model_weights" / "vint.pth"))
    parser.add_argument("--data-folder", type=str,
                        default=str(PROJECT_ROOT / "datasets" / "scand"),
                        help="Source dataset folder (for steering vector computation)")
    parser.add_argument("--data-split-folder", type=str,
                        default=str(PROJECT_ROOT / "train" / "vint_train" / "data" / "data_splits" / "scand" / "train"))
    parser.add_argument("--dataset-name", type=str, default="scand")
    parser.add_argument("--eval-data-folder", type=str,
                        default=str(PROJECT_ROOT / "datasets" / "go_stanford_cropped" / "go_stanford"),
                        help="Eval dataset folder (if different from source)")
    parser.add_argument("--eval-data-split-folder", type=str,
                        default=str(PROJECT_ROOT / "train" / "vint_train" / "data" / "data_splits" / "go_stanford" / "train"))
    parser.add_argument("--eval-dataset-name", type=str, default="go_stanford")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--partition-frac", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_experiment(args)


if __name__ == "__main__":
    main()
