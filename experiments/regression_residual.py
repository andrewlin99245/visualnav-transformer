"""
Regression residual analysis for steering vectors.

1. Load stored indices + datasets to extract ground truth action labels
2. Compute ground truth features: goal_distance, tortuosity, heading_change
3. Fit linear regression: loss ~ dist + tortuosity + heading_change
4. Partition by regression residual to compute steering vectors
5. Compute Spearman correlation between cos_sim(h, v) and residual

Usage:
  cd experiments && python regression_residual.py
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "train"))

from vint_train.data.vint_dataset import ViNT_Dataset
from steering_injection import make_dataset

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "steering_injection"
NUM_LAYERS = 4
PARTITION_FRAC = 0.25


def compute_gt_features(dataset, indices):
    """Extract ground truth action labels and compute tortuosity + heading change."""
    np.random.seed(42)
    tortuosities = []
    heading_changes = []
    skipped = set()

    for i, idx in enumerate(indices):
        if (i + 1) % 5000 == 0:
            print(f"  Extracting GT features: {i+1}/{len(indices)}")

        try:
            sample = dataset[int(idx)]
        except Exception:
            skipped.add(i)
            tortuosities.append(0.0)
            heading_changes.append(0.0)
            continue

        if sample is None:
            skipped.add(i)
            tortuosities.append(0.0)
            heading_changes.append(0.0)
            continue

        if len(sample) == 7:
            action_label = sample[2]
        else:
            action_label = sample[2]

        # action_label: [5, 4] = (x, y, sin, cos) per waypoint
        al = action_label.view(5, 4).numpy()
        xy = al[:, :2]  # [5, 2]

        # Tortuosity: path_length / euclidean_distance_to_last_waypoint
        # Path goes through waypoints sequentially from origin (0,0)
        points = np.vstack([[0, 0], xy])  # [6, 2] including origin
        segments = np.diff(points, axis=0)  # [5, 2]
        seg_lengths = np.linalg.norm(segments, axis=1)  # [5]
        path_length = seg_lengths.sum()
        euclidean = np.linalg.norm(xy[-1])  # distance from origin to last waypoint
        if euclidean > 1e-6:
            tort = path_length / euclidean
        else:
            tort = 1.0
        tortuosities.append(tort)

        # Heading change: sum of absolute angle differences between consecutive segments
        angles = np.arctan2(segments[:, 1], segments[:, 0])  # [5]
        angle_diffs = np.abs(np.diff(angles))  # [4]
        angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)  # wrap to [0, pi]
        heading_changes.append(angle_diffs.sum())

    if skipped:
        print(f"  Skipped {len(skipped)} samples")
    return np.array(tortuosities), np.array(heading_changes)


def main():
    # Load stored data
    print("=== Loading stored data ===")
    datasets = {}
    for name in ["go_stanford", "scand"]:
        d = OUTPUT_DIR / name
        datasets[name] = {
            "losses": np.load(d / "losses.npy"),
            "action_masks": np.load(d / "action_masks.npy"),
            "dist_labels": np.load(d / "dist_labels.npy"),
            "indices": np.load(d / "indices.npy"),
        }
        for li in range(NUM_LAYERS):
            datasets[name][f"hiddens_L{li}"] = np.load(d / f"hiddens_L{li}.npy")
        n = len(datasets[name]["losses"])
        n_valid = (datasets[name]["action_masks"] > 0).sum()
        print(f"  {name}: {n} samples, {n_valid} valid")

    # Load ViNT datasets to extract GT features
    print("\n=== Loading ViNT datasets for GT features ===")
    gs_dataset = make_dataset(
        str(PROJECT_ROOT / "datasets" / "go_stanford_cropped" / "go_stanford"),
        str(PROJECT_ROOT / "train" / "vint_train" / "data" / "data_splits" / "go_stanford" / "train"),
        "go_stanford")
    sc_dataset = make_dataset(
        str(PROJECT_ROOT / "datasets" / "scand"),
        str(PROJECT_ROOT / "train" / "vint_train" / "data" / "data_splits" / "scand" / "train"),
        "scand")

    print("\n--- go_stanford GT features ---")
    gs_tort, gs_heading = compute_gt_features(gs_dataset, datasets["go_stanford"]["indices"])
    print("\n--- scand GT features ---")
    sc_tort, sc_heading = compute_gt_features(sc_dataset, datasets["scand"]["indices"])

    datasets["go_stanford"]["tortuosity"] = gs_tort
    datasets["go_stanford"]["heading_change"] = gs_heading
    datasets["scand"]["tortuosity"] = sc_tort
    datasets["scand"]["heading_change"] = sc_heading

    # Fit regression and compute residuals for each dataset
    print("\n=== Fitting linear regression: loss ~ dist + tortuosity + heading_change ===")
    for name in ["go_stanford", "scand"]:
        d = datasets[name]
        valid = d["action_masks"] > 0
        dist = d["dist_labels"][valid]
        tort = d["tortuosity"][valid]
        head = d["heading_change"][valid]
        total_loss = d["losses"][valid, 0]

        X = np.column_stack([dist, tort, head])
        reg = LinearRegression().fit(X, total_loss)
        pred = reg.predict(X)
        residual = total_loss - pred

        print(f"\n  {name} (valid only, n={valid.sum()}):")
        print(f"    R² = {reg.score(X, total_loss):.4f}")
        print(f"    Coefficients: dist={reg.coef_[0]:.6f}, tort={reg.coef_[1]:.6f}, heading={reg.coef_[2]:.6f}")
        print(f"    Intercept: {reg.intercept_:.6f}")
        print(f"    Residual std: {residual.std():.6f}")

        # Store residuals (full array, 0 for invalid)
        full_residual = np.zeros(len(d["losses"]))
        full_residual[valid] = residual
        d["residual"] = full_residual

    # Compute steering vectors from residual partitioning
    print("\n=== Computing steering vectors from regression residual ===")
    vectors = {}
    for name in ["go_stanford", "scand"]:
        d = datasets[name]
        valid = d["action_masks"] > 0
        valid_indices = np.where(valid)[0]
        residuals = d["residual"][valid_indices]

        n = len(valid_indices)
        n_part = max(1, int(n * PARTITION_FRAC))
        sorted_idx = np.argsort(residuals)
        pos_idx = valid_indices[sorted_idx[:n_part]]   # lowest residual = best
        neg_idx = valid_indices[sorted_idx[-n_part:]]   # highest residual = worst

        print(f"\n  {name}: partition {n_part} samples each")
        print(f"    Positive residual range: [{residuals[sorted_idx[:n_part]].min():.4f}, {residuals[sorted_idx[:n_part]].max():.4f}]")
        print(f"    Negative residual range: [{residuals[sorted_idx[-n_part:]].min():.4f}, {residuals[sorted_idx[-n_part:]].max():.4f}]")
        print(f"    Positive dist_label mean: {d['dist_labels'][pos_idx].mean():.1f}, Negative: {d['dist_labels'][neg_idx].mean():.1f}")

        vecs = {}
        for li in range(NUM_LAYERS):
            h = torch.from_numpy(d[f"hiddens_L{li}"]).float()
            pos_mean = h[pos_idx].mean(dim=0)
            neg_mean = h[neg_idx].mean(dim=0)
            v = pos_mean - neg_mean
            vecs[li] = v
            cos = F.cosine_similarity(pos_mean.unsqueeze(0), neg_mean.unsqueeze(0)).item()
            print(f"    Layer {li}: norm={v.norm().item():.2f}, cos(pos,neg)={cos:.4f}")
        vectors[name] = vecs

    # Spearman: cos_sim(h, v) vs regression residual
    print("\n=== Spearman: cos_sim(h, v) vs regression residual (layer 3) ===")
    li = 3
    for vec_name in ["go_stanford", "scand"]:
        for data_name in ["go_stanford", "scand"]:
            d = datasets[data_name]
            valid = d["action_masks"] > 0
            h = torch.from_numpy(d[f"hiddens_L{li}"]).float()[valid]
            v = vectors[vec_name][li].unsqueeze(0)
            cos_sims = F.cosine_similarity(h, v, dim=1).numpy()
            residual = d["residual"][valid]
            total_loss = d["losses"][valid, 0]
            dist_loss = d["losses"][valid, 1]
            action_loss = d["losses"][valid, 2]

            rho_res, _ = spearmanr(cos_sims, residual)
            rho_total, _ = spearmanr(cos_sims, total_loss)
            rho_dist, _ = spearmanr(cos_sims, dist_loss)
            rho_action, _ = spearmanr(cos_sims, action_loss)
            print(f"  vec={vec_name} -> data={data_name}: "
                  f"rho(residual)={rho_res:+.4f}  rho(total)={rho_total:+.4f}  "
                  f"rho(dist)={rho_dist:+.4f}  rho(action)={rho_action:+.4f}")

    # Also distance-conditioned Spearman vs residual
    print("\n=== Distance-conditioned Spearman vs residual (layer 3) ===")
    for vec_name in ["go_stanford", "scand"]:
        for data_name in ["go_stanford", "scand"]:
            d = datasets[data_name]
            valid = d["action_masks"] > 0
            h = torch.from_numpy(d[f"hiddens_L{li}"]).float()[valid]
            v = vectors[vec_name][li].unsqueeze(0)
            cos_sims = F.cosine_similarity(h, v, dim=1).numpy()
            residual = d["residual"][valid]
            dist_labels = d["dist_labels"][valid]

            unique_dists = np.unique(dist_labels)
            bucket_rhos = []
            for dd in unique_dists:
                mask = dist_labels == dd
                if mask.sum() < 5:
                    continue
                r, _ = spearmanr(cos_sims[mask], residual[mask])
                bucket_rhos.append(r)
            avg_rho = np.mean(bucket_rhos) if bucket_rhos else float('nan')
            print(f"  vec={vec_name} -> data={data_name} (dist-cond): "
                  f"rho(residual)={avg_rho:+.4f}  ({len(bucket_rhos)} buckets)")

    print("\nDone.")


if __name__ == "__main__":
    main()
