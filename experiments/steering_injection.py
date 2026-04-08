"""
Steering Injection Experiment

6 runs total:
  1. Baseline inference on go_stanford → save losses, hiddens
  2. Baseline inference on SCAND → save losses, hiddens
  3. Compute steering vectors from both datasets (top 25% - bottom 25%)
  4. Inject go_stanford vector → inference on go_stanford
  5. Inject go_stanford vector → inference on SCAND
  6. Inject SCAND vector → inference on go_stanford
  7. Inject SCAND vector → inference on SCAND

Usage:
  conda run -n nomad_train python experiments/steering_injection.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "train"))

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.data.data_utils import get_data_path, img_path_to_data, to_local_coords, calculate_sin_cos
from uncertainty_correlation import load_vint_model
import yaml
import pickle
import io
import lmdb
import tqdm

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "steering_injection"


# ===========================================================================
# Hooks
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


class SteeringInjector:
    """Adds normalized steering vector to residual stream.

    Perturbation = alpha * (h_mean_norm / v_norm) * v
    This makes alpha a fraction of the typical activation magnitude,
    consistent across datasets regardless of vector or residual stream scale.
    """
    def __init__(self, vectors, alpha=0.05, h_mean_norms=None):
        self.scaled_vectors = {}
        for idx, v in vectors.items():
            v_norm = v.norm().item()
            if h_mean_norms is not None and idx in h_mean_norms:
                scale = h_mean_norms[idx] / max(v_norm, 1e-12)
            else:
                scale = 1.0
            self.scaled_vectors[idx] = v * scale
        self.alpha = alpha
        self._handles = []

    def install(self, layers):
        for i, layer in enumerate(layers):
            def make_hook(idx):
                def hook_fn(module, input, output):
                    v = self.scaled_vectors[idx].to(output.device)
                    v = v.reshape(1, output.shape[1], output.shape[2])
                    orig_norm = output.norm(dim=-1, keepdim=True)
                    steered = output + self.alpha * v
                    steered = steered * (orig_norm / steered.norm(dim=-1, keepdim=True).clamp(min=1e-12))
                    return steered
                return hook_fn
            self._handles.append(layer.register_forward_hook(make_hook(i)))

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


# ===========================================================================
# Loss computation
# ===========================================================================
def compute_per_sample_loss(dist_pred, action_pred, dist_label, action_label,
                            action_mask, alpha=0.5):
    dist_loss = (dist_pred.squeeze(-1) - dist_label.float()) ** 2
    action_loss = F.mse_loss(action_pred, action_label, reduction="none")
    while action_loss.dim() > 1:
        action_loss = action_loss.mean(dim=-1)
    action_loss = action_loss * action_mask
    total_loss = alpha * 1e-2 * dist_loss + (1 - alpha) * action_loss

    # Relative errors (scale-invariant)
    eps = 1e-6
    dist_rel = torch.abs(dist_pred.squeeze(-1) - dist_label.float()) / dist_label.float().clamp(min=1.0)

    # Per-waypoint relative error, computed separately for position and heading
    # action shapes: [B, 5, 4] where 4 = [x, y, sin, cos]
    ap = action_pred.view(-1, 5, 4) if action_pred.dim() >= 2 else action_pred.unsqueeze(0).view(-1, 5, 4)
    al = action_label.view(-1, 5, 4) if action_label.dim() >= 2 else action_label.unsqueeze(0).view(-1, 5, 4)
    # Position relative error: ||pred_xy - label_xy|| / max(||label_xy||, eps)
    pos_err = (ap[..., :2] - al[..., :2]).norm(dim=-1)          # [B, 5]
    pos_scale = al[..., :2].norm(dim=-1).clamp(min=eps)          # [B, 5]
    pos_rel = (pos_err / pos_scale).mean(dim=-1)                 # [B]
    # Heading error: ||pred_sc - label_sc|| (no normalization, ground truth norm ≈ 1)
    heading_err = (ap[..., 2:] - al[..., 2:]).norm(dim=-1).mean(dim=-1)  # [B]
    # Weighted combination: 0.7 position + 0.3 heading
    action_rel = (0.7 * pos_rel + 0.3 * heading_err) * action_mask  # [B]

    total_rel = 0.5 * action_loss + 0.01 * dist_rel

    return total_loss, dist_loss, action_loss, dist_rel, action_rel, total_rel


# ===========================================================================
# Sequential trajectory loader (no random sampling)
# ===========================================================================
class SequentialTrajectoryLoader:
    """Iterates through trajectories sequentially. For each valid timestep,
    pairs observation (with temporal context) with a goal at a fixed offset.

    Yields: (obs_image, goal_image, action_label, dist_label, action_mask)
    """
    def __init__(self, data_folder, data_split_folder, dataset_name,
                 image_size=(85, 64), context_size=5, waypoint_spacing=1,
                 len_traj_pred=5, goal_offset=5, normalize=True):
        self.data_folder = data_folder
        self.image_size = image_size
        self.context_size = context_size
        self.waypoint_spacing = waypoint_spacing
        self.len_traj_pred = len_traj_pred
        self.goal_offset = goal_offset
        self.normalize = normalize
        self.num_action_params = 3  # x, y, yaw (learn_angle=True)

        # Load data config
        data_config_path = os.path.join(
            PROJECT_ROOT, "train", "vint_train", "data", "data_config.yaml")
        with open(data_config_path, "r") as f:
            all_data_config = yaml.safe_load(f)
        self.data_config = all_data_config[dataset_name]

        # Load trajectory names
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            self.traj_names = [l.strip() for l in f if l.strip()]

        # Build LMDB cache path
        cache_filename = os.path.join(data_split_folder, f"dataset_{dataset_name}.lmdb")
        if not os.path.exists(cache_filename):
            # Build cache
            print(f"  Building LMDB cache for {dataset_name}...")
            all_image_paths = []
            for traj_name in self.traj_names:
                traj_data = self._load_traj_data(traj_name)
                traj_len = len(traj_data["position"])
                for t in range(traj_len):
                    all_image_paths.append(
                        get_data_path(self.data_folder, traj_name, t))
            with lmdb.open(cache_filename, map_size=2**40) as env:
                with env.begin(write=True) as txn:
                    for path in tqdm.tqdm(all_image_paths, desc="Caching images"):
                        if os.path.exists(path):
                            with open(path, "rb") as f:
                                txn.put(path.encode(), f.read())

        self._image_cache = lmdb.open(cache_filename, readonly=True)

        # Build sequential index: (traj_name, curr_time, goal_time)
        self.samples = []
        for traj_name in self.traj_names:
            traj_data = self._load_traj_data(traj_name)
            traj_len = len(traj_data["position"])

            begin_time = self.context_size * self.waypoint_spacing
            end_time = traj_len - self.len_traj_pred * self.waypoint_spacing
            for curr_time in range(begin_time, end_time):
                goal_time = min(curr_time + self.goal_offset, traj_len - 1)
                self.samples.append((traj_name, curr_time, goal_time))

        print(f"  [{dataset_name}] {len(self.samples)} sequential samples "
              f"from {len(self.traj_names)} trajectories")

    def _load_traj_data(self, traj_name):
        path = os.path.join(self.data_folder, traj_name, "traj_data.pkl")
        with open(path, "rb") as f:
            traj_data = pickle.load(f)
        for key in ["position", "yaw"]:
            if key in traj_data and hasattr(traj_data[key], "dtype") and traj_data[key].dtype == object:
                traj_data[key] = np.array(traj_data[key].tolist(), dtype=np.float64)
        return traj_data

    def _load_image(self, traj_name, time):
        image_path = get_data_path(self.data_folder, traj_name, time)
        try:
            with self._image_cache.begin() as txn:
                buf = txn.get(image_path.encode())
                image_bytes = io.BytesIO(bytes(buf))
            return img_path_to_data(image_bytes, self.image_size)
        except TypeError:
            return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        traj_name, curr_time, goal_time = self.samples[i]
        traj_data = self._load_traj_data(traj_name)

        # Load context + current observation
        context_times = list(range(
            curr_time - self.context_size * self.waypoint_spacing,
            curr_time + 1,
            self.waypoint_spacing,
        ))
        obs_images = [self._load_image(traj_name, t) for t in context_times]
        if any(img is None for img in obs_images):
            return None
        obs_image = torch.cat(obs_images)

        # Load goal
        goal_image = self._load_image(traj_name, goal_time)
        if goal_image is None:
            return None

        # Compute actions (same as ViNT_Dataset._compute_actions)
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
        yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]
        positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]
        goal_pos = traj_data["position"][min(goal_time, len(traj_data["position"]) - 1)]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)
        if yaw.shape[0] < self.len_traj_pred + 1:
            pad_len = self.len_traj_pred + 1 - yaw.shape[0]
            yaw = np.concatenate([yaw, np.repeat(yaw[-1], pad_len)])
            positions = np.concatenate(
                [positions, np.repeat(positions[-1:], pad_len, axis=0)], axis=0)

        waypoints = to_local_coords(positions, positions[0], yaw[0])
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])

        yaw_diff = yaw[1:] - yaw[0]
        actions = np.concatenate([waypoints[1:], yaw_diff[:, None]], axis=-1)

        if self.normalize:
            actions[:, :2] /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing
            goal_pos /= self.data_config["metric_waypoint_spacing"] * self.waypoint_spacing

        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        actions_torch = calculate_sin_cos(actions_torch)

        distance = goal_time - curr_time
        action_mask = 1.0  # always valid, no negative mining

        return (
            obs_image.float(),
            goal_image.float(),
            actions_torch,
            torch.tensor(distance, dtype=torch.int64),
            torch.tensor(goal_pos, dtype=torch.float32),
            torch.tensor(action_mask, dtype=torch.float32),
        )


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
        negative_mining=False,
        len_traj_pred=5,
        learn_angle=True,
        context_size=5,
        context_type="temporal",
        end_slack=0,
        goals_per_obs=1,
        normalize=True,
    )


TRANSFORM = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ===========================================================================
# Inference: baseline (collect hiddens) or with injection (losses only)
# ===========================================================================
def run_inference(model, dataset, indices, device, num_layers,
                  hook=None, injector=None, label="", save_hiddens=False,
                  metric_waypoint_spacing=1.0):
    """Run inference on dataset samples.

    If save_hiddens=True, collects and returns hidden states (needs capture hook).
    If injector is provided, steering vectors are injected during forward pass.
    """
    t0 = time.time()
    all_total = []
    all_dist = []
    all_action = []
    all_dist_rel = []
    all_action_rel = []
    all_total_rel = []
    all_dist_label = []
    all_action_mask = []
    all_hiddens = {i: [] for i in range(num_layers)} if save_hiddens else None
    all_dist_pred = []
    all_action_pred = []
    all_ade = []
    all_fde = []
    all_heading_err = []
    all_action_label = []

    # Reset RNG so random goal sampling is deterministic across runs
    np.random.seed(42)

    skipped = 0
    with torch.no_grad():
        for count, idx in enumerate(indices):
            if (count + 1) % 500 == 0 or count == 0:
                elapsed = time.time() - t0
                rate = (count + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{label}] {count + 1}/{len(indices)} ({rate:.1f}/sec)")

            try:
                sample = dataset[int(idx)]
            except (TypeError, FileNotFoundError, OSError) as e:
                skipped += 1
                if skipped <= 3:
                    print(f"    Skip {idx}: {e}")
                continue

            if sample is None:
                skipped += 1
                continue

            # ViNT_Dataset returns 7 elements, SequentialTrajectoryLoader returns 6
            if len(sample) == 7:
                (obs_image, goal_image, action_label, dist_label,
                 goal_pos, _dataset_index, action_mask) = sample
            else:
                (obs_image, goal_image, action_label, dist_label,
                 goal_pos, action_mask) = sample

            obs_images = torch.split(obs_image.unsqueeze(0), 3, dim=1)
            obs_images = [TRANSFORM(img).to(device) for img in obs_images]
            obs_image_t = torch.cat(obs_images, dim=1)
            goal_image_t = TRANSFORM(goal_image.unsqueeze(0)).to(device)

            dist_label_t = dist_label.unsqueeze(0).to(device)
            action_label_t = action_label.unsqueeze(0).to(device)
            action_mask_t = action_mask.unsqueeze(0).float().to(device)

            dist_pred, action_pred, _ = model(obs_image_t, goal_image_t)

            total_loss, dist_loss, action_loss, dist_rel, action_rel, total_rel = compute_per_sample_loss(
                dist_pred.cpu(), action_pred.cpu(),
                dist_label_t.cpu(), action_label_t.cpu(),
                action_mask_t.cpu(),
            )

            all_total.append(total_loss.item())
            all_dist.append(dist_loss.item())
            all_action.append(action_loss.item())
            all_dist_rel.append(dist_rel.item())
            all_action_rel.append(action_rel.item())
            all_total_rel.append(total_rel.item())
            all_dist_label.append(float(dist_label))
            all_dist_pred.append(dist_pred.cpu().item())
            all_action_pred.append(action_pred.cpu().squeeze(0).numpy())
            all_action_mask.append(float(action_mask))

            # ADE/FDE in metric space (meters)
            ap = action_pred.cpu().view(5, 4)
            al = action_label.view(5, 4)
            # Unnormalize xy positions back to meters
            scale = metric_waypoint_spacing  # waypoint_spacing=1
            pred_xy = ap[:, :2] * scale
            label_xy = al[:, :2] * scale
            wp_displacements = (pred_xy - label_xy).norm(dim=-1)  # [5]
            ade = wp_displacements.mean().item()
            fde = wp_displacements[-1].item()
            # Heading error in radians: atan2(sin, cos) difference
            pred_angle = torch.atan2(ap[:, 2], ap[:, 3])
            label_angle = torch.atan2(al[:, 2], al[:, 3])
            angle_diff = torch.abs(pred_angle - label_angle)
            angle_diff = torch.min(angle_diff, 2 * np.pi - angle_diff)  # wrap to [0, pi]
            heading_err_mean = angle_diff.mean().item()

            mask_val = float(action_mask)
            all_ade.append(ade * mask_val)
            all_fde.append(fde * mask_val)
            all_heading_err.append(heading_err_mean * mask_val)
            all_action_label.append(action_label.numpy())

            if save_hiddens and hook is not None:
                for li in range(num_layers):
                    h = hook.captured[li].reshape(-1).cpu().half()  # float16
                    all_hiddens[li].append(h)

    elapsed = time.time() - t0
    n = len(all_total)
    if skipped > 0:
        print(f"  [{label}] Skipped {skipped}")
    print(f"  [{label}] Done: {n} samples in {elapsed:.1f}s ({n/elapsed:.1f}/sec)")

    losses = np.stack([all_total, all_dist, all_action, all_dist_rel, all_action_rel,
                       all_ade, all_fde, all_heading_err, all_total_rel], axis=1)  # [N, 9]
    # For ADE/FDE/heading, compute mean over valid-action samples only
    am = np.array(all_action_mask)
    n_valid = max((am > 0).sum(), 1)
    ade_mean = losses[:, 5].sum() / n_valid
    fde_mean = losses[:, 6].sum() / n_valid
    heading_mean = np.degrees(losses[:, 7].sum() / n_valid)
    valid_losses = losses[am > 0]
    print(f"  [{label}] Total={valid_losses[:, 0].mean():.4f}, Dist={valid_losses[:, 1].mean():.4f}, "
          f"Action={valid_losses[:, 2].mean():.4f}  ({int(n_valid)} valid)")
    print(f"  [{label}] ADE={ade_mean:.4f}m, FDE={fde_mean:.4f}m, "
          f"HeadingErr={heading_mean:.2f}°")

    action_masks = np.array(all_action_mask)
    n_masked = (action_masks == 0).sum()
    if n_masked > 0:
        print(f"  [{label}] {n_masked}/{n} samples have action_mask=0")

    result = {
        "losses": losses,
        "action_masks": action_masks,
        "dist_labels": np.array(all_dist_label),
        "dist_pred": np.array(all_dist_pred),
        "action_pred": np.array(all_action_pred),
    }
    if save_hiddens:
        for li in range(num_layers):
            result[f"hiddens_L{li}"] = torch.stack(all_hiddens[li]).numpy()

    return result


# ===========================================================================
# Steering vector computation
# ===========================================================================
def compute_steering_vectors(losses, hiddens_dict, num_layers, partition_frac=0.25,
                             action_masks=None, partition_col=6, dist_labels=None):
    """Compute steering vectors from top/bottom partition.
    partition_col: 0=total, 1=dist, 2=action, 3=dist_rel, 4=action_rel, 5=ADE, 6=FDE (default),
                   7=heading_err, 8=total_rel, -2=distance-conditioned residual.
    Filters out samples with action_mask=0 before partitioning."""
    col_names = {0: "total", 1: "dist", 2: "action", 3: "dist_rel", 4: "action_rel",
                 5: "ADE", 6: "FDE", 7: "heading_err", 8: "total_rel", -2: "dist-cond-total", -3: "dist-cond-action", -4: "dist-cond-dist"}

    # Filter to valid samples
    if action_masks is not None:
        valid = action_masks > 0
        valid_indices = np.where(valid)[0]
        print(f"  Filtering: {valid.sum()}/{len(valid)} samples have action_mask=1")
    else:
        valid_indices = np.arange(len(losses))

    if partition_col == -4:
        # Distance-conditioned dist-only: percentile within each distance bucket using dist loss
        print(f"  Partitioning by: distance-conditioned residual (dist loss)")
        assert dist_labels is not None, "dist_labels required for distance-conditioned partitioning"
        sort_losses = losses[:, 1]  # dist loss
        percentiles = np.zeros(len(losses))
        dist_vals = dist_labels[valid_indices]
        unique_dists = np.unique(dist_vals)
        print(f"  Distance buckets: {len(unique_dists)} unique values, range [{unique_dists.min():.0f}, {unique_dists.max():.0f}]")
        for d in unique_dists:
            bucket_mask = dist_labels[valid_indices] == d
            bucket_idx = valid_indices[bucket_mask]
            if len(bucket_idx) < 2:
                continue
            bucket_losses = sort_losses[bucket_idx]
            ranks = np.argsort(np.argsort(bucket_losses)).astype(float) / (len(bucket_losses) - 1)
            percentiles[bucket_idx] = ranks

        n = len(valid_indices)
        n_part = max(1, int(n * partition_frac))
        valid_percentiles = percentiles[valid_indices]
        sorted_idx = np.argsort(valid_percentiles)
        pos_idx = valid_indices[sorted_idx[:n_part]]
        neg_idx = valid_indices[sorted_idx[-n_part:]]

        print(f"  Partition: {partition_frac:.0%} → {n_part} samples each")
        print(f"  Positive percentiles: [{percentiles[pos_idx].min():.3f}, {percentiles[pos_idx].max():.3f}]")
        print(f"  Negative percentiles: [{percentiles[neg_idx].min():.3f}, {percentiles[neg_idx].max():.3f}]")
        print(f"  Positive dist_label mean: {dist_labels[pos_idx].mean():.1f}, Negative dist_label mean: {dist_labels[neg_idx].mean():.1f}")
    elif partition_col == -3:
        # Distance-conditioned action-only: percentile within each distance bucket using action loss
        print(f"  Partitioning by: distance-conditioned residual (action loss)")
        assert dist_labels is not None, "dist_labels required for distance-conditioned partitioning"
        sort_losses = losses[:, 2]  # action loss
        percentiles = np.zeros(len(losses))
        dist_vals = dist_labels[valid_indices]
        unique_dists = np.unique(dist_vals)
        print(f"  Distance buckets: {len(unique_dists)} unique values, range [{unique_dists.min():.0f}, {unique_dists.max():.0f}]")
        for d in unique_dists:
            bucket_mask = dist_labels[valid_indices] == d
            bucket_idx = valid_indices[bucket_mask]
            if len(bucket_idx) < 2:
                continue
            bucket_losses = sort_losses[bucket_idx]
            ranks = np.argsort(np.argsort(bucket_losses)).astype(float) / (len(bucket_losses) - 1)
            percentiles[bucket_idx] = ranks

        n = len(valid_indices)
        n_part = max(1, int(n * partition_frac))
        valid_percentiles = percentiles[valid_indices]
        sorted_idx = np.argsort(valid_percentiles)
        pos_idx = valid_indices[sorted_idx[:n_part]]
        neg_idx = valid_indices[sorted_idx[-n_part:]]

        print(f"  Partition: {partition_frac:.0%} → {n_part} samples each")
        print(f"  Positive percentiles: [{percentiles[pos_idx].min():.3f}, {percentiles[pos_idx].max():.3f}]")
        print(f"  Negative percentiles: [{percentiles[neg_idx].min():.3f}, {percentiles[neg_idx].max():.3f}]")
        print(f"  Positive dist_label mean: {dist_labels[pos_idx].mean():.1f}, Negative dist_label mean: {dist_labels[neg_idx].mean():.1f}")
    elif partition_col == -2:
        # Distance-conditioned: compute loss percentile within each distance bucket
        print(f"  Partitioning by: distance-conditioned residual (total loss)")
        assert dist_labels is not None, "dist_labels required for distance-conditioned partitioning"
        total_losses = losses[:, 0]
        percentiles = np.zeros(len(losses))
        dist_vals = dist_labels[valid_indices]
        unique_dists = np.unique(dist_vals)
        print(f"  Distance buckets: {len(unique_dists)} unique values, range [{unique_dists.min():.0f}, {unique_dists.max():.0f}]")
        for d in unique_dists:
            bucket_mask = dist_labels[valid_indices] == d
            bucket_idx = valid_indices[bucket_mask]
            if len(bucket_idx) < 2:
                continue
            bucket_losses = total_losses[bucket_idx]
            ranks = np.argsort(np.argsort(bucket_losses)).astype(float) / (len(bucket_losses) - 1)
            percentiles[bucket_idx] = ranks

        n = len(valid_indices)
        n_part = max(1, int(n * partition_frac))
        valid_percentiles = percentiles[valid_indices]
        sorted_idx = np.argsort(valid_percentiles)
        pos_idx = valid_indices[sorted_idx[:n_part]]
        neg_idx = valid_indices[sorted_idx[-n_part:]]

        print(f"  Partition: {partition_frac:.0%} → {n_part} samples each")
        print(f"  Positive percentiles: [{percentiles[pos_idx].min():.3f}, {percentiles[pos_idx].max():.3f}]")
        print(f"  Negative percentiles: [{percentiles[neg_idx].min():.3f}, {percentiles[neg_idx].max():.3f}]")
        print(f"  Positive dist_label mean: {dist_labels[pos_idx].mean():.1f}, Negative dist_label mean: {dist_labels[neg_idx].mean():.1f}")
    else:
        print(f"  Partitioning by: {col_names.get(partition_col, partition_col)}")
        partition_losses = losses[:, partition_col]
        partition_losses_valid = partition_losses[valid_indices]

        n = len(partition_losses_valid)
        n_part = max(1, int(n * partition_frac))
        sorted_idx = np.argsort(partition_losses_valid)
        pos_idx = valid_indices[sorted_idx[:n_part]]
        neg_idx = valid_indices[sorted_idx[-n_part:]]

        print(f"  Partition: {partition_frac:.0%} → {n_part} samples each")
        print(f"  Positive loss: [{partition_losses[pos_idx].min():.6f}, {partition_losses[pos_idx].max():.6f}]")
        print(f"  Negative loss: [{partition_losses[neg_idx].min():.6f}, {partition_losses[neg_idx].max():.6f}]")

    vectors = {}
    for li in range(num_layers):
        h = torch.from_numpy(hiddens_dict[f"hiddens_L{li}"]).float()
        pos_mean = h[pos_idx].mean(dim=0)
        neg_mean = h[neg_idx].mean(dim=0)
        v = pos_mean - neg_mean
        vectors[li] = v
        print(f"  Layer {li}: norm={v.norm().item():.2f}, "
              f"cos(pos,neg)={F.cosine_similarity(pos_mean.unsqueeze(0), neg_mean.unsqueeze(0)).item():.4f}")

    return vectors


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Steering Injection Experiment")
    parser.add_argument("--checkpoint", type=str,
                        default=str(PROJECT_ROOT / "deployment" / "model_weights" / "vint.pth"))
    parser.add_argument("--gs-data-folder", type=str,
                        default=str(PROJECT_ROOT / "datasets" / "go_stanford_cropped" / "go_stanford"))
    parser.add_argument("--gs-split-folder", type=str,
                        default=str(PROJECT_ROOT / "train" / "vint_train" / "data" / "data_splits" / "go_stanford" / "train"))
    parser.add_argument("--scand-data-folder", type=str,
                        default=str(PROJECT_ROOT / "datasets" / "scand"))
    parser.add_argument("--scand-split-folder", type=str,
                        default=str(PROJECT_ROOT / "train" / "vint_train" / "data" / "data_splits" / "scand" / "train"))
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per dataset (default: all)")
    parser.add_argument("--partition-frac", type=float, default=0.25)
    parser.add_argument("--partition-col", type=int, default=6,
                        help="Loss column for partitioning: 0=total, 1=dist, 2=action, 3=dist_rel, 4=action_rel, 5=ADE, 6=FDE, 7=heading, 8=total_rel, -2=dist-cond-total, -3=dist-cond-action, -4=dist-cond-dist")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Steering strength as fraction of mean activation norm")
    parser.add_argument("--sequential", action="store_true",
                        help="Use sequential trajectory loader instead of ViNT_Dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    model = load_vint_model(args.checkpoint, device)
    model.eval()
    num_layers = len(model.decoder.sa_decoder.layers)

    # Helper to get indices
    def get_indices(dataset, label):
        n = len(dataset)
        if args.max_samples and args.max_samples < n:
            rng = np.random.RandomState(args.seed)
            indices = rng.choice(n, args.max_samples, replace=False)
            print(f"  [{label}] Subsampled {args.max_samples} from {n}")
        else:
            indices = np.arange(n)
            print(f"  [{label}] Using all {n} samples")
        return indices

    # ===== Load datasets once =====
    print(f"\n=== Loading datasets ({'sequential' if args.sequential else 'ViNT_Dataset'}) ===")
    if args.sequential:
        gs_dataset = SequentialTrajectoryLoader(args.gs_data_folder, args.gs_split_folder, "go_stanford")
        sc_dataset = SequentialTrajectoryLoader(args.scand_data_folder, args.scand_split_folder, "scand")
    else:
        gs_dataset = make_dataset(args.gs_data_folder, args.gs_split_folder, "go_stanford")
        sc_dataset = make_dataset(args.scand_data_folder, args.scand_split_folder, "scand")
    gs_indices = get_indices(gs_dataset, "go_stanford")
    sc_indices = get_indices(sc_dataset, "scand")

    # ===== Phase 1: Baseline go_stanford =====
    print("\n=== Phase 1: Baseline inference on go_stanford ===")
    hook = MultiLayerCapture()
    hook.install(model.decoder.sa_decoder.layers)
    gs_result = run_inference(model, gs_dataset, gs_indices, device, num_layers,
                              hook=hook, save_hiddens=True, label="go_stanford",
                              metric_waypoint_spacing=0.12)
    hook.remove()
    gs_losses = gs_result["losses"]
    gs_hiddens = {k: v for k, v in gs_result.items() if k.startswith("hiddens_")}
    gs_action_masks = gs_result["action_masks"]
    gs_dist_labels = gs_result["dist_labels"]

    # Save baseline go_stanford
    gs_dir = OUTPUT_DIR / "go_stanford"
    gs_dir.mkdir(parents=True, exist_ok=True)
    np.save(gs_dir / "losses.npy", gs_losses)
    np.save(gs_dir / "action_masks.npy", gs_action_masks)
    np.save(gs_dir / "dist_labels.npy", gs_dist_labels)
    np.save(gs_dir / "dist_pred.npy", gs_result["dist_pred"])
    np.save(gs_dir / "action_pred.npy", gs_result["action_pred"])
    np.save(gs_dir / "indices.npy", gs_indices)
    for k, v in gs_hiddens.items():
        np.save(gs_dir / f"{k}.npy", v)
    print(f"  Saved baseline go_stanford to {gs_dir}")
    del gs_result

    # ===== Phase 2: Baseline SCAND =====
    print("\n=== Phase 2: Baseline inference on SCAND ===")
    hook = MultiLayerCapture()
    hook.install(model.decoder.sa_decoder.layers)
    sc_result = run_inference(model, sc_dataset, sc_indices, device, num_layers,
                              hook=hook, save_hiddens=True, label="scand",
                              metric_waypoint_spacing=0.38)
    hook.remove()
    sc_losses = sc_result["losses"]
    sc_hiddens = {k: v for k, v in sc_result.items() if k.startswith("hiddens_")}
    sc_action_masks = sc_result["action_masks"]
    sc_dist_labels = sc_result["dist_labels"]

    # Save baseline scand
    sc_dir = OUTPUT_DIR / "scand"
    sc_dir.mkdir(parents=True, exist_ok=True)
    np.save(sc_dir / "losses.npy", sc_losses)
    np.save(sc_dir / "action_masks.npy", sc_action_masks)
    np.save(sc_dir / "dist_labels.npy", sc_dist_labels)
    np.save(sc_dir / "dist_pred.npy", sc_result["dist_pred"])
    np.save(sc_dir / "action_pred.npy", sc_result["action_pred"])
    np.save(sc_dir / "indices.npy", sc_indices)
    for k, v in sc_hiddens.items():
        np.save(sc_dir / f"{k}.npy", v)
    print(f"  Saved baseline scand to {sc_dir}")
    del sc_result

    # ===== Phase 3: Compute steering vectors =====
    print("\n=== Phase 3: Computing steering vectors ===")

    if args.partition_col == -1:
        # Dual-vector mode: blend FDE (col 6) and heading (col 7) vectors 50/50
        print("\n--- Dual-vector mode: 50% FDE + 50% heading ---")
        print("\ngo_stanford FDE vectors:")
        gs_fde_vectors = compute_steering_vectors(gs_losses, gs_hiddens, num_layers, args.partition_frac,
                                                  action_masks=gs_action_masks, partition_col=6)
        print("\ngo_stanford heading vectors:")
        gs_head_vectors = compute_steering_vectors(gs_losses, gs_hiddens, num_layers, args.partition_frac,
                                                   action_masks=gs_action_masks, partition_col=7)
        gs_vectors = {li: 0.5 * gs_fde_vectors[li] + 0.5 * gs_head_vectors[li] for li in range(num_layers)}

        print("\nSCAND FDE vectors:")
        sc_fde_vectors = compute_steering_vectors(sc_losses, sc_hiddens, num_layers, args.partition_frac,
                                                  action_masks=sc_action_masks, partition_col=6)
        print("\nSCAND heading vectors:")
        sc_head_vectors = compute_steering_vectors(sc_losses, sc_hiddens, num_layers, args.partition_frac,
                                                   action_masks=sc_action_masks, partition_col=7)
        sc_vectors = {li: 0.5 * sc_fde_vectors[li] + 0.5 * sc_head_vectors[li] for li in range(num_layers)}

        for li in range(num_layers):
            cos = F.cosine_similarity(gs_fde_vectors[li].unsqueeze(0), gs_head_vectors[li].unsqueeze(0)).item()
            print(f"  go_stanford Layer {li}: cos(FDE_vec, heading_vec)={cos:.4f}, blended norm={gs_vectors[li].norm().item():.2f}")
        for li in range(num_layers):
            cos = F.cosine_similarity(sc_fde_vectors[li].unsqueeze(0), sc_head_vectors[li].unsqueeze(0)).item()
            print(f"  scand Layer {li}: cos(FDE_vec, heading_vec)={cos:.4f}, blended norm={sc_vectors[li].norm().item():.2f}")
    else:
        print("\ngo_stanford steering vectors:")
        gs_vectors = compute_steering_vectors(gs_losses, gs_hiddens, num_layers, args.partition_frac,
                                              action_masks=gs_action_masks, partition_col=args.partition_col,
                                              dist_labels=gs_dist_labels)
        print("\nSCAND steering vectors:")
        sc_vectors = compute_steering_vectors(sc_losses, sc_hiddens, num_layers, args.partition_frac,
                                              action_masks=sc_action_masks, partition_col=args.partition_col,
                                              dist_labels=sc_dist_labels)

    # Save steering vectors
    col_names = {0: "total", 1: "dist", 2: "action", 3: "dist_rel", 4: "action_rel",
                 5: "ADE", 6: "FDE", 7: "heading_err", 8: "total_rel",
                 -1: "dual", -2: "dist-cond-total", -3: "dist-cond-action", -4: "dist-cond-dist"}
    vec_dir = OUTPUT_DIR / f"vectors_p{col_names.get(args.partition_col, args.partition_col)}"
    vec_dir.mkdir(parents=True, exist_ok=True)
    for li in range(num_layers):
        torch.save(gs_vectors[li], vec_dir / f"gs_vector_L{li}.pt")
        torch.save(sc_vectors[li], vec_dir / f"sc_vector_L{li}.pt")
    print(f"  Saved steering vectors to {vec_dir}")

    # Compute mean hidden norms per layer for each eval dataset
    # Used to normalize steering perturbation relative to activation scale
    def compute_mean_norms(hiddens_dict, num_layers):
        norms = {}
        for li in range(num_layers):
            h = torch.from_numpy(hiddens_dict[f"hiddens_L{li}"]).float()
            # h is [N, 3584], reshape to [N, 7, 512] to get per-token norms
            h = h.reshape(h.shape[0], 7, 512)
            norms[li] = h.norm(dim=-1).mean().item()  # mean norm across samples and tokens
        return norms

    print("\n  Mean hidden norms per layer:")
    gs_h_norms = compute_mean_norms(gs_hiddens, num_layers)
    sc_h_norms = compute_mean_norms(sc_hiddens, num_layers)
    for li in range(num_layers):
        print(f"    Layer {li}: go_stanford={gs_h_norms[li]:.2f}, scand={sc_h_norms[li]:.2f}")

    # Spearman: cos_sim(h_baseline, steering_vector) vs losses (layer 3, valid-only)
    # Both global and distance-conditioned (average per-bucket Spearman)
    from scipy.stats import spearmanr
    print("\n=== Spearman correlations (layer 3, valid-only) ===")
    datasets_map = {
        "stanford": (gs_losses, gs_hiddens, gs_action_masks, gs_dist_labels),
        "scand": (sc_losses, sc_hiddens, sc_action_masks, sc_dist_labels),
    }
    vectors_map = {"stanford": gs_vectors, "scand": sc_vectors}
    li = 3
    for vec_name, vectors in vectors_map.items():
        for data_name, (losses, hiddens, am, dl) in datasets_map.items():
            valid = am > 0
            h = torch.from_numpy(hiddens[f"hiddens_L{li}"]).float()[valid]
            v = vectors[li].unsqueeze(0)
            cos_sims = F.cosine_similarity(h, v, dim=1).numpy()
            valid_losses = losses[valid]
            valid_dl = dl[valid]

            # Global Spearman
            rho_total, _ = spearmanr(cos_sims, valid_losses[:, 0])
            rho_dist, _ = spearmanr(cos_sims, valid_losses[:, 1])
            rho_action, _ = spearmanr(cos_sims, valid_losses[:, 2])
            print(f"  vec={vec_name} -> data={data_name} (global): "
                  f"rho(total)={rho_total:+.4f}  rho(dist)={rho_dist:+.4f}  rho(action)={rho_action:+.4f}")

            # Distance-conditioned Spearman: average per-bucket
            unique_dists = np.unique(valid_dl)
            bucket_rhos = {"total": [], "dist": [], "action": []}
            for d in unique_dists:
                mask = valid_dl == d
                if mask.sum() < 5:
                    continue
                cs = cos_sims[mask]
                ls = valid_losses[mask]
                r_t, _ = spearmanr(cs, ls[:, 0])
                r_d, _ = spearmanr(cs, ls[:, 1])
                r_a, _ = spearmanr(cs, ls[:, 2])
                bucket_rhos["total"].append(r_t)
                bucket_rhos["dist"].append(r_d)
                bucket_rhos["action"].append(r_a)
            avg_total = np.mean(bucket_rhos["total"]) if bucket_rhos["total"] else float('nan')
            avg_dist = np.mean(bucket_rhos["dist"]) if bucket_rhos["dist"] else float('nan')
            avg_action = np.mean(bucket_rhos["action"]) if bucket_rhos["action"] else float('nan')
            print(f"  vec={vec_name} -> data={data_name} (dist-cond): "
                  f"rho(total)={avg_total:+.4f}  rho(dist)={avg_dist:+.4f}  rho(action)={avg_action:+.4f}  "
                  f"({len(bucket_rhos['total'])} buckets)")

    # Free hidden states
    del gs_hiddens, sc_hiddens

    # ===== Phase 4-7: Injection runs =====
    injection_configs = [
        ("gs_on_gs", gs_vectors, gs_dataset, gs_indices, 0.12, gs_h_norms),
        ("gs_on_scand", gs_vectors, sc_dataset, sc_indices, 0.38, sc_h_norms),
        ("scand_on_gs", sc_vectors, gs_dataset, gs_indices, 0.12, gs_h_norms),
        ("scand_on_scand", sc_vectors, sc_dataset, sc_indices, 0.38, sc_h_norms),
    ]

    injection_results = {}
    for name, vectors, dataset, indices, mws, h_norms in injection_configs:
        print(f"\n=== Injection {name} (alpha={args.alpha}) ===")
        injector = SteeringInjector(vectors, alpha=args.alpha, h_mean_norms=h_norms)
        injector.install(model.decoder.sa_decoder.layers)
        inj_result = run_inference(model, dataset, indices, device, num_layers,
                                   injector=injector, save_hiddens=False, label=name,
                                   metric_waypoint_spacing=mws)
        injector.remove()
        injection_results[name] = inj_result["losses"]

        # Save injection results
        inj_dir = OUTPUT_DIR / f"inject_{name}_a{args.alpha}"
        inj_dir.mkdir(parents=True, exist_ok=True)
        np.save(inj_dir / "losses.npy", inj_result["losses"])
        np.save(inj_dir / "action_masks.npy", inj_result["action_masks"])
        np.save(inj_dir / "dist_labels.npy", inj_result["dist_labels"])
        np.save(inj_dir / "dist_pred.npy", inj_result["dist_pred"])
        np.save(inj_dir / "action_pred.npy", inj_result["action_pred"])
        print(f"  Saved injection {name} to {inj_dir}")
        del inj_result

    # ===== Summary =====
    # Filter to valid samples only (action_mask=1)
    gs_valid = gs_action_masks > 0
    sc_valid = sc_action_masks > 0
    gs_bl = gs_losses[gs_valid]
    sc_bl = sc_losses[sc_valid]

    print("\n" + "=" * 70)
    print("SUMMARY (valid samples only)")
    print("=" * 70)

    print(f"\nBaseline go_stanford ({len(gs_bl)} valid): TotalLoss={gs_bl[:, 0].mean():.6f}, "
          f"DistLoss={gs_bl[:, 1].mean():.6f}, ActionLoss={gs_bl[:, 2].mean():.6f}")
    print(f"Baseline SCAND ({len(sc_bl)} valid):      TotalLoss={sc_bl[:, 0].mean():.6f}, "
          f"DistLoss={sc_bl[:, 1].mean():.6f}, ActionLoss={sc_bl[:, 2].mean():.6f}")

    print(f"\n{'Setting':<25} {'TotalLoss':<12} {'DistLoss':<12} {'ActionLoss':<12}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    for name, inj_losses in injection_results.items():
        valid_mask = gs_valid if name.endswith("_gs") else sc_valid
        il = inj_losses[valid_mask]
        print(f"{name:<25} {il[:, 0].mean():<12.6f} {il[:, 1].mean():<12.6f} {il[:, 2].mean():<12.6f}")

    print(f"\n{'Setting':<25} {'Total Chg':<14} {'Dist Chg':<14} {'Action Chg':<14}")
    print(f"{'-'*25} {'-'*14} {'-'*14} {'-'*14}")
    for name, inj_losses in injection_results.items():
        valid_mask = gs_valid if name.endswith("_gs") else sc_valid
        bl = gs_bl if name.endswith("_gs") else sc_bl
        il = inj_losses[valid_mask]
        total_chg = (il[:, 0].mean() / bl[:, 0].mean() - 1) * 100
        dist_chg = (il[:, 1].mean() / bl[:, 1].mean() - 1) * 100
        action_chg = (il[:, 2].mean() / bl[:, 2].mean() - 1) * 100
        print(f"{name:<25} {total_chg:+.1f}%{'':<8} {dist_chg:+.1f}%{'':<8} {action_chg:+.1f}%")

    # Save summary JSON
    summary = {
        "config": {
            "checkpoint": args.checkpoint,
            "alpha": args.alpha,
            "partition_frac": args.partition_frac,
            "max_samples": args.max_samples,
            "seed": args.seed,
        },
        "baseline": {
            "go_stanford": {
                "total_loss": float(gs_bl[:, 0].mean()),
                "dist_loss": float(gs_bl[:, 1].mean()),
                "action_loss": float(gs_bl[:, 2].mean()),
                "n": len(gs_bl),
            },
            "scand": {
                "total_loss": float(sc_bl[:, 0].mean()),
                "dist_loss": float(sc_bl[:, 1].mean()),
                "action_loss": float(sc_bl[:, 2].mean()),
                "n": len(sc_bl),
            },
        },
        "injection": {},
    }
    for name, inj_losses in injection_results.items():
        bl = gs_bl if name.endswith("_gs") else sc_bl
        summary["injection"][name] = {
            "total_loss": float(inj_losses[:, 0].mean()),
            "dist_loss": float(inj_losses[:, 1].mean()),
            "action_loss": float(inj_losses[:, 2].mean()),
            "total_chg_pct": float((inj_losses[:, 0].mean() / bl[:, 0].mean() - 1) * 100),
            "dist_chg_pct": float((inj_losses[:, 1].mean() / bl[:, 1].mean() - 1) * 100),
            "action_chg_pct": float((inj_losses[:, 2].mean() / bl[:, 2].mean() - 1) * 100),
        }

    # Save summary JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {OUTPUT_DIR / 'summary.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
