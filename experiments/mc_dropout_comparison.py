"""
Compare MC Dropout variance vs Steering Vector cos_sim as uncertainty signals.

Uses 500 random samples from go_stanford.
MC Dropout: N=10 forward passes, measure dist/action prediction variance.
Cos_sim: single forward pass hidden states (cached) dot steering vector.

Metrics: Spearman correlation with actual loss, selective prediction curves.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "train"))

from uncertainty_correlation import load_vint_model
from vint_train.data.vint_dataset import ViNT_Dataset
import yaml
import tqdm

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "steering_injection"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "mc_dropout_comparison"


def make_dataset(data_folder, split_folder, dataset_name):
    return ViNT_Dataset(
        data_folder=data_folder,
        data_split_folder=split_folder,
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


def compute_per_sample_loss(dist_pred, action_pred, dist_label, action_label, action_mask):
    dist_loss = (dist_pred.squeeze(-1) - dist_label.float()) ** 2
    action_loss = F.mse_loss(action_pred, action_label, reduction="none")
    while action_loss.dim() > 1:
        action_loss = action_loss.mean(dim=-1)
    action_loss = action_loss * action_mask
    total_loss = 0.5 * 1e-2 * dist_loss + 0.5 * action_loss
    return total_loss, dist_loss, action_loss


def run_mc_dropout(model, dataset, sample_indices, device, n_passes=10):
    """Run MC Dropout: N forward passes with dropout enabled, return variance."""
    dist_vars = []
    action_vars = []
    losses_list = []
    action_masks_list = []

    for idx in tqdm.tqdm(sample_indices, desc="MC Dropout"):
        obs_img, goal_img, action_label, dist_label, goal_pos, dataset_idx, action_mask = dataset[idx]
        obs = obs_img.unsqueeze(0).to(device)
        goal = goal_img.unsqueeze(0).to(device)
        action_label = action_label.unsqueeze(0).to(device)
        dist_label = dist_label.unsqueeze(0).to(device)
        action_mask = action_mask.item()

        # Collect N predictions with dropout enabled
        model.train()  # enable dropout
        dist_preds = []
        action_preds = []
        with torch.no_grad():
            for _ in range(n_passes):
                dist_pred, action_pred, _ = model(obs, goal)
                dist_preds.append(dist_pred.cpu())
                action_preds.append(action_pred.cpu())

        model.eval()

        dist_preds = torch.stack(dist_preds)  # [N, 1, 1]
        action_preds = torch.stack(action_preds)  # [N, 1, 5, 4]

        # Variance across passes
        dist_var = dist_preds.squeeze().var().item()
        action_var = action_preds.squeeze(1).var(dim=0).mean().item()

        dist_vars.append(dist_var)
        action_vars.append(action_var)

        # Compute actual loss using mean prediction
        dist_mean = dist_preds.mean(dim=0).to(device)
        action_mean = action_preds.mean(dim=0).to(device)
        total_loss, dist_loss, action_loss = compute_per_sample_loss(
            dist_mean, action_mean, dist_label, action_label, action_mask
        )
        losses_list.append([total_loss.item(), dist_loss.item(), action_loss.item()])
        action_masks_list.append(action_mask)

    return {
        "dist_var": np.array(dist_vars),
        "action_var": np.array(action_vars),
        "losses": np.array(losses_list),
        "action_masks": np.array(action_masks_list),
    }


def compute_cos_sim(hiddens_L3, steering_vector):
    """Compute cosine similarity between hidden states and steering vector."""
    v_flat = steering_vector.flatten().unsqueeze(0)
    h_flat = hiddens_L3.reshape(hiddens_L3.shape[0], -1)
    return F.cosine_similarity(h_flat, v_flat, dim=1).numpy()


def selective_prediction_curve(scores, losses, n_bins=20):
    """Reject bottom X% by score, return (rejection_rate, remaining_mean_loss)."""
    sorted_idx = np.argsort(scores)  # low score = reject first
    curve = []
    n = len(scores)
    for pct in np.linspace(0, 0.5, n_bins + 1):
        n_reject = int(pct * n)
        keep_idx = sorted_idx[n_reject:]
        if len(keep_idx) == 0:
            break
        curve.append((pct, losses[keep_idx].mean()))
    return curve


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-passes", type=int, default=10)
    parser.add_argument("--checkpoint", type=str,
                        default=str(PROJECT_ROOT / "deployment" / "model_weights" / "vint.pth"))
    parser.add_argument("--gs-data-folder", type=str,
                        default=str(PROJECT_ROOT / "datasets" / "go_stanford_cropped" / "go_stanford"))
    parser.add_argument("--gs-split-folder", type=str,
                        default=str(PROJECT_ROOT / "train" / "vint_train" / "data" / "data_splits" / "go_stanford" / "train"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}")

    # Load model
    model = load_vint_model(args.checkpoint, device)
    model.eval()

    # Load dataset
    print("Loading go_stanford dataset...")
    dataset = make_dataset(args.gs_data_folder, args.gs_split_folder, "go_stanford")

    # Select 500 random samples from stored valid indices
    stored_indices = np.load(RESULTS_DIR / "go_stanford" / "indices.npy")
    stored_masks = np.load(RESULTS_DIR / "go_stanford" / "action_masks.npy")
    valid_mask = stored_masks.astype(bool)
    valid_positions = np.where(valid_mask)[0]  # positions in stored array

    rng = np.random.RandomState(args.seed)
    chosen_positions = rng.choice(len(valid_positions), args.n_samples, replace=False)
    chosen_positions.sort()

    # Map back to dataset indices
    sample_stored_pos = valid_positions[chosen_positions]
    sample_dataset_idx = stored_indices[sample_stored_pos]

    print(f"Selected {args.n_samples} valid samples from go_stanford")

    # === MC Dropout ===
    print(f"\n=== MC Dropout (N={args.n_passes}) ===")
    mc_result = run_mc_dropout(model, dataset, sample_dataset_idx, device, n_passes=args.n_passes)

    # === Cos_sim from cached data ===
    print("\n=== Steering Vector Cos_sim (cached) ===")
    hiddens_L3 = np.load(RESULTS_DIR / "go_stanford" / "hiddens_L3.npy")
    gs_vec = torch.load(RESULTS_DIR / "vectors_pdist-cond-total" / "gs_vector_L3.pt",
                        weights_only=True).float()
    stored_losses = np.load(RESULTS_DIR / "go_stanford" / "losses.npy")

    # Extract same samples from cached data
    cached_hiddens = torch.from_numpy(hiddens_L3[sample_stored_pos]).float()
    cached_losses = stored_losses[sample_stored_pos]
    cos_sims = compute_cos_sim(cached_hiddens, gs_vec)

    # Use cached losses for cos_sim (same samples, but from baseline inference)
    # Use MC dropout losses for MC dropout (from MC mean predictions)
    # For fair comparison, correlate both signals against cached baseline losses
    baseline_total = cached_losses[:, 0]
    baseline_dist = cached_losses[:, 1]
    baseline_action = cached_losses[:, 2]

    # === Spearman Correlations ===
    print("\n" + "=" * 70)
    print("SPEARMAN CORRELATIONS (signal vs actual loss)")
    print("=" * 70)
    print(f"{'Signal':<25} {'vs Total':>12} {'vs Dist':>12} {'vs Action':>12}")
    print("-" * 70)

    # MC Dropout: higher variance should correlate with higher loss (positive rho)
    rho_mc_dist_total, _ = spearmanr(mc_result["dist_var"], baseline_total)
    rho_mc_dist_dist, _ = spearmanr(mc_result["dist_var"], baseline_dist)
    rho_mc_dist_action, _ = spearmanr(mc_result["dist_var"], baseline_action)
    print(f"{'MC Dropout (dist var)':<25} {rho_mc_dist_total:>12.4f} {rho_mc_dist_dist:>12.4f} {rho_mc_dist_action:>12.4f}")

    rho_mc_act_total, _ = spearmanr(mc_result["action_var"], baseline_total)
    rho_mc_act_dist, _ = spearmanr(mc_result["action_var"], baseline_dist)
    rho_mc_act_action, _ = spearmanr(mc_result["action_var"], baseline_action)
    print(f"{'MC Dropout (action var)':<25} {rho_mc_act_total:>12.4f} {rho_mc_act_dist:>12.4f} {rho_mc_act_action:>12.4f}")

    # Combined MC variance
    mc_combined = mc_result["dist_var"] / (mc_result["dist_var"].max() + 1e-12) + \
                  mc_result["action_var"] / (mc_result["action_var"].max() + 1e-12)
    rho_mc_comb_total, _ = spearmanr(mc_combined, baseline_total)
    rho_mc_comb_dist, _ = spearmanr(mc_combined, baseline_dist)
    rho_mc_comb_action, _ = spearmanr(mc_combined, baseline_action)
    print(f"{'MC Dropout (combined)':<25} {rho_mc_comb_total:>12.4f} {rho_mc_comb_dist:>12.4f} {rho_mc_comb_action:>12.4f}")

    # Cos_sim: lower cos_sim should correlate with higher loss (negative rho)
    rho_cos_total, _ = spearmanr(cos_sims, baseline_total)
    rho_cos_dist, _ = spearmanr(cos_sims, baseline_dist)
    rho_cos_action, _ = spearmanr(cos_sims, baseline_action)
    print(f"{'Cos_sim (steering L3)':<25} {rho_cos_total:>12.4f} {rho_cos_dist:>12.4f} {rho_cos_action:>12.4f}")

    # === Selective Prediction ===
    print("\n" + "=" * 70)
    print("SELECTIVE PREDICTION (reject bottom X%, report remaining total loss)")
    print("=" * 70)
    print(f"{'Reject %':<10} {'MC dist_var':>14} {'MC act_var':>14} {'MC combined':>14} {'Cos_sim':>14}")
    print("-" * 70)

    # For MC: reject high variance (low score = keep)
    # For cos_sim: reject low cos_sim (high score = keep)
    mc_dist_curve = selective_prediction_curve(-mc_result["dist_var"], baseline_total)
    mc_act_curve = selective_prediction_curve(-mc_result["action_var"], baseline_total)
    mc_comb_curve = selective_prediction_curve(-mc_combined, baseline_total)
    cos_curve = selective_prediction_curve(cos_sims, baseline_total)

    for i, pct in enumerate([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]):
        # Find closest entries
        def get_loss(curve, target_pct):
            for p, l in curve:
                if abs(p - target_pct) < 0.01:
                    return l
            return float('nan')

        mc_d = get_loss(mc_dist_curve, pct)
        mc_a = get_loss(mc_act_curve, pct)
        mc_c = get_loss(mc_comb_curve, pct)
        cs = get_loss(cos_curve, pct)
        print(f"{pct*100:>5.0f}%     {mc_d:>14.5f} {mc_a:>14.5f} {mc_c:>14.5f} {cs:>14.5f}")

    baseline_mean = baseline_total.mean()
    print(f"\nBaseline total loss (all samples): {baseline_mean:.5f}")
    print(f"N forward passes: MC={args.n_passes}, Cos_sim=1 (cached)")

    # === Save results ===
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_DIR / "results.npz",
             mc_dist_var=mc_result["dist_var"],
             mc_action_var=mc_result["action_var"],
             mc_combined=mc_combined,
             cos_sims=cos_sims,
             baseline_total=baseline_total,
             baseline_dist=baseline_dist,
             baseline_action=baseline_action,
             sample_stored_pos=sample_stored_pos,
             sample_dataset_idx=sample_dataset_idx)
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
