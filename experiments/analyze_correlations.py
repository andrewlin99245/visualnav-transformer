"""
Analyze correlations from stored residual streams.
Computes Spearman/Kendall for all 4 combinations {stanford, scand}^2,
comparing action-only vs total-loss partitioning for steering vectors.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "experiments" / "results" / "steering_injection"
NUM_LAYERS = 4
PARTITION_FRAC = 0.25


def load_data(name):
    d = OUTPUT_DIR / name
    losses = np.load(d / "losses.npy")  # [N, 3]: total, dist, action
    hiddens = {}
    for li in range(NUM_LAYERS):
        hiddens[li] = np.load(d / f"hiddens_L{li}.npy")  # [N, 3584] float16
    return losses, hiddens


def compute_vectors(losses, hiddens, partition_col, label=""):
    """Compute steering vectors by partitioning on losses[:, partition_col]."""
    sort_losses = losses[:, partition_col]
    n = len(sort_losses)
    n_part = max(1, int(n * PARTITION_FRAC))
    sorted_idx = np.argsort(sort_losses)
    pos_idx = sorted_idx[:n_part]
    neg_idx = sorted_idx[-n_part:]

    vectors = {}
    for li in range(NUM_LAYERS):
        h = torch.from_numpy(hiddens[li]).float()
        v = h[pos_idx].mean(dim=0) - h[neg_idx].mean(dim=0)
        vectors[li] = v
    return vectors


def eval_correlations(vectors, losses, hiddens, label=""):
    """Compute Spearman/Kendall between cos_sim and losses for layer 3."""
    li = 3  # use last layer (all layers ~identical)
    h = torch.from_numpy(hiddens[li]).float()
    v = vectors[li].unsqueeze(0)
    cos_sims = F.cosine_similarity(h, v, dim=1).numpy()

    total_sp, _ = spearmanr(cos_sims, losses[:, 0])
    total_kt, _ = kendalltau(cos_sims, losses[:, 0])
    dist_sp, _ = spearmanr(cos_sims, losses[:, 1])
    dist_kt, _ = kendalltau(cos_sims, losses[:, 1])
    action_sp, _ = spearmanr(cos_sims, losses[:, 2])
    action_kt, _ = kendalltau(cos_sims, losses[:, 2])

    print(f"  {label}")
    print(f"    Total:  ρ={total_sp:+.4f}  τ={total_kt:+.4f}")
    print(f"    Dist:   ρ={dist_sp:+.4f}  τ={dist_kt:+.4f}")
    print(f"    Action: ρ={action_sp:+.4f}  τ={action_kt:+.4f}")
    return {"total_sp": total_sp, "dist_sp": dist_sp, "action_sp": action_sp,
            "total_kt": total_kt, "dist_kt": dist_kt, "action_kt": action_kt}


def main():
    print("Loading data...")
    gs_losses, gs_hiddens = load_data("go_stanford")
    sc_losses, sc_hiddens = load_data("scand")
    print(f"  go_stanford: {len(gs_losses)} samples")
    print(f"  scand: {len(sc_losses)} samples")

    datasets = {
        "stanford": (gs_losses, gs_hiddens),
        "scand": (sc_losses, sc_hiddens),
    }

    for partition_name, partition_col in [("TOTAL LOSS", 0), ("DIST ONLY", 1), ("ACTION ONLY", 2)]:
        print(f"\n{'='*60}")
        print(f"Partitioning by: {partition_name}")
        print(f"{'='*60}")

        # Compute vectors from each source
        gs_vectors = compute_vectors(gs_losses, gs_hiddens, partition_col)
        sc_vectors = compute_vectors(sc_losses, sc_hiddens, partition_col)

        # Averaged vector: normalize both then average
        avg_vectors = {}
        for li in range(NUM_LAYERS):
            gs_norm = gs_vectors[li] / gs_vectors[li].norm()
            sc_norm = sc_vectors[li] / sc_vectors[li].norm()
            avg_vectors[li] = (gs_norm + sc_norm) / 2

        vectors_map = {"stanford": gs_vectors, "scand": sc_vectors, "averaged": avg_vectors}

        for vec_src in ["stanford", "scand", "averaged"]:
            for eval_dst in ["stanford", "scand"]:
                losses, hiddens = datasets[eval_dst]
                label = f"vector={vec_src} → eval={eval_dst}"
                eval_correlations(vectors_map[vec_src], losses, hiddens, label=label)


if __name__ == "__main__":
    main()
