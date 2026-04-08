"""
Cross-dataset steering evaluation with mist_bags.

Runs 5 injection combinations:
  1. gs_on_mist       (go_stanford vector → mist data)
  2. sc_on_mist       (scand vector → mist data)
  3. mist_on_mist     (mist vector → mist data)
  4. mist_on_stanford  (mist vector → go_stanford data)
  5. mist_on_scand     (mist vector → scand data)

Usage:
  cd experiments && python eval_mist.py --alpha 0.1
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "train"))

from steering_injection import (
    make_dataset, load_vint_model, run_inference,
    MultiLayerCapture, SteeringInjector, compute_steering_vectors,
)

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "steering_injection"


def compute_mean_norms(hiddens_dict, num_layers):
    norms = {}
    for li in range(num_layers):
        h = torch.from_numpy(hiddens_dict[f"hiddens_L{li}"]).float()
        h = h.reshape(h.shape[0], 7, 512)
        norms[li] = h.norm(dim=-1).mean().item()
    return norms


def run_injection(model, vectors, dataset, indices, device, num_layers,
                  h_norms, alpha, label, mws):
    injector = SteeringInjector(vectors, alpha=alpha, h_mean_norms=h_norms)
    injector.install(model.decoder.sa_decoder.layers)
    result = run_inference(model, dataset, indices, device, num_layers,
                           injector=injector, save_hiddens=False, label=label,
                           metric_waypoint_spacing=mws)
    injector.remove()
    return result["losses"], result["action_masks"]


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset steering evaluation")
    parser.add_argument("--checkpoint", type=str,
                        default=str(PROJECT_ROOT / "deployment" / "model_weights" / "vint.pth"))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
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

    def get_indices(dataset, label):
        n = len(dataset)
        if args.max_samples and args.max_samples < n:
            rng = np.random.RandomState(args.seed)
            idx = rng.choice(n, args.max_samples, replace=False)
            print(f"  [{label}] Subsampled {args.max_samples} from {n}")
        else:
            idx = np.arange(n)
            print(f"  [{label}] Using all {n} samples")
        return idx

    # ===== Load datasets =====
    print("\n=== Loading datasets ===")
    mist_dataset = make_dataset(
        str(PROJECT_ROOT / "datasets" / "mist_bags_converted"),
        str(PROJECT_ROOT / "train" / "vint_train" / "data" / "data_splits" / "mist_bags" / "train"),
        "mist_bags")
    gs_dataset = make_dataset(
        str(PROJECT_ROOT / "datasets" / "go_stanford_cropped" / "go_stanford"),
        str(PROJECT_ROOT / "train" / "vint_train" / "data" / "data_splits" / "go_stanford" / "train"),
        "go_stanford")
    sc_dataset = make_dataset(
        str(PROJECT_ROOT / "datasets" / "scand"),
        str(PROJECT_ROOT / "train" / "vint_train" / "data" / "data_splits" / "scand" / "train"),
        "scand")

    mist_indices = get_indices(mist_dataset, "mist")

    # ===== Phase 1: Mist baseline with hidden capture =====
    print("\n=== Phase 1: Baseline mist_bags ===")
    hook = MultiLayerCapture()
    hook.install(model.decoder.sa_decoder.layers)
    mist_result = run_inference(model, mist_dataset, mist_indices, device, num_layers,
                                hook=hook, save_hiddens=True, label="mist",
                                metric_waypoint_spacing=0.12)
    hook.remove()

    mist_losses = mist_result["losses"]
    mist_am = mist_result["action_masks"]
    mist_dl = mist_result["dist_labels"]
    mist_hiddens = {k: v for k, v in mist_result.items() if k.startswith("hiddens_")}
    mist_h_norms = compute_mean_norms(mist_hiddens, num_layers)
    del mist_result

    # ===== Phase 2: Compute mist steering vectors =====
    print("\n=== Phase 2: Computing mist steering vectors (dist-cond total) ===")
    mist_vectors = compute_steering_vectors(
        mist_losses, mist_hiddens, num_layers, partition_frac=0.25,
        action_masks=mist_am, partition_col=-2, dist_labels=mist_dl)
    del mist_hiddens

    # ===== Phase 3: Load stored gs/sc vectors =====
    vec_dir = sorted(OUTPUT_DIR.glob("vectors_*"))[-1]
    print(f"\n=== Phase 3: Loading stored vectors from {vec_dir} ===")
    gs_vectors = {}
    sc_vectors = {}
    for li in range(num_layers):
        gs_vectors[li] = torch.load(vec_dir / f"gs_vector_L{li}.pt", weights_only=True)
        sc_vectors[li] = torch.load(vec_dir / f"sc_vector_L{li}.pt", weights_only=True)
    print(f"  Loaded gs and sc vectors")

    # ===== Spearman: mist vector on stored stanford/scand data =====
    from scipy.stats import spearmanr
    print("\n=== Spearman: mist vector on stored stanford/scand (layer 3) ===")
    li = 3
    for data_name in ["go_stanford", "scand"]:
        stored_losses = np.load(OUTPUT_DIR / data_name / "losses.npy")
        stored_am = np.load(OUTPUT_DIR / data_name / "action_masks.npy")
        stored_dl = np.load(OUTPUT_DIR / data_name / "dist_labels.npy")
        stored_h = np.load(OUTPUT_DIR / data_name / f"hiddens_L{li}.npy")

        valid = stored_am > 0
        h = torch.from_numpy(stored_h[valid]).float()
        v = mist_vectors[li].unsqueeze(0)
        cos_sims = F.cosine_similarity(h, v, dim=1).numpy()
        vl = stored_losses[valid]
        vdl = stored_dl[valid]

        rho_t, _ = spearmanr(cos_sims, vl[:, 0])
        rho_d, _ = spearmanr(cos_sims, vl[:, 1])
        rho_a, _ = spearmanr(cos_sims, vl[:, 2])
        print(f"  vec=mist -> data={data_name} (global): rho(total)={rho_t:+.4f}  rho(dist)={rho_d:+.4f}  rho(action)={rho_a:+.4f}")

        unique_dists = np.unique(vdl)
        bucket_rhos = {"total": [], "dist": [], "action": []}
        for d in unique_dists:
            mask = vdl == d
            if mask.sum() < 5:
                continue
            cs = cos_sims[mask]
            ls = vl[mask]
            r_t, _ = spearmanr(cs, ls[:, 0])
            r_d, _ = spearmanr(cs, ls[:, 1])
            r_a, _ = spearmanr(cs, ls[:, 2])
            bucket_rhos["total"].append(r_t)
            bucket_rhos["dist"].append(r_d)
            bucket_rhos["action"].append(r_a)
        avg_t = np.mean(bucket_rhos["total"]) if bucket_rhos["total"] else float('nan')
        avg_d = np.mean(bucket_rhos["dist"]) if bucket_rhos["dist"] else float('nan')
        avg_a = np.mean(bucket_rhos["action"]) if bucket_rhos["action"] else float('nan')
        print(f"  vec=mist -> data={data_name} (dist-cond): rho(total)={avg_t:+.4f}  rho(dist)={avg_d:+.4f}  rho(action)={avg_a:+.4f}  ({len(bucket_rhos['total'])} buckets)")
        del stored_h

    # ===== Phase 4: Stanford and scand baselines =====
    # Load stored baselines and hidden norms for stanford/scand
    print("\n=== Phase 4: Loading stored stanford/scand baselines ===")
    gs_losses = np.load(OUTPUT_DIR / "go_stanford" / "losses.npy")
    gs_am = np.load(OUTPUT_DIR / "go_stanford" / "action_masks.npy")
    gs_h_norms = {}
    for li in range(num_layers):
        h = np.load(OUTPUT_DIR / "go_stanford" / f"hiddens_L{li}.npy")
        h_t = torch.from_numpy(h).float().reshape(h.shape[0], 7, 512)
        gs_h_norms[li] = h_t.norm(dim=-1).mean().item()
        del h, h_t

    sc_losses = np.load(OUTPUT_DIR / "scand" / "losses.npy")
    sc_am = np.load(OUTPUT_DIR / "scand" / "action_masks.npy")
    sc_h_norms = {}
    for li in range(num_layers):
        h = np.load(OUTPUT_DIR / "scand" / f"hiddens_L{li}.npy")
        h_t = torch.from_numpy(h).float().reshape(h.shape[0], 7, 512)
        sc_h_norms[li] = h_t.norm(dim=-1).mean().item()
        del h, h_t

    gs_indices = np.load(OUTPUT_DIR / "go_stanford" / "indices.npy")
    sc_indices = np.load(OUTPUT_DIR / "scand" / "indices.npy")
    print(f"  stanford: {len(gs_losses)} samples, scand: {len(sc_losses)} samples")

    # ===== Phase 5: All injection runs =====
    results = {}

    print(f"\n=== Injections on mist (alpha={args.alpha}) ===")
    for name, vectors in [("gs_on_mist", gs_vectors), ("sc_on_mist", sc_vectors), ("mist_on_mist", mist_vectors)]:
        losses, _ = run_injection(model, vectors, mist_dataset, mist_indices, device,
                                  num_layers, mist_h_norms, args.alpha, name, 0.12)
        results[name] = losses

    print(f"\n=== Injections on stanford (alpha={args.alpha}) ===")
    losses, _ = run_injection(model, mist_vectors, gs_dataset, gs_indices, device,
                              num_layers, gs_h_norms, args.alpha, "mist_on_stanford", 0.12)
    results["mist_on_stanford"] = losses

    print(f"\n=== Injections on scand (alpha={args.alpha}) ===")
    losses, _ = run_injection(model, mist_vectors, sc_dataset, sc_indices, device,
                              num_layers, sc_h_norms, args.alpha, "mist_on_scand", 0.38)
    results["mist_on_scand"] = losses

    # ===== Summary =====
    mist_valid = mist_am > 0
    gs_valid = gs_am > 0
    sc_valid = sc_am > 0

    mist_bl = mist_losses[mist_valid]
    gs_bl = gs_losses[gs_valid]
    sc_bl = sc_losses[sc_valid]

    print("\n" + "=" * 70)
    print(f"SUMMARY (alpha={args.alpha}, valid samples only)")
    print("=" * 70)

    print(f"\nBaseline mist ({mist_valid.sum()} valid):     Total={mist_bl[:, 0].mean():.6f}  Dist={mist_bl[:, 1].mean():.4f}  Action={mist_bl[:, 2].mean():.6f}")
    print(f"Baseline stanford ({gs_valid.sum()} valid): Total={gs_bl[:, 0].mean():.6f}  Dist={gs_bl[:, 1].mean():.4f}  Action={gs_bl[:, 2].mean():.6f}")
    print(f"Baseline scand ({sc_valid.sum()} valid):    Total={sc_bl[:, 0].mean():.6f}  Dist={sc_bl[:, 1].mean():.4f}  Action={sc_bl[:, 2].mean():.6f}")

    print(f"\n{'Setting':<25} {'Total Chg':<14} {'Dist Chg':<14} {'Action Chg':<14}")
    print(f"{'-'*25} {'-'*14} {'-'*14} {'-'*14}")

    for name in ["gs_on_mist", "sc_on_mist", "mist_on_mist", "mist_on_stanford", "mist_on_scand"]:
        inj = results[name]
        if name.endswith("_mist"):
            valid_mask, bl = mist_valid, mist_bl
        elif name.endswith("_stanford"):
            valid_mask, bl = gs_valid, gs_bl
        elif name.endswith("_scand"):
            valid_mask, bl = sc_valid, sc_bl
        il = inj[valid_mask]
        total_chg = (il[:, 0].mean() / bl[:, 0].mean() - 1) * 100
        dist_chg = (il[:, 1].mean() / bl[:, 1].mean() - 1) * 100
        action_chg = (il[:, 2].mean() / bl[:, 2].mean() - 1) * 100
        print(f"{name:<25} {total_chg:+.1f}%{'':<8} {dist_chg:+.1f}%{'':<8} {action_chg:+.1f}%")

    # ===== Save results =====
    import json
    save_dir = OUTPUT_DIR / "eval_mist"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save mist baseline
    np.save(save_dir / "mist_baseline_losses.npy", mist_losses)
    np.save(save_dir / "mist_action_masks.npy", mist_am)
    np.save(save_dir / "mist_dist_labels.npy", mist_dl)

    # Save mist steering vectors
    for li in range(num_layers):
        torch.save(mist_vectors[li], save_dir / f"mist_vector_L{li}.pt")

    # Save injection losses
    for name, inj_losses in results.items():
        np.save(save_dir / f"{name}_losses.npy", inj_losses)

    # Save summary JSON
    summary = {
        "config": {"alpha": args.alpha, "max_samples": args.max_samples, "seed": args.seed},
        "baselines": {
            "mist": {"total": float(mist_bl[:, 0].mean()), "dist": float(mist_bl[:, 1].mean()),
                      "action": float(mist_bl[:, 2].mean()), "n_valid": int(mist_valid.sum())},
            "stanford": {"total": float(gs_bl[:, 0].mean()), "dist": float(gs_bl[:, 1].mean()),
                          "action": float(gs_bl[:, 2].mean()), "n_valid": int(gs_valid.sum())},
            "scand": {"total": float(sc_bl[:, 0].mean()), "dist": float(sc_bl[:, 1].mean()),
                       "action": float(sc_bl[:, 2].mean()), "n_valid": int(sc_valid.sum())},
        },
        "injections": {},
    }
    for name in ["gs_on_mist", "sc_on_mist", "mist_on_mist", "mist_on_stanford", "mist_on_scand"]:
        inj = results[name]
        if name.endswith("_mist"):
            valid_mask, bl = mist_valid, mist_bl
        elif name.endswith("_stanford"):
            valid_mask, bl = gs_valid, gs_bl
        elif name.endswith("_scand"):
            valid_mask, bl = sc_valid, sc_bl
        il = inj[valid_mask]
        summary["injections"][name] = {
            "total": float(il[:, 0].mean()), "dist": float(il[:, 1].mean()), "action": float(il[:, 2].mean()),
            "total_chg": float((il[:, 0].mean() / bl[:, 0].mean() - 1) * 100),
            "dist_chg": float((il[:, 1].mean() / bl[:, 1].mean() - 1) * 100),
            "action_chg": float((il[:, 2].mean() / bl[:, 2].mean() - 1) * 100),
        }

    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved results to {save_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
