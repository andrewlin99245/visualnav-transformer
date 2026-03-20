"""
SIRA-style correlation experiment for ViNT.

Computes source-dataset steering vectors from per-sample action MSE, then
evaluates cosine-vs-loss correlation on either the same dataset or a transfer
dataset using hooked ViNT transformer layer outputs.

Usage:
  python experiments/sira_correlation.py \
    --checkpoint deployment/model_weights/vint.pth \
    --vector-data-folder /path/to/go_stanford \
    --vector-split-folder /path/to/data_splits/go_stanford/test \
    --vector-dataset-name go_stanford \
    --eval-data-folder /path/to/scand \
    --eval-split-folder /path/to/data_splits/scand/test \
    --eval-dataset-name scand \
    --max-samples 512 \
    --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "train"))

from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.models.vint.vint import ViNT


IMAGE_NORMALIZE = transforms.Compose(
    [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


class SIRAHook:
    """Capture transformer layer outputs without changing forward behavior."""

    def __init__(self) -> None:
        self.captured: Dict[int, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def install(self, layers: Sequence[nn.Module]) -> None:
        for i, layer in enumerate(layers):
            def make_hook(idx: int):
                def hook_fn(module, inputs, output):
                    if output.requires_grad:
                        output.retain_grad()
                    self.captured[idx] = output
                return hook_fn

            self._handles.append(layer.register_forward_hook(make_hook(i)))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.captured.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIRA-style correlation experiment for ViNT")
    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--vector-data-folder", required=True)
    parser.add_argument("--vector-split-folder", required=True)
    parser.add_argument("--vector-dataset-name", required=True)

    parser.add_argument("--eval-data-folder", default=None)
    parser.add_argument("--eval-split-folder", default=None)
    parser.add_argument("--eval-dataset-name", default=None)

    parser.add_argument("--max-samples", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "experiments" / "results" / "sira_correlation"))

    parser.add_argument("--context-size", type=int, default=5)
    parser.add_argument("--len-traj-pred", type=int, default=5)
    parser.add_argument("--learn-angle", action="store_true", default=True)
    parser.add_argument("--obs-encoder", default="efficientnet-b0")
    parser.add_argument("--obs-encoding-size", type=int, default=512)
    parser.add_argument("--late-fusion", action="store_true")
    parser.add_argument("--mha-num-attention-heads", type=int, default=4)
    parser.add_argument("--mha-num-attention-layers", type=int, default=4)
    parser.add_argument("--mha-ff-dim-factor", type=int, default=4)

    parser.add_argument("--image-width", type=int, default=85)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--waypoint-spacing", type=int, default=1)
    parser.add_argument("--min-dist-cat", type=int, default=0)
    parser.add_argument("--max-dist-cat", type=int, default=20)
    parser.add_argument("--min-action-distance", type=int, default=0)
    parser.add_argument("--max-action-distance", type=int, default=10)
    parser.add_argument("--negative-mining", action="store_true", default=True)
    parser.add_argument("--goals-per-obs", type=int, default=1)
    parser.add_argument("--end-slack", type=int, default=0)
    parser.add_argument("--context-type", default="temporal")
    parser.add_argument("--goal-type", default="image")
    parser.add_argument("--normalize", action="store_true", default=True)
    return parser.parse_args()


def make_device(user_value: str | None) -> torch.device:
    if user_value:
        return torch.device(user_value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_vint_model(args: argparse.Namespace, device: torch.device) -> ViNT:
    model = ViNT(
        context_size=args.context_size,
        len_traj_pred=args.len_traj_pred,
        learn_angle=args.learn_angle,
        obs_encoder=args.obs_encoder,
        obs_encoding_size=args.obs_encoding_size,
        late_fusion=args.late_fusion,
        mha_num_attention_heads=args.mha_num_attention_heads,
        mha_num_attention_layers=args.mha_num_attention_layers,
        mha_ff_dim_factor=args.mha_ff_dim_factor,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    loaded_model = checkpoint["model"]
    try:
        state_dict = loaded_model.module.state_dict()
    except AttributeError:
        state_dict = loaded_model.state_dict()
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def build_dataset(data_folder: str, split_folder: str, dataset_name: str, args: argparse.Namespace) -> ViNT_Dataset:
    return ViNT_Dataset(
        data_folder=data_folder,
        data_split_folder=split_folder,
        dataset_name=dataset_name,
        image_size=(args.image_width, args.image_height),
        waypoint_spacing=args.waypoint_spacing,
        min_dist_cat=args.min_dist_cat,
        max_dist_cat=args.max_dist_cat,
        min_action_distance=args.min_action_distance,
        max_action_distance=args.max_action_distance,
        negative_mining=args.negative_mining,
        len_traj_pred=args.len_traj_pred,
        learn_angle=args.learn_angle,
        context_size=args.context_size,
        context_type=args.context_type,
        end_slack=args.end_slack,
        goals_per_obs=args.goals_per_obs,
        normalize=args.normalize,
        goal_type=args.goal_type,
    )


def infer_layer_indices(num_layers: int, requested: str) -> List[int]:
    if requested == "all":
        return list(range(num_layers))
    indices = [int(item.strip()) for item in requested.split(",") if item.strip()]
    for idx in indices:
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {idx} out of range for {num_layers} layers")
    return indices


def snapshot_dataset(dataset: ViNT_Dataset, max_samples: int, seed: int) -> List[Tuple[torch.Tensor, ...]]:
    rng = np.random.RandomState(seed)
    n = len(dataset) if max_samples <= 0 else min(len(dataset), max_samples)
    indices = rng.choice(len(dataset), size=n, replace=False)
    cached = []
    for idx in tqdm(indices.tolist(), desc="snapshot-dataset"):
        cached.append(dataset[idx])
    return cached


def iterate_batches(samples: List[Tuple[torch.Tensor, ...]], batch_size: int) -> Iterable[Tuple[torch.Tensor, ...]]:
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        fields = list(zip(*batch_samples))
        yield tuple(torch.stack(list(items), dim=0) for items in fields)


def preprocess_obs_goal(obs_image: torch.Tensor, goal_image: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    obs_images = torch.split(obs_image, 3, dim=1)
    obs_images = [IMAGE_NORMALIZE(img).to(device) for img in obs_images]
    obs_image = torch.cat(obs_images, dim=1)
    goal_image = IMAGE_NORMALIZE(goal_image).to(device)
    return obs_image, goal_image


def flatten_hidden(hidden: torch.Tensor) -> torch.Tensor:
    return hidden.reshape(hidden.shape[0], -1).to(torch.float32)


def collect_losses_and_hiddens(
    model: ViNT,
    samples: List[Tuple[torch.Tensor, ...]],
    device: torch.device,
    batch_size: int,
    layer_indices: List[int],
) -> tuple[Dict[int, torch.Tensor], np.ndarray]:
    hook = SIRAHook()
    hook.install(model.decoder.sa_decoder.layers)

    all_hiddens = {i: [] for i in layer_indices}
    all_losses = []

    with torch.no_grad():
        total = (len(samples) + batch_size - 1) // batch_size
        for batch in tqdm(iterate_batches(samples, batch_size), total=total, desc="collect-loss-hiddens"):
            (
                obs_image,
                goal_image,
                action_label,
                _dist_label,
                _goal_pos,
                _dataset_index,
                action_mask,
            ) = batch

            obs_image, goal_image = preprocess_obs_goal(obs_image, goal_image, device)
            action_label = action_label.to(device)
            action_mask = action_mask.to(device)

            _dist_pred, action_pred, _final_repr = model(obs_image, goal_image)
            batch_losses = F.mse_loss(action_pred, action_label, reduction="none")
            while batch_losses.dim() > 1:
                batch_losses = batch_losses.mean(dim=-1)

            valid_mask = action_mask > 0.5
            if not valid_mask.any():
                continue

            all_losses.append(batch_losses[valid_mask].detach().cpu())

            for layer_idx in layer_indices:
                hidden = hook.captured[layer_idx].detach().cpu()
                all_hiddens[layer_idx].append(flatten_hidden(hidden[valid_mask.cpu()]).to(torch.float16))

    hook.remove()

    losses = torch.cat(all_losses, dim=0).numpy()
    for layer_idx in layer_indices:
        all_hiddens[layer_idx] = torch.cat(all_hiddens[layer_idx], dim=0)
    return all_hiddens, losses


def compute_steering_vectors_from_hiddens(
    all_hiddens: Dict[int, torch.Tensor],
    losses: np.ndarray,
    layer_indices: List[int],
) -> tuple[Dict[int, torch.Tensor], int]:
    losses_t = torch.from_numpy(losses)
    sorted_indices = torch.argsort(losses_t)
    q = max(1, len(sorted_indices) // 4)
    good_indices = sorted_indices[:q]
    bad_indices = sorted_indices[len(sorted_indices) - q :]

    vectors = {}
    for layer_idx in layer_indices:
        h = all_hiddens[layer_idx].to(torch.float32)
        good_mean = h[good_indices].mean(dim=0)
        bad_mean = h[bad_indices].mean(dim=0)
        v = good_mean - bad_mean
        v_normalized = v / (v.norm() + 1e-12)
        vectors[layer_idx] = v_normalized
        cos_sim = F.cosine_similarity(good_mean.unsqueeze(0), bad_mean.unsqueeze(0)).item()
        print(f"  SIRA L{layer_idx}: raw_norm={v.norm().item():.2f}, cos_sim={cos_sim:.4f}")

    print(f"  SIRA: {q} good, {q} bad samples (middle {len(sorted_indices) - 2*q} ignored) from {len(losses_t)} valid")
    return vectors, q


def compute_cosine_alignments_from_hiddens(
    all_hiddens: Dict[int, torch.Tensor],
    layer_indices: List[int],
    steering_vectors: Dict[int, torch.Tensor],
) -> Dict[int, np.ndarray]:
    alignments = {}
    for layer_idx in layer_indices:
        hidden = F.normalize(all_hiddens[layer_idx].to(torch.float32), dim=1)
        steering = steering_vectors[layer_idx].unsqueeze(0)
        cosine_values = torch.matmul(hidden, steering.T).squeeze(1).numpy()
        alignments[layer_idx] = cosine_values.astype(np.float64, copy=False)
    return alignments


def summarize_correlations(losses: np.ndarray, alignments: Dict[int, np.ndarray]) -> List[dict]:
    rows = []
    for layer_idx, cos_values in alignments.items():
        valid = np.isfinite(losses) & np.isfinite(cos_values)
        spearman = spearmanr(losses[valid], cos_values[valid])
        kendall = kendalltau(losses[valid], cos_values[valid])
        rows.append(
            {
                "layer_index": layer_idx,
                "num_samples": int(valid.sum()),
                "spearman_rho": float(spearman.statistic),
                "spearman_pvalue": float(spearman.pvalue),
                "kendall_tau": float(kendall.statistic),
                "kendall_pvalue": float(kendall.pvalue),
            }
        )
    rows.sort(key=lambda row: row["spearman_rho"])
    return rows


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.eval_data_folder is None:
        args.eval_data_folder = args.vector_data_folder
    if args.eval_split_folder is None:
        args.eval_split_folder = args.vector_split_folder
    if args.eval_dataset_name is None:
        args.eval_dataset_name = args.vector_dataset_name

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = make_device(args.device)
    model = load_vint_model(args, device)
    hook_layers = list(model.decoder.sa_decoder.layers)
    hook_layer_names = [f"decoder.sa_decoder.layers.{i}" for i in range(len(hook_layers))]
    layer_indices = infer_layer_indices(len(hook_layers), args.layers)

    print("SIRA hook layers:")
    for idx in layer_indices:
        print(f"  [{idx}] {hook_layer_names[idx]}")

    same_dataset = (
        args.vector_data_folder == args.eval_data_folder
        and args.vector_split_folder == args.eval_split_folder
        and args.vector_dataset_name == args.eval_dataset_name
    )

    vector_dataset = build_dataset(args.vector_data_folder, args.vector_split_folder, args.vector_dataset_name, args)
    vector_samples = snapshot_dataset(vector_dataset, args.max_samples, args.seed)

    if same_dataset:
        eval_dataset = vector_dataset
        eval_samples = vector_samples
    else:
        eval_dataset = build_dataset(args.eval_data_folder, args.eval_split_folder, args.eval_dataset_name, args)
        eval_samples = snapshot_dataset(eval_dataset, args.max_samples, args.seed)

    vector_hiddens, vector_losses = collect_losses_and_hiddens(
        model=model,
        samples=vector_samples,
        device=device,
        batch_size=args.batch_size,
        layer_indices=layer_indices,
    )
    steering_vectors, quartile = compute_steering_vectors_from_hiddens(vector_hiddens, vector_losses, layer_indices)

    if same_dataset:
        eval_hiddens, eval_losses = vector_hiddens, vector_losses
    else:
        eval_hiddens, eval_losses = collect_losses_and_hiddens(
            model=model,
            samples=eval_samples,
            device=device,
            batch_size=args.batch_size,
            layer_indices=layer_indices,
        )

    alignments = compute_cosine_alignments_from_hiddens(eval_hiddens, layer_indices, steering_vectors)
    summary_rows = summarize_correlations(eval_losses, alignments)

    per_sample = pd.DataFrame({"sample_index": np.arange(len(eval_losses)), "loss": eval_losses})
    for layer_idx in layer_indices:
        per_sample[f"cosine_layer_{layer_idx}"] = alignments[layer_idx]
    per_sample.to_csv(output_dir / "per_sample.csv", index=False)

    summary = {
        "checkpoint": args.checkpoint,
        "vector_dataset": {
            "dataset_name": args.vector_dataset_name,
            "data_folder": args.vector_data_folder,
            "split_folder": args.vector_split_folder,
        },
        "eval_dataset": {
            "dataset_name": args.eval_dataset_name,
            "data_folder": args.eval_data_folder,
            "split_folder": args.eval_split_folder,
        },
        "num_vector_samples": len(vector_samples),
        "num_eval_samples": len(eval_samples),
        "hook_layer_names": hook_layer_names,
        "selected_layers": layer_indices,
        "good_quartile_size": quartile,
        "bad_quartile_size": quartile,
        "correlations": summary_rows,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
