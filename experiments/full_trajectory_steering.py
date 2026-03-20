"""
Deterministic full-trajectory ViNT steering evaluation.

This script avoids the stochastic ViNT goal sampling path. It enumerates every
same-trajectory positive goal within the valid action-distance band, so every
example has a meaningful action target and no action mask is needed.

Outputs:
- baseline chunk files containing residual streams, model outputs, labels, and metadata
- steering vectors + baseline cosine/loss correlations
- steered chunk files for an all-layer intervention pass
- run summary comparing baseline vs steered action MSE
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "train"))

from vint_train.data.data_utils import calculate_sin_cos
from vint_train.data.vint_dataset import ViNT_Dataset
from vint_train.models.vint.vint import ViNT

try:
    from scipy.stats import kendalltau, spearmanr  # type: ignore
except ImportError:  # pragma: no cover - fallback for cluster envs missing scipy
    kendalltau = None
    spearmanr = None


IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


class DeterministicViNTDataset(ViNT_Dataset):
    """Enumerate all valid same-trajectory positive goals without sampling."""

    def _build_index(self, use_tqdm: bool = False):
        samples_index = []
        goals_index = []

        iterator = tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True, desc="build-index")
        for traj_name in iterator:
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])

            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size * self.waypoint_spacing
            end_time = traj_len - self.end_slack - self.len_traj_pred * self.waypoint_spacing
            for curr_time in range(begin_time, end_time):
                max_goal_cat = min(
                    self.max_dist_cat,
                    (traj_len - curr_time - 1) // self.waypoint_spacing,
                )
                lower = self.min_action_distance + 1
                upper = min(self.max_action_distance - 1, max_goal_cat)
                for dist_cat in range(lower, upper + 1):
                    goal_time = curr_time + dist_cat * self.waypoint_spacing
                    samples_index.append((traj_name, curr_time, goal_time, dist_cat))

        return samples_index, goals_index

    def __getitem__(self, i: int):
        traj_name, curr_time, goal_time, distance = self.index_to_data[i]

        context_times = list(
            range(
                curr_time - self.context_size * self.waypoint_spacing,
                curr_time + 1,
                self.waypoint_spacing,
            )
        )
        context = [(traj_name, t) for t in context_times]

        obs_image = torch.cat([self._load_image(f, t) for f, t in context])
        goal_image = self._load_image(traj_name, goal_time)

        curr_traj_data = self._get_trajectory(traj_name)
        actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)

        actions_torch = torch.as_tensor(actions, dtype=torch.float32)
        if self.learn_angle:
            actions_torch = calculate_sin_cos(actions_torch)

        return {
            "obs_image": torch.as_tensor(obs_image, dtype=torch.float32),
            "goal_image": torch.as_tensor(goal_image, dtype=torch.float32),
            "action_label": actions_torch,
            "dist_label": torch.as_tensor(distance, dtype=torch.int64),
            "goal_pos": torch.as_tensor(goal_pos, dtype=torch.float32),
            "dataset_index": torch.as_tensor(self.dataset_index, dtype=torch.int64),
            "trajectory_name": traj_name,
            "curr_time": curr_time,
            "goal_time": goal_time,
            "goal_offset": distance,
        }


class SteeringCapture:
    def __init__(
        self,
        layer_indices: Sequence[int],
        steering_vectors: Dict[int, torch.Tensor] | None = None,
        coeff: float = 0.0,
    ) -> None:
        self.layer_indices = set(layer_indices)
        self.steering_vectors = steering_vectors or {}
        self.coeff = coeff
        self.captured: Dict[int, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def install(self, layers: Sequence[nn.Module]) -> None:
        for idx, layer in enumerate(layers):
            if idx not in self.layer_indices:
                continue

            def make_hook(layer_idx: int):
                def hook_fn(module, inputs, output):
                    steered = output
                    if layer_idx in self.steering_vectors:
                        delta = self.steering_vectors[layer_idx].to(device=output.device, dtype=output.dtype)
                        steered = output + self.coeff * delta.unsqueeze(0)
                    self.captured[layer_idx] = steered.detach()
                    return steered

                return hook_fn

            self._handles.append(layer.register_forward_hook(make_hook(idx)))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.captured.clear()


class ChunkWriter:
    def __init__(self, root: Path, residual_dtype: torch.dtype, save_chunk_size: int) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.residual_dtype = residual_dtype
        self.save_chunk_size = save_chunk_size
        self.buffer: List[dict] = []
        self.example_count = 0
        self.chunk_idx = 0
        self.manifest: List[dict] = []

    def add(self, payload: dict) -> None:
        self.buffer.append(payload)
        buffered_examples = sum(item["num_examples"] for item in self.buffer)
        if buffered_examples >= self.save_chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return

        num_examples = sum(item["num_examples"] for item in self.buffer)
        file_path = self.root / f"chunk_{self.chunk_idx:05d}.pt"

        trajectories: List[str] = []
        tensors: Dict[str, List[torch.Tensor]] = {
            "example_index": [],
            "curr_time": [],
            "goal_time": [],
            "goal_offset": [],
            "dist_label": [],
            "dist_pred": [],
            "goal_pos": [],
            "action_label": [],
            "action_pred": [],
            "final_repr": [],
            "loss": [],
        }
        residuals: Dict[int, List[torch.Tensor]] = {}

        start_index = self.buffer[0]["start_index"]
        end_index = self.buffer[-1]["end_index"]

        for item in self.buffer:
            trajectories.extend(item["trajectory_name"])
            for key in tensors:
                tensors[key].append(item[key])
            for layer_idx, tensor in item["residuals"].items():
                residuals.setdefault(layer_idx, []).append(tensor.to(self.residual_dtype))

        payload = {
            "trajectory_name": trajectories,
            "residual_dtype": str(self.residual_dtype).replace("torch.", ""),
        }
        for key, values in tensors.items():
            payload[key] = torch.cat(values, dim=0)
        payload["residuals"] = {
            layer_idx: torch.cat(values, dim=0) for layer_idx, values in residuals.items()
        }

        torch.save(payload, file_path)
        self.manifest.append(
            {
                "file": file_path.name,
                "num_examples": num_examples,
                "start_index": int(start_index),
                "end_index": int(end_index),
            }
        )
        self.example_count += num_examples
        self.chunk_idx += 1
        self.buffer.clear()

    def finalize(self) -> None:
        self.flush()
        with open(self.root / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deterministic full-trajectory ViNT steering evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-folder", required=True)
    parser.add_argument("--split-folder", required=True)
    parser.add_argument("--dataset-name", default="go_stanford")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--coeff", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-chunk-size", type=int, default=2048)
    parser.add_argument("--max-examples", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--device", default=None)
    parser.add_argument("--residual-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")

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


def build_dataset(args: argparse.Namespace) -> DeterministicViNTDataset:
    return DeterministicViNTDataset(
        data_folder=args.data_folder,
        data_split_folder=args.split_folder,
        dataset_name=args.dataset_name,
        image_size=(args.image_width, args.image_height),
        waypoint_spacing=args.waypoint_spacing,
        min_dist_cat=args.min_dist_cat,
        max_dist_cat=args.max_dist_cat,
        min_action_distance=args.min_action_distance,
        max_action_distance=args.max_action_distance,
        negative_mining=False,
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
    indices = [int(part.strip()) for part in requested.split(",") if part.strip()]
    for idx in indices:
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {idx} out of range for {num_layers} layers")
    return indices


def torch_dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def normalize_images(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    image = image.to(device=device, dtype=torch.float32)
    chunks = torch.split(image, 3, dim=1)
    normalized = [(chunk - IMAGE_MEAN.to(device)) / IMAGE_STD.to(device) for chunk in chunks]
    return torch.cat(normalized, dim=1)


def normalize_goal(goal: torch.Tensor, device: torch.device) -> torch.Tensor:
    goal = goal.to(device=device, dtype=torch.float32)
    return (goal - IMAGE_MEAN.to(device)) / IMAGE_STD.to(device)


def collate_examples(examples: Sequence[dict]) -> dict:
    return {
        "obs_image": torch.stack([example["obs_image"] for example in examples], dim=0),
        "goal_image": torch.stack([example["goal_image"] for example in examples], dim=0),
        "action_label": torch.stack([example["action_label"] for example in examples], dim=0),
        "dist_label": torch.stack([example["dist_label"] for example in examples], dim=0),
        "goal_pos": torch.stack([example["goal_pos"] for example in examples], dim=0),
        "dataset_index": torch.stack([example["dataset_index"] for example in examples], dim=0),
        "trajectory_name": [example["trajectory_name"] for example in examples],
        "curr_time": torch.tensor([example["curr_time"] for example in examples], dtype=torch.int64),
        "goal_time": torch.tensor([example["goal_time"] for example in examples], dtype=torch.int64),
        "goal_offset": torch.tensor([example["goal_offset"] for example in examples], dtype=torch.int64),
    }


def iterate_dataset_batches(dataset: DeterministicViNTDataset, batch_size: int, max_examples: int) -> Iterable[dict]:
    limit = len(dataset) if max_examples <= 0 else min(len(dataset), max_examples)
    for start in range(0, limit, batch_size):
        stop = min(start + batch_size, limit)
        examples = [dataset[idx] for idx in range(start, stop)]
        batch = collate_examples(examples)
        batch["start_index"] = start
        batch["end_index"] = stop - 1
        yield batch


def compute_action_loss(action_pred: torch.Tensor, action_label: torch.Tensor) -> torch.Tensor:
    loss = F.mse_loss(action_pred, action_label, reduction="none")
    while loss.dim() > 1:
        loss = loss.mean(dim=-1)
    return loss


def save_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_pass(
    model: ViNT,
    dataset: DeterministicViNTDataset,
    device: torch.device,
    batch_size: int,
    layer_indices: List[int],
    output_dir: Path,
    residual_dtype: torch.dtype,
    save_chunk_size: int,
    steering_vectors: Dict[int, torch.Tensor] | None = None,
    coeff: float = 0.0,
    max_examples: int = 0,
) -> np.ndarray:
    hook = SteeringCapture(layer_indices, steering_vectors=steering_vectors, coeff=coeff)
    hook.install(model.decoder.sa_decoder.layers)
    writer = ChunkWriter(output_dir, residual_dtype=residual_dtype, save_chunk_size=save_chunk_size)
    losses: List[torch.Tensor] = []

    total_examples = len(dataset) if max_examples <= 0 else min(len(dataset), max_examples)
    total_batches = math.ceil(total_examples / batch_size)

    with torch.no_grad():
        for batch in tqdm(
            iterate_dataset_batches(dataset, batch_size=batch_size, max_examples=max_examples),
            total=total_batches,
            desc=f"pass-{output_dir.name}",
        ):
            obs_image = normalize_images(batch["obs_image"], device)
            goal_image = normalize_goal(batch["goal_image"], device)
            action_label = batch["action_label"].to(device)

            dist_pred, action_pred, final_repr = model(obs_image, goal_image)
            batch_loss = compute_action_loss(action_pred, action_label)
            losses.append(batch_loss.detach().cpu())

            residuals = {
                layer_idx: hook.captured[layer_idx].detach().cpu()
                for layer_idx in layer_indices
            }
            writer.add(
                {
                    "num_examples": action_pred.shape[0],
                    "start_index": batch["start_index"],
                    "end_index": batch["end_index"],
                    "trajectory_name": batch["trajectory_name"],
                    "example_index": torch.arange(batch["start_index"], batch["end_index"] + 1, dtype=torch.int64),
                    "curr_time": batch["curr_time"],
                    "goal_time": batch["goal_time"],
                    "goal_offset": batch["goal_offset"],
                    "dist_label": batch["dist_label"].cpu(),
                    "dist_pred": dist_pred.detach().cpu().to(torch.float32),
                    "goal_pos": batch["goal_pos"].cpu().to(torch.float32),
                    "action_label": batch["action_label"].cpu().to(torch.float32),
                    "action_pred": action_pred.detach().cpu().to(torch.float32),
                    "final_repr": final_repr.detach().cpu().to(torch.float32),
                    "loss": batch_loss.detach().cpu().to(torch.float32),
                    "residuals": residuals,
                }
            )

    hook.remove()
    writer.finalize()
    losses_np = torch.cat(losses, dim=0).numpy()
    np.save(output_dir / "losses.npy", losses_np)
    return losses_np


def load_manifest_chunks(run_dir: Path) -> List[Path]:
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return [run_dir / item["file"] for item in manifest]


def compute_steering_vectors(
    baseline_dir: Path,
    losses: np.ndarray,
    layer_indices: List[int],
) -> tuple[Dict[int, torch.Tensor], Dict[int, dict], int]:
    order = np.argsort(losses)
    q = max(1, len(order) // 4)
    good = set(order[:q].tolist())
    bad = set(order[-q:].tolist())

    sums_good: Dict[int, torch.Tensor] = {}
    sums_bad: Dict[int, torch.Tensor] = {}
    count_good = 0
    count_bad = 0

    for chunk_path in tqdm(load_manifest_chunks(baseline_dir), desc="compute-vectors"):
        chunk = torch.load(chunk_path, map_location="cpu")
        example_index = chunk["example_index"].tolist()
        for local_idx, global_idx in enumerate(example_index):
            target = None
            if global_idx in good:
                target = "good"
                count_good += 1
            elif global_idx in bad:
                target = "bad"
                count_bad += 1
            if target is None:
                continue
            for layer_idx in layer_indices:
                hidden = chunk["residuals"][layer_idx][local_idx].to(torch.float32)
                if target == "good":
                    sums_good[layer_idx] = sums_good.get(layer_idx, torch.zeros_like(hidden)) + hidden
                else:
                    sums_bad[layer_idx] = sums_bad.get(layer_idx, torch.zeros_like(hidden)) + hidden

    vectors: Dict[int, torch.Tensor] = {}
    stats: Dict[int, dict] = {}
    for layer_idx in layer_indices:
        good_mean = sums_good[layer_idx] / count_good
        bad_mean = sums_bad[layer_idx] / count_bad
        raw = good_mean - bad_mean
        normalized = raw / (raw.norm() + 1e-12)
        vectors[layer_idx] = raw
        cos_sim = F.cosine_similarity(good_mean.reshape(1, -1), bad_mean.reshape(1, -1)).item()
        stats[layer_idx] = {
            "raw_norm": float(raw.norm().item()),
            "good_bad_cosine": float(cos_sim),
        }
    return vectors, stats, q


def compute_alignments(
    baseline_dir: Path,
    vectors: Dict[int, torch.Tensor],
    layer_indices: List[int],
    num_examples: int,
) -> Dict[int, np.ndarray]:
    alignments = {layer_idx: np.empty(num_examples, dtype=np.float32) for layer_idx in layer_indices}
    normalized_vectors = {
        layer_idx: (vector / (vector.norm() + 1e-12)).reshape(-1).to(torch.float32)
        for layer_idx, vector in vectors.items()
    }

    for chunk_path in tqdm(load_manifest_chunks(baseline_dir), desc="compute-alignments"):
        chunk = torch.load(chunk_path, map_location="cpu")
        example_index = chunk["example_index"].numpy()
        for layer_idx in layer_indices:
            hidden = chunk["residuals"][layer_idx].to(torch.float32).reshape(chunk["residuals"][layer_idx].shape[0], -1)
            hidden = F.normalize(hidden, dim=1)
            cosine = torch.matmul(hidden, normalized_vectors[layer_idx].unsqueeze(1)).squeeze(1).numpy()
            alignments[layer_idx][example_index] = cosine
    return alignments


def summarize_correlations(losses: np.ndarray, alignments: Dict[int, np.ndarray]) -> List[dict]:
    rows = []
    for layer_idx, cosine_values in alignments.items():
        valid = np.isfinite(losses) & np.isfinite(cosine_values)
        if spearmanr is not None and kendalltau is not None:
            spearman = spearmanr(losses[valid], cosine_values[valid])
            kendall = kendalltau(losses[valid], cosine_values[valid])
            spearman_rho = float(spearman.statistic)
            spearman_pvalue = float(spearman.pvalue)
            kendall_tau_value = float(kendall.statistic)
            kendall_pvalue = float(kendall.pvalue)
        else:
            loss_series = pd.Series(losses[valid])
            cosine_series = pd.Series(cosine_values[valid])
            spearman_rho = float(loss_series.corr(cosine_series, method="spearman"))
            kendall_tau_value = float(loss_series.corr(cosine_series, method="kendall"))
            spearman_pvalue = float("nan")
            kendall_pvalue = float("nan")
        rows.append(
            {
                "layer_index": layer_idx,
                "num_samples": int(valid.sum()),
                "spearman_rho": spearman_rho,
                "spearman_pvalue": spearman_pvalue,
                "kendall_tau": kendall_tau_value,
                "kendall_pvalue": kendall_pvalue,
            }
        )
    rows.sort(key=lambda row: row["spearman_rho"])
    return rows


def write_alignment_csv(
    baseline_dir: Path,
    losses: np.ndarray,
    alignments: Dict[int, np.ndarray],
) -> None:
    frame = pd.DataFrame({"example_index": np.arange(len(losses)), "loss": losses})
    for layer_idx, cosine_values in alignments.items():
        frame[f"cosine_layer_{layer_idx}"] = cosine_values
    frame.to_csv(baseline_dir / "per_sample_alignment.csv", index=False)


def summarize_steering_effect(
    baseline_losses: np.ndarray,
    steered_losses: np.ndarray,
) -> dict:
    delta = steered_losses - baseline_losses
    baseline_avg = float(baseline_losses.mean())
    steered_avg = float(steered_losses.mean())
    avg_delta = float(delta.mean())
    return {
        "baseline_average_loss": baseline_avg,
        "steered_average_loss": steered_avg,
        "average_loss_delta": avg_delta,
        "relative_change": float(avg_delta / baseline_avg) if baseline_avg != 0 else 0.0,
        "improved_fraction": float((delta < 0).mean()),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = output_dir / "baseline"
    steered_dir = output_dir / "steered"

    device = make_device(args.device)
    residual_dtype = torch_dtype_from_name(args.residual_dtype)
    model = load_vint_model(args, device)

    dataset = build_dataset(args)
    total_examples = len(dataset) if args.max_examples <= 0 else min(len(dataset), args.max_examples)
    num_layers = len(model.decoder.sa_decoder.layers)
    layer_indices = infer_layer_indices(num_layers, args.layers)
    hook_layer_names = [f"decoder.sa_decoder.layers.{idx}" for idx in range(num_layers)]

    print("Steering hook layers:")
    for idx in layer_indices:
        print(f"  [{idx}] {hook_layer_names[idx]}")
    print(f"Dataset examples: {total_examples}")

    baseline_losses = run_pass(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        layer_indices=layer_indices,
        output_dir=baseline_dir,
        residual_dtype=residual_dtype,
        save_chunk_size=args.save_chunk_size,
        steering_vectors=None,
        coeff=0.0,
        max_examples=args.max_examples,
    )

    steering_vectors, vector_stats, quartile = compute_steering_vectors(
        baseline_dir=baseline_dir,
        losses=baseline_losses,
        layer_indices=layer_indices,
    )
    torch.save(
        {
            "raw_vectors": steering_vectors,
            "normalized_vectors": {
                layer_idx: vector / (vector.norm() + 1e-12) for layer_idx, vector in steering_vectors.items()
            },
            "vector_stats": vector_stats,
        },
        output_dir / "steering_vectors.pt",
    )

    alignments = compute_alignments(
        baseline_dir=baseline_dir,
        vectors=steering_vectors,
        layer_indices=layer_indices,
        num_examples=len(baseline_losses),
    )
    correlations = summarize_correlations(baseline_losses, alignments)
    write_alignment_csv(baseline_dir, baseline_losses, alignments)

    steered_losses = run_pass(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        layer_indices=layer_indices,
        output_dir=steered_dir,
        residual_dtype=residual_dtype,
        save_chunk_size=args.save_chunk_size,
        steering_vectors=steering_vectors,
        coeff=args.coeff,
        max_examples=args.max_examples,
    )

    steering_summary = summarize_steering_effect(baseline_losses, steered_losses)
    pd.DataFrame(
        {
            "example_index": np.arange(len(baseline_losses)),
            "baseline_loss": baseline_losses,
            "steered_loss": steered_losses,
            "loss_delta": steered_losses - baseline_losses,
        }
    ).to_csv(output_dir / "baseline_vs_steered_loss.csv", index=False)

    summary = {
        "checkpoint": args.checkpoint,
        "dataset": {
            "dataset_name": args.dataset_name,
            "data_folder": args.data_folder,
            "split_folder": args.split_folder,
        },
        "pipeline": {
            "mode": "deterministic_full_trajectory",
            "description": "All same-trajectory positive goals within the valid action-distance band.",
            "min_valid_goal_offset": args.min_action_distance + 1,
            "max_valid_goal_offset": args.max_action_distance - 1,
        },
        "num_examples": len(baseline_losses),
        "hook_layer_names": hook_layer_names,
        "selected_layers": layer_indices,
        "coeff": args.coeff,
        "residual_dtype": args.residual_dtype,
        "good_quartile_size": quartile,
        "bad_quartile_size": quartile,
        "vector_stats": {str(layer_idx): stats for layer_idx, stats in vector_stats.items()},
        "correlations": correlations,
        **steering_summary,
    }
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
