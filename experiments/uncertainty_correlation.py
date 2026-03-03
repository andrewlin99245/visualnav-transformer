"""
Uncertainty Correlation Experiment for ViNT

Tests whether uncertainty signals derived from ViNT's internal representations
correlate with navigation prediction failures. Extracts 4 uncertainty signals
and measures how well each predicts distance prediction errors.

Signals:
  1. MC Dropout Variance — variance of distance predictions under stochastic dropout
  2. Triangle Inequality Violation — metric consistency check (A→C vs A→B + B→C)
  3. Attention Entropy — Shannon entropy of self-attention weight distributions
  4. Mahalanobis Distance — OOD score of observation embeddings

Usage:
  python experiments/uncertainty_correlation.py \
    --checkpoint deployment/model_weights/vint.pth \
    --data-folder /path/to/go_stanford_cropped \
    --dataset-name go_stanford \
    --max-trajectories 5 \
    --samples-per-traj 20 \
    --compute-triangle \
    --seed 42
"""

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

# ---------------------------------------------------------------------------
# Add project root to path so we can import vint_train modules
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "train"))

from vint_train.models.vint.vint import ViNT
from vint_train.data.data_utils import IMAGE_ASPECT_RATIO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE = (85, 64)  # (width, height) — from vint.yaml
CONTEXT_SIZE = 5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MAX_DIST = 20  # from vint.yaml distance.max_dist_cat

NORMALIZE = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


# ===========================================================================
# 1. Data Loading
# ===========================================================================
class TrajectoryLoader:
    """Lightweight loader that reads trajectory folders directly."""

    def __init__(self, data_folder: str, max_trajectories: Optional[int] = None,
                 seed: int = 42):
        self.data_folder = Path(data_folder)
        self.rng = np.random.RandomState(seed)
        self.trajectories = self._discover_trajectories()
        if max_trajectories is not None and max_trajectories < len(self.trajectories):
            self.trajectories = self.rng.choice(
                self.trajectories, max_trajectories, replace=False
            ).tolist()
        print(f"Found {len(self.trajectories)} trajectories in {data_folder}")

    def _discover_trajectories(self) -> List[str]:
        """Walk data_folder, find subdirs containing traj_data.pkl."""
        trajs = []
        for entry in sorted(os.listdir(self.data_folder)):
            traj_dir = self.data_folder / entry
            if traj_dir.is_dir() and (traj_dir / "traj_data.pkl").exists():
                trajs.append(entry)
        if not trajs:
            raise FileNotFoundError(
                f"No trajectories found in {self.data_folder}. "
                "Expected subdirs with traj_data.pkl."
            )
        return trajs

    def _traj_length(self, traj_name: str) -> int:
        """Count number of images in trajectory (0.jpg, 1.jpg, ...)."""
        traj_dir = self.data_folder / traj_name
        idx = 0
        while (traj_dir / f"{idx}.jpg").exists():
            idx += 1
        return idx

    def load_image(self, traj_name: str, time_idx: int) -> torch.Tensor:
        """Load and preprocess a single image: center-crop, resize, normalize."""
        path = self.data_folder / traj_name / f"{time_idx}.jpg"
        img = Image.open(str(path)).convert("RGB")
        w, h = img.size
        if w > h:
            img = TF.center_crop(img, (h, int(h * IMAGE_ASPECT_RATIO)))
        else:
            img = TF.center_crop(img, (int(w / IMAGE_ASPECT_RATIO), w))
        img = img.resize(IMAGE_SIZE)
        tensor = TF.to_tensor(img)  # [3, H, W] in [0, 1]
        tensor = NORMALIZE(tensor)
        return tensor

    def build_obs_tensor(self, traj_name: str, curr_time: int) -> torch.Tensor:
        """Stack context_size+1 images → [1, 3*(context_size+1), H, W]."""
        imgs = []
        for t in range(curr_time - CONTEXT_SIZE, curr_time + 1):
            t_clamped = max(0, t)
            imgs.append(self.load_image(traj_name, t_clamped))
        obs = torch.cat(imgs, dim=0)  # [3*(context_size+1), H, W]
        return obs.unsqueeze(0)  # [1, 18, H, W]

    def build_goal_tensor(self, traj_name: str, goal_time: int) -> torch.Tensor:
        """Single goal image → [1, 3, H, W]."""
        img = self.load_image(traj_name, goal_time)
        return img.unsqueeze(0)

    def generate_samples(
        self, samples_per_traj: int = 20
    ) -> List[Dict]:
        """Generate (traj, curr_time, goal_time) samples with d_actual in [1, 20]."""
        samples = []
        for traj_name in self.trajectories:
            traj_len = self._traj_length(traj_name)
            if traj_len < CONTEXT_SIZE + 2:
                print(f"  Skipping {traj_name}: too short ({traj_len} frames)")
                continue
            count = 0
            attempts = 0
            while count < samples_per_traj and attempts < samples_per_traj * 10:
                attempts += 1
                curr_time = self.rng.randint(CONTEXT_SIZE, traj_len - 1)
                max_goal = min(curr_time + MAX_DIST, traj_len - 1)
                if max_goal <= curr_time:
                    continue
                goal_time = self.rng.randint(curr_time + 1, max_goal + 1)
                d_actual = goal_time - curr_time
                samples.append({
                    "traj_name": traj_name,
                    "curr_time": int(curr_time),
                    "goal_time": int(goal_time),
                    "d_actual": int(d_actual),
                })
                count += 1
        print(f"Generated {len(samples)} samples")
        return samples


# ===========================================================================
# 2. Model Loading
# ===========================================================================
def load_vint_model(checkpoint_path: str, device: torch.device) -> ViNT:
    """Load ViNT model from checkpoint (inlined to avoid ROS imports)."""
    model = ViNT(
        context_size=CONTEXT_SIZE,
        len_traj_pred=5,
        learn_angle=True,
        obs_encoder="efficientnet-b0",
        obs_encoding_size=512,
        late_fusion=False,
        mha_num_attention_heads=4,
        mha_num_attention_layers=4,
        mha_ff_dim_factor=4,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    loaded_model = checkpoint["model"]
    try:
        state_dict = loaded_model.module.state_dict()
    except AttributeError:
        state_dict = loaded_model.state_dict()
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded ViNT from {checkpoint_path}")
    return model


# ===========================================================================
# 3. Attention Capture (Signal 3 helper)
# ===========================================================================
class AttentionCapture:
    """Monkey-patches TransformerEncoderLayer.forward to capture attention weights.

    PyTorch >= 2.0 uses a fused fast-path in TransformerEncoderLayer.forward()
    that bypasses _sa_block entirely. We therefore replace forward() on each
    layer instance with a version that always takes the non-fused path and
    calls self_attn with need_weights=True.
    """

    def __init__(self):
        self.attention_weights: Dict[int, torch.Tensor] = {}
        self._originals: Dict[int, object] = {}

    def install(self, model: ViNT):
        """Replace forward on each transformer layer to capture attention weights."""
        layers = model.decoder.sa_decoder.layers
        for i, layer in enumerate(layers):
            self._originals[i] = layer.forward
            capture = self

            def make_patched_forward(layer_ref, layer_idx):
                def patched_forward(src, src_mask=None, src_key_padding_mask=None, is_causal=False):
                    # Self-attention with weight capture (norm-first path,
                    # matching the model's norm_first=True config)
                    x = src
                    sa_input = layer_ref.norm1(x)
                    attn_out, attn_w = layer_ref.self_attn(
                        sa_input, sa_input, sa_input,
                        attn_mask=src_mask,
                        key_padding_mask=src_key_padding_mask,
                        need_weights=True,
                        average_attn_weights=False,
                    )
                    capture.attention_weights[layer_idx] = attn_w.detach()
                    x = x + layer_ref.dropout1(attn_out)
                    # Feed-forward (norm-first path)
                    ff_input = layer_ref.norm2(x)
                    x = x + layer_ref._ff_block(ff_input)
                    return x
                return patched_forward

            layer.forward = make_patched_forward(layer, i)

    def remove(self, model: ViNT):
        """Restore original forward methods."""
        layers = model.decoder.sa_decoder.layers
        for i, layer in enumerate(layers):
            if i in self._originals:
                layer.forward = self._originals[i]
        self._originals.clear()
        self.attention_weights.clear()

    def get_entropy(self) -> float:
        """Compute mean Shannon entropy across all layers and heads."""
        entropies = []
        for layer_idx, weights in self.attention_weights.items():
            # weights shape: [B, num_heads, seq_len, seq_len]
            w = weights.float()
            w = w.clamp(min=1e-12)
            entropy = -(w * w.log()).sum(dim=-1)  # [B, heads, seq_len]
            entropies.append(entropy.mean().item())
        return float(np.mean(entropies)) if entropies else 0.0
    def get_goal_attention_entropy(self) -> float:
        """Compute entropy of Goal Token's attention to past Context Tokens."""
        entropies = []
        for layer_idx, weights in self.attention_weights.items():
            # weights shape: [B, num_heads, seq_len, seq_len]
            # seq_len = context_size + 2. Last token (-1) is goal token.
            w_goal = weights[:, :, -1, :-1].float()
            
            w_goal_sum = w_goal.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            w_goal_norm = w_goal / w_goal_sum
            w_goal_norm = w_goal_norm.clamp(min=1e-12)
            
            entropy = -(w_goal_norm * w_goal_norm.log()).sum(dim=-1)
            entropies.append(entropy.mean().item())
        return float(np.mean(entropies)) if entropies else 0.0

# ===========================================================================
# 4. Embedding extraction (Signal 4 helper)
# ===========================================================================
def extract_obs_embedding(model: ViNT, obs_img: torch.Tensor) -> np.ndarray:
    """
    Extract the 512-dim observation embedding for the current frame.
    Runs partial forward pass through obs_encoder + compress_obs_enc.
    """
    with torch.no_grad():
        # Split obs into individual frames: each [1, 3, H, W]
        obs_split = torch.split(obs_img, 3, dim=1)
        obs_cat = torch.cat(obs_split, dim=0)  # [(context+1), 3, H, W]

        enc = model.obs_encoder.extract_features(obs_cat)
        enc = model.obs_encoder._avg_pooling(enc)
        if model.obs_encoder._global_params.include_top:
            enc = enc.flatten(start_dim=1)
            # No dropout for deterministic embeddings (model.eval())
        enc = model.compress_obs_enc(enc)
        # enc shape: [(context+1), 512]
        # Reshape like forward pass: [context+1, batch=1, 512] → [1, context+1, 512]
        enc = enc.reshape((CONTEXT_SIZE + 1, -1, 512))
        enc = enc.transpose(0, 1)
        # Return last (current) observation embedding
        return enc[0, -1, :].cpu().numpy()  # [512]

def compute_latent_temporal_variance(model: ViNT, obs_img: torch.Tensor) -> float:
    """
    Method 3: Latent Temporal Variance
    Measures the variance of cosine similarities between consecutive frames in the latent space.
    """
    import torch.nn.functional as F
    with torch.no_grad():
        # Split obs into individual frames: each [1, 3, H, W]
        obs_split = torch.split(obs_img, 3, dim=1)
        obs_cat = torch.cat(obs_split, dim=0)  # [(context+1), 3, H, W]

        enc = model.obs_encoder.extract_features(obs_cat)
        enc = model.obs_encoder._avg_pooling(enc)
        if model.obs_encoder._global_params.include_top:
            enc = enc.flatten(start_dim=1)
        enc = model.compress_obs_enc(enc)  # [(context+1), 512]
        
        # Compute cosine similarities between consecutive frames
        sims = F.cosine_similarity(enc[:-1], enc[1:], dim=1)  # [context]
        
        var = torch.var(sims).item()
        # Fallback if there's only 1 context frame (nan variance)
        if torch.isnan(torch.tensor(var)):
            return 0.0
            
        return float(var)
def compute_task_dissonance(model: ViNT, obs: torch.Tensor, goal: torch.Tensor) -> float:
    """
    Method 4: Task Dissonance (Distance vs. Action Gradient Alignment)
    Calculates gradient cosine similarity of dist_pred and action_pred.norm() w.r.t the bottleneck representation.
    """
    import torch.nn.functional as F
    
    # We need to compute gradients, so we must be in train mode momentarily
    was_eval = not model.training
    model.train()
    
    # Enable gradients for inputs if not already
    obs = obs.clone().detach().requires_grad_(True)
    goal = goal.clone().detach().requires_grad_(True)
    
    # Let's attach a hook to capture gradients on the bottleneck layer
    # Since we can't easily hook just the output of decoder without modifying forward,
    # we can simulate the forward pass up to the bottleneck.
    # Note: ViNT forward pass does this:
    #   tokens = cat(obs_enc, goal_enc) -> final_repr = decoder(tokens) -> dist_pred, action_pred
            
    # Recreate the initial encoding part of forward pass to get tokens manually
    # Late Fusion vs Early Fusion handling:
    if model.late_fusion:
        goal_encoding = model.goal_encoder.extract_features(goal)
    else:
        obsgoal_img = torch.cat([obs[:, 3*model.context_size:, :, :], goal], dim=1)
        goal_encoding = model.goal_encoder.extract_features(obsgoal_img)
    goal_encoding = model.goal_encoder._avg_pooling(goal_encoding)
    if model.goal_encoder._global_params.include_top:
        goal_encoding = goal_encoding.flatten(start_dim=1)
        goal_encoding = model.goal_encoder._dropout(goal_encoding)
    goal_encoding = model.compress_goal_enc(goal_encoding)
    if len(goal_encoding.shape) == 2:
        goal_encoding = goal_encoding.unsqueeze(1)
        
    obs_img_split = torch.split(obs, 3, dim=1)
    obs_img_cat = torch.concat(obs_img_split, dim=0)
    obs_encoding = model.obs_encoder.extract_features(obs_img_cat)
    obs_encoding = model.obs_encoder._avg_pooling(obs_encoding)
    if model.obs_encoder._global_params.include_top:
        obs_encoding = obs_encoding.flatten(start_dim=1)
        obs_encoding = model.obs_encoder._dropout(obs_encoding)
    obs_encoding = model.compress_obs_enc(obs_encoding)
    obs_encoding = obs_encoding.reshape((model.context_size+1, -1, model.obs_encoding_size))
    obs_encoding = torch.transpose(obs_encoding, 0, 1)

    tokens = torch.cat((obs_encoding, goal_encoding), dim=1)
    
    # Ensure tokens require grad
    tokens.retain_grad()
    
    final_repr = model.decoder(tokens)
    final_repr.retain_grad()
    
    dist_pred = model.dist_predictor(final_repr)
    action_pred = model.action_predictor(final_repr)
    
    # 1. Gradient of Distance Prediction
    # Zero gradients first
    model.zero_grad()
    if final_repr.grad is not None:
        final_repr.grad.zero_()
        
    dist_pred.sum().backward(retain_graph=True)
    grad_dist = final_repr.grad.clone().detach()
    
    # 2. Gradient of Action Prediction Norm
    model.zero_grad()
    if final_repr.grad is not None:
        final_repr.grad.zero_()
        
    action_norm = action_pred.norm(p=2, dim=-1)
    action_norm.sum().backward()
    grad_action = final_repr.grad.clone().detach()
    
    # Return to previous mode
    if was_eval:
        model.eval()
        
    # Calculate cosine similarity
    # Shape of grad_dist & grad_action is [batch_size, 32]
    # If the gradients are severely misaligned (score approaches -1), it indicates task dissonance.
    cos_sim = F.cosine_similarity(grad_dist, grad_action, dim=1).item()
    
    return float(cos_sim)

def compute_latent_jacobian_jerk(
    model: ViNT, loader: TrajectoryLoader, sample: Dict, device: torch.device
) -> float:
    """
    Method 5: Latent Jacobian Jerk
    Measures the temporal instability of the gradients/Jacobians of the task heads
    with respect to the bottleneck temporal representation `final_repr`.
    """
    curr_time = sample["curr_time"]
    traj_name = sample["traj_name"]
    
    # We need obs at t and t-1. The loader automatically pads frames < 0 inside build_obs_tensor.
    obs_t = loader.build_obs_tensor(traj_name, curr_time).to(device)
    obs_prev = loader.build_obs_tensor(traj_name, curr_time - 1).to(device)
    goal = loader.build_goal_tensor(traj_name, sample["goal_time"]).to(device)
    
    z_cache = []
    def hook(module, inp, out):
        z_cache.append(out)
        
    handle = model.decoder.register_forward_hook(hook)
    
    was_eval = not model.training
    model.eval()
    
    with torch.no_grad():
        model(obs_t, goal)
        z_t = z_cache[0].clone().detach()
        z_cache.clear()
        
        model(obs_prev, goal)
        z_prev = z_cache[0].clone().detach()
        z_cache.clear()
        
    handle.remove()
    
    z_t.requires_grad_(True)
    z_prev.requires_grad_(True)
    
    # Enable grad calculation momentarily for MLP jacobians
    with torch.enable_grad():
        def get_gradients(z):
            # Distance Gradient
            d_pred = model.dist_predictor(z) # shape [1, 1]
            g_d = torch.autograd.grad(d_pred, z, create_graph=True)[0] # [1, 32]
            
            # Action Jacobian
            a_pred = model.action_predictor(z) # usually shape [1, 5, 2]
            a_pred = a_pred.view(-1)
            J_a_rows = []
            for i in range(a_pred.shape[0]):
                g_a_i = torch.autograd.grad(a_pred[i], z, retain_graph=True)[0]
                J_a_rows.append(g_a_i)
            J_a = torch.cat(J_a_rows, dim=0) # [num_actions, 32]
            
            return g_d, J_a

        g_d_t, J_a_t = get_gradients(z_t)
        g_d_prev, J_a_prev = get_gradients(z_prev)
        
    if not was_eval:
        model.train()
        
    import torch.nn.functional as F
    # 1. Cosine distance between distance gradients
    cos_sim = F.cosine_similarity(g_d_t, g_d_prev, dim=1).item()
    d_jerk = 1.0 - cos_sim
    
    # 2. Frobenius norm of Jacobian difference
    a_jerk = torch.norm(J_a_t - J_a_prev, p='fro').item()
    
    # Return summed instability 
    return float(d_jerk + a_jerk)

# ===========================================================================
# 5. Mahalanobis Distance Estimator (Signal 4)
# ===========================================================================
class MahalanobisEstimator:
    """Fit PCA on reference embeddings, score new points via Mahalanobis."""

    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.cov_inv = None
        self._fitted = False

    def fit(self, embeddings: np.ndarray):
        """Fit PCA + covariance on reference embeddings [N, D]."""
        from sklearn.decomposition import PCA

        if embeddings.ndim != 2 or embeddings.shape[0] < 3:
            print("  WARNING: Not enough reference embeddings for Mahalanobis fitting")
            return
        n_samples = embeddings.shape[0]
        n_comp = min(self.n_components, n_samples - 1, embeddings.shape[1])
        pca = PCA(n_components=n_comp)
        projected = pca.fit_transform(embeddings)  # [N, n_comp]
        self.mean = pca.mean_
        self.components = pca.components_  # [n_comp, D]
        cov = np.cov(projected, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        # Regularize for numerical stability using a fraction of the max variance
        max_var = np.max(np.diag(cov)) if cov.shape[0] > 0 else 1.0
        cov += np.eye(cov.shape[0]) * (1e-3 * max_var + 1e-6)
        self.cov_inv = np.linalg.inv(cov)
        self._fitted = True
        print(f"  MahalanobisEstimator fitted with {n_samples} samples, {n_comp} components")

    def score(self, embedding: np.ndarray) -> float:
        """Mahalanobis distance of a single embedding."""
        if not self._fitted:
            return 0.0
        centered = embedding - self.mean
        projected = self.components @ centered  # [n_comp]
        return float(projected @ self.cov_inv @ projected)


# ===========================================================================
# 6. Uncertainty Signals
# ===========================================================================
def compute_mc_dropout_variance(
    model: ViNT, obs: torch.Tensor, goal: torch.Tensor,
    n_passes: int = 20,
) -> Tuple[float, float]:
    """
    Signal 1: MC Dropout Variance.
    Returns (variance, mean_prediction).
    """
    model.train()  # enable dropout
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            dist_pred, _, _ = model(obs, goal)
            preds.append(dist_pred.item())
    model.eval()
    preds = np.array(preds)
    return float(np.var(preds)), float(np.mean(preds))


def compute_triangle_violation(
    model: ViNT, loader: TrajectoryLoader,
    traj_name: str, curr_time: int, goal_time: int,
    d_ac: float, num_triplets: int = 50, device: torch.device = torch.device("cpu"),
) -> float:
    """
    Signal 2: Triangle Inequality Violation.
    violation = max(0, d(A,C) - d(A,B) - d(B,C))
    Returns max violation across K intermediate points.
    """
    traj_len = loader._traj_length(traj_name)
    # Choose K intermediate points between curr_time and goal_time
    possible_intermediates = list(range(curr_time + 1, goal_time))
    if not possible_intermediates:
        return 0.0

    K = min(num_triplets, len(possible_intermediates))
    intermediates = loader.rng.choice(possible_intermediates, K, replace=(K > len(possible_intermediates)))

    obs_a = loader.build_obs_tensor(traj_name, curr_time).to(device)

    # Batch d(A, B_i): build [K, 3, H, W] goal batch
    goal_imgs = []
    for b in intermediates:
        goal_imgs.append(loader.load_image(traj_name, b))
    goal_batch = torch.stack(goal_imgs, dim=0).to(device)  # [K, 3, H, W]
    obs_a_rep = obs_a.expand(K, -1, -1, -1)  # [K, 18, H, W]

    with torch.no_grad():
        d_ab, _, _ = model(obs_a_rep, goal_batch)
        d_ab = d_ab.squeeze(-1).cpu().numpy()  # [K]

    # Individual passes for d(B_i, C)
    goal_c = loader.build_goal_tensor(traj_name, goal_time).to(device)
    d_bc_list = []
    with torch.no_grad():
        for b in intermediates:
            obs_b = loader.build_obs_tensor(traj_name, b).to(device)
            goal_c_single = goal_c  # [1, 3, H, W]
            d_bc, _, _ = model(obs_b, goal_c_single)
            d_bc_list.append(d_bc.item())
    d_bc = np.array(d_bc_list)

    violations = np.maximum(0, d_ac - d_ab - d_bc)
    return float(np.max(violations))


def compute_attention_entropy(
    model: ViNT, obs: torch.Tensor, goal: torch.Tensor,
    capture: AttentionCapture,
) -> float:
    """Signal 3: Attention Entropy."""
    capture.attention_weights.clear()
    with torch.no_grad():
        model(obs, goal)
    return capture.get_entropy()

def compute_goal_masking_score(
    model: ViNT, obs: torch.Tensor, goal: torch.Tensor,
    d_original: float
) -> float:
    """
    Method 1: Goal-Masking Sensitivity.
    How much does the distance prediction change if the model is blind to the goal?
    """
    with torch.no_grad():
        dummy_goal = torch.zeros_like(goal)
        d_blind, _, _ = model(obs, dummy_goal)
        d_blind_val = d_blind.item()
    
    diff = abs(d_original - d_blind_val)
    return float(diff)

# ===========================================================================
# 7. Failure Proxy
# ===========================================================================
def compute_failure_proxy(
    d_predicted: float, d_actual: float, threshold: float = 0.3
) -> Tuple[bool, float]:
    """
    Returns (is_failure, relative_error).
    failure = |d_predicted - d_actual| > threshold * d_actual
    """
    abs_error = abs(d_predicted - d_actual)
    relative_error = abs_error / max(d_actual, 1e-6)
    is_failure = abs_error > threshold * d_actual
    return bool(is_failure), float(relative_error)


# ===========================================================================
# 8. Sanity Checks
# ===========================================================================
def run_sanity_checks(
    model: ViNT, loader: TrajectoryLoader, device: torch.device
):
    """Run sanity checks before the main experiment."""
    print("\n=== Sanity Checks ===")
    traj = loader.trajectories[0]
    traj_len = loader._traj_length(traj)
    curr_t = min(CONTEXT_SIZE, traj_len - 2)
    goal_t = min(curr_t + 3, traj_len - 1)

    obs = loader.build_obs_tensor(traj, curr_t).to(device)
    goal = loader.build_goal_tensor(traj, goal_t).to(device)

    # Check 1: Distance predictions in reasonable range
    model.eval()
    with torch.no_grad():
        dist_pred, _, _ = model(obs, goal)
    d = dist_pred.item()
    print(f"  [1] Distance prediction for nearby frames (dt={goal_t - curr_t}): {d:.4f}")
    assert -5 < d < 50, f"Distance prediction {d} out of reasonable range"
    print("      PASS")

    # Check 2: MC Dropout produces varying outputs
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(2):
            dp, _, _ = model(obs, goal)
            preds.append(dp.item())
    model.eval()
    differ = abs(preds[0] - preds[1]) > 1e-8
    print(f"  [2] MC Dropout: pass1={preds[0]:.6f}, pass2={preds[1]:.6f}, differ={differ}")
    if not differ:
        print("      WARNING: MC Dropout passes identical — dropout may not be active")
    else:
        print("      PASS")

    # Check 3: Attention hooks capture correct shape
    capture = AttentionCapture()
    capture.install(model)
    with torch.no_grad():
        model(obs, goal)
    for layer_idx, w in capture.attention_weights.items():
        print(f"  [3] Attention layer {layer_idx} shape: {list(w.shape)}")
        # Expected: [1, 4, 7, 7] (batch=1, 4 heads, 7 tokens = context+1+goal)
        assert w.shape[1] == 4, f"Expected 4 heads, got {w.shape[1]}"
        assert w.shape[2] == CONTEXT_SIZE + 2, f"Expected {CONTEXT_SIZE + 2} tokens, got {w.shape[2]}"
    capture.remove(model)
    print("      PASS")

    print("=== Sanity Checks Complete ===\n")


# ===========================================================================
# 9. Main Experiment
# ===========================================================================
def run_experiment(args):
    """Run the full uncertainty correlation experiment."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_vint_model(args.checkpoint, device)

    # Load data
    loader = TrajectoryLoader(
        args.data_folder,
        max_trajectories=args.max_trajectories,
        seed=args.seed,
    )

    # Sanity checks
    run_sanity_checks(model, loader, device)

    # Generate samples
    samples = loader.generate_samples(samples_per_traj=args.samples_per_traj)
    if not samples:
        print("ERROR: No samples generated. Check your data.")
        return

    # Install attention hooks
    capture = AttentionCapture()
    capture.install(model)

    # Collect reference embeddings from longest trajectory for Mahalanobis
    print("Fitting Mahalanobis reference distribution...")
    ref_traj = max(loader.trajectories, key=lambda t: loader._traj_length(t))
    ref_len = loader._traj_length(ref_traj)
    print(f"  Using reference trajectory: {ref_traj} ({ref_len} frames)")
    ref_embeddings = []
    model.eval()
    for t in range(CONTEXT_SIZE, ref_len):
        obs = loader.build_obs_tensor(ref_traj, t).to(device)
        emb = extract_obs_embedding(model, obs)
        ref_embeddings.append(emb)
    if len(ref_embeddings) < 3:
        print("  WARNING: Reference trajectory too short for Mahalanobis. Using dummy scores.")
    ref_embeddings = np.array(ref_embeddings) if ref_embeddings else np.empty((0, 512))
    mahal = MahalanobisEstimator(n_components=args.pca_components)
    mahal.fit(ref_embeddings)

    # Process samples
    print(f"\nProcessing {len(samples)} samples...")
    results = []
    t_start = time.time()

    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  Sample {i + 1}/{len(samples)} ({rate:.1f} samples/sec)")

        obs = loader.build_obs_tensor(sample["traj_name"], sample["curr_time"]).to(device)
        goal = loader.build_goal_tensor(sample["traj_name"], sample["goal_time"]).to(device)
        d_actual = sample["d_actual"]

        # Get baseline prediction (eval mode)
        model.eval()
        capture.attention_weights.clear()
        with torch.no_grad():
            dist_pred, _, _ = model(obs, goal)
        d_predicted = dist_pred.item()

        # Signal 3: Attention entropy (captured during eval forward pass above)
        attn_entropy = capture.get_entropy()
        
        # New Method 2: Goal Attention Entropy
        goal_attn_entropy = capture.get_goal_attention_entropy()
        
        # New Method 1: Goal-Masking Sensitivity
        goal_masking_score = compute_goal_masking_score(model, obs, goal, d_predicted)

        # New Method 3: Latent Temporal Variance
        latent_temp_var = compute_latent_temporal_variance(model, obs)
        
        # New Method 4: Task Dissonance
        task_dissonance = compute_task_dissonance(model, obs, goal)
        
        # New Method 5: Latent Jacobian Jerk
        latent_jacobian_jerk = compute_latent_jacobian_jerk(model, loader, sample, device)

        # Signal 1: MC Dropout variance
        mc_var, mc_mean = compute_mc_dropout_variance(
            model, obs, goal, n_passes=args.mc_passes
        )
        # Re-install attention hooks after model.train()/eval() cycle
        # (hooks persist since we patched instance methods)

        # Signal 4: Mahalanobis distance
        model.eval()
        emb = extract_obs_embedding(model, obs)
        mahal_dist = mahal.score(emb)

        # Signal 2: Triangle violation (optional, expensive)
        tri_violation = 0.0
        if args.compute_triangle:
            model.eval()
            tri_violation = compute_triangle_violation(
                model, loader,
                sample["traj_name"], sample["curr_time"], sample["goal_time"],
                d_ac=d_predicted, num_triplets=args.num_triplets,
                device=device,
            )

        # Failure proxy
        is_failure, rel_error = compute_failure_proxy(d_predicted, d_actual)

        results.append({
            "traj_name": sample["traj_name"],
            "curr_time": sample["curr_time"],
            "goal_time": sample["goal_time"],
            "d_actual": d_actual,
            "d_predicted": d_predicted,
            "is_failure": is_failure,
            "relative_error": rel_error,
            "mc_dropout_var": mc_var,
            "mc_dropout_mean": mc_mean,
            "triangle_violation": tri_violation,
            "attention_entropy": attn_entropy,
            "mahalanobis_distance": mahal_dist,
            "goal_attn_entropy": goal_attn_entropy,
            "goal_masking_score": goal_masking_score,
            "latent_temp_var": latent_temp_var,
            "task_dissonance": task_dissonance,
            "latent_jacobian_jerk": latent_jacobian_jerk,
        })

    capture.remove(model)
    elapsed = time.time() - t_start
    print(f"\nProcessing complete in {elapsed:.1f}s")

    # Print signal statistics
    print("\n=== Signal Statistics ===")
    for signal in ["mc_dropout_var", "triangle_violation", "attention_entropy", "mahalanobis_distance", "goal_attn_entropy", "goal_masking_score", "latent_temp_var", "task_dissonance", "latent_jacobian_jerk"]:
        vals = [r[signal] for r in results]
        print(f"  {signal}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}, "
              f"min={np.min(vals):.4f}, max={np.max(vals):.4f}")

    # Analysis
    metrics = compute_metrics(results, compute_triangle=args.compute_triangle)
    print("\n=== Results ===")
    for signal, m in metrics.items():
        print(f"  {signal}:")
        print(f"    AUROC:    {m['auroc']:.4f}" if m["auroc"] is not None else "    AUROC:    N/A")
        print(f"    Spearman: {m['spearman_rho']:.4f} (p={m['spearman_p']:.4e})")

    # Save results
    output_dir = PROJECT_ROOT / "experiments" / "results" / "uncertainty_correlation"
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "checkpoint": args.checkpoint,
            "data_folder": args.data_folder,
            "dataset_name": args.dataset_name,
            "max_trajectories": args.max_trajectories,
            "samples_per_traj": args.samples_per_traj,
            "mc_passes": args.mc_passes,
            "num_triplets": args.num_triplets,
            "pca_components": args.pca_components,
            "compute_triangle": args.compute_triangle,
            "seed": args.seed,
            "device": str(device),
            "n_samples": len(results),
            "elapsed_seconds": elapsed,
        },
        "metrics": metrics,
        "samples": results,
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Plots
    plot_results(results, metrics, output_dir, compute_triangle=args.compute_triangle)
    print(f"Plots saved to {output_dir}")


# ===========================================================================
# 10. Analysis
# ===========================================================================
def compute_metrics(
    results: List[Dict], compute_triangle: bool = False
) -> Dict:
    """Compute AUROC and Spearman correlation for each signal."""
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    failures = np.array([r["is_failure"] for r in results])
    rel_errors = np.array([r["relative_error"] for r in results])

    signals = {
        "mc_dropout_var": np.array([r["mc_dropout_var"] for r in results]),
        "attention_entropy": np.array([r["attention_entropy"] for r in results]),
        "mahalanobis_distance": np.array([r["mahalanobis_distance"] for r in results]),
        "goal_attn_entropy": np.array([r["goal_attn_entropy"] for r in results]),
        "goal_masking_score": np.array([r["goal_masking_score"] for r in results]),
        "latent_temp_var": np.array([r["latent_temp_var"] for r in results]),
        "task_dissonance": np.array([r["task_dissonance"] for r in results]),
        "latent_jacobian_jerk": np.array([r["latent_jacobian_jerk"] for r in results]),
    }
    if compute_triangle:
        signals["triangle_violation"] = np.array([r["triangle_violation"] for r in results])

    metrics = {}
    for name, values in signals.items():
        # AUROC (needs both classes present)
        auroc = None
        if len(np.unique(failures)) == 2:
            try:
                auroc = float(roc_auc_score(failures, values))
            except ValueError:
                pass

        # Spearman correlation
        rho, p = spearmanr(values, rel_errors)

        metrics[name] = {
            "auroc": auroc,
            "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
            "spearman_p": float(p) if not np.isnan(p) else 1.0,
        }

    return metrics


def plot_results(
    results: List[Dict], metrics: Dict, output_dir: Path,
    compute_triangle: bool = False,
):
    """Generate scatter plots and ROC curves."""
    from sklearn.metrics import roc_curve, roc_auc_score

    failures = np.array([r["is_failure"] for r in results])
    rel_errors = np.array([r["relative_error"] for r in results])

    signal_configs = [
        ("mc_dropout_var", "MC Dropout Variance", "scatter_mc_dropout.png"),
        ("attention_entropy", "Attention Entropy", "scatter_attention.png"),
        ("mahalanobis_distance", "Mahalanobis Distance", "scatter_mahalanobis.png"),
        ("goal_attn_entropy", "Goal Attn Entropy", "scatter_goal_attn.png"),
        ("goal_masking_score", "Goal Masking Score", "scatter_goal_masking.png"),
        ("latent_temp_var", "Latent Temp Variance", "scatter_latent_temp.png"),
        ("task_dissonance", "Task Dissonance", "scatter_task_dissonance.png"),
        ("latent_jacobian_jerk", "Latent Jacobian Jerk", "scatter_ljj.png"),
    ]
    if compute_triangle:
        signal_configs.append(
            ("triangle_violation", "Triangle Violation", "scatter_triangle.png")
        )

    # Scatter plots
    for signal_key, signal_label, filename in signal_configs:
        values = np.array([r[signal_key] for r in results])
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["green" if not f else "red" for f in failures]
        ax.scatter(values, rel_errors, c=colors, alpha=0.6, s=20)
        ax.set_xlabel(signal_label)
        ax.set_ylabel("Relative Error")
        m = metrics.get(signal_key, {})
        title = f"{signal_label} vs Relative Error"
        if m.get("spearman_rho") is not None:
            title += f"\n(Spearman ρ={m['spearman_rho']:.3f}, AUROC={m.get('auroc', 'N/A')})"
        ax.set_title(title)
        ax.legend(
            handles=[
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", label="Success"),
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="red", label="Failure"),
            ]
        )
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)

    # Combined ROC curves
    if len(np.unique(failures)) == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        for signal_key, signal_label, _ in signal_configs:
            values = np.array([r[signal_key] for r in results])
            try:
                fpr, tpr, _ = roc_curve(failures, values)
                auc = roc_auc_score(failures, values)
                ax.plot(fpr, tpr, label=f"{signal_label} (AUC={auc:.3f})")
            except ValueError:
                pass
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — Uncertainty Signals as Failure Predictors")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "roc_curves.png", dpi=150)
        plt.close(fig)
    else:
        print("  WARNING: Only one class present in failures, skipping ROC curve plot")


# ===========================================================================
# 11. CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Uncertainty Correlation Experiment for ViNT"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to vint.pth checkpoint",
    )
    parser.add_argument(
        "--data-folder", type=str, required=True,
        help="Path to dataset folder (e.g., go_stanford_cropped)",
    )
    parser.add_argument(
        "--dataset-name", type=str, default="go_stanford",
        help="Dataset name for metric_waypoint_spacing lookup",
    )
    parser.add_argument(
        "--max-trajectories", type=int, default=None,
        help="Max number of trajectories to use (default: all)",
    )
    parser.add_argument(
        "--samples-per-traj", type=int, default=20,
        help="Number of samples per trajectory (default: 20)",
    )
    parser.add_argument(
        "--mc-passes", type=int, default=20,
        help="Number of MC Dropout forward passes (default: 20)",
    )
    parser.add_argument(
        "--num-triplets", type=int, default=50,
        help="Number of intermediate points for triangle inequality (default: 50)",
    )
    parser.add_argument(
        "--pca-components", type=int, default=50,
        help="Number of PCA components for Mahalanobis (default: 50)",
    )
    parser.add_argument(
        "--compute-triangle", action="store_true",
        help="Compute triangle inequality violations (expensive)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    run_experiment(args)


if __name__ == "__main__":
    main()
