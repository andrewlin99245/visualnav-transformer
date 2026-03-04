import wandb
import os
import numpy as np
from typing import List, Optional, Dict
from prettytable import PrettyTable

from vint_train.training.train_utils import train, evaluate, SIRAHook

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms

try:
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.training_utils import EMAModel
    from vint_train.training.train_utils import train_nomad, evaluate_nomad
except ImportError:
    pass

def recompute_sira_vectors(
    model: nn.Module,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    subset_frac: float = 0.1,
    error_threshold: float = 0.3,
) -> dict:
    """Compute per-layer steering vectors from model's own prediction errors.

    Runs a forward pass over a subset of training data, partitions samples by
    prediction quality, and returns normalized mean difference of residual
    stream activations at each transformer layer.

    Returns:
        Dict mapping layer_idx -> unit steering vector of shape [seq_len * embed_dim],
        or None if insufficient samples in either partition.
    """
    target_model = model.module if hasattr(model, 'module') else model
    num_layers = len(target_model.decoder.sa_decoder.layers)
    was_training = model.training
    model.eval()

    # Install hooks on all layers
    hook = SIRAHook()
    hook.install(target_model.decoder.sa_decoder.layers)

    # Per-layer accumulators
    good_hiddens = {i: [] for i in range(num_layers)}
    bad_hiddens = {i: [] for i in range(num_layers)}
    n_batches = max(1, int(len(dataloader) * subset_frac))

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= n_batches:
                break

            (obs_image, goal_image, action_label, dist_label,
             goal_pos, dataset_index, action_mask) = data

            obs_images = torch.split(obs_image, 3, dim=1)
            obs_images = [transform(img).to(device) for img in obs_images]
            obs_image = torch.cat(obs_images, dim=1)
            goal_image = transform(goal_image).to(device)
            dist_label = dist_label.to(device)

            dist_pred, _, _ = model(obs_image, goal_image)

            # Compute per-sample relative error
            rel_error = torch.abs(
                dist_pred.squeeze(-1) - dist_label.float()
            ) / dist_label.float().clamp(min=1.0)

            for layer_idx in range(num_layers):
                h = hook.captured[layer_idx].detach()
                h_flat = h.reshape(h.shape[0], -1)  # [B, 3584]
                for j in range(h_flat.shape[0]):
                    if rel_error[j].item() < error_threshold:
                        good_hiddens[layer_idx].append(h_flat[j])
                    else:
                        bad_hiddens[layer_idx].append(h_flat[j])

    hook.remove()

    if was_training:
        model.train()

    # Check we have enough samples (use layer 0 as proxy — all layers have same count)
    n_good = len(good_hiddens[0])
    n_bad = len(bad_hiddens[0])
    if n_good < 5 or n_bad < 5:
        print(f"  SIRA: insufficient samples (good={n_good}, "
              f"bad={n_bad}), skipping recomputation")
        return None

    vectors = {}
    for layer_idx in range(num_layers):
        good_mean = torch.stack(good_hiddens[layer_idx]).mean(dim=0)
        bad_mean = torch.stack(bad_hiddens[layer_idx]).mean(dim=0)
        v = good_mean - bad_mean
        v_normalized = v / (v.norm() + 1e-12)
        vectors[layer_idx] = v_normalized.to(device)

        cos_sim = F.cosine_similarity(good_mean.unsqueeze(0), bad_mean.unsqueeze(0)).item()
        print(f"  SIRA L{layer_idx}: raw_norm={v.norm().item():.2f}, cos_sim={cos_sim:.4f}")

    print(f"  SIRA: {n_good} good, {n_bad} bad samples across {n_batches} batches")
    return vectors


def train_eval_loop(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    dataloader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    epochs: int,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    wandb_log_freq: int = 10,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    confidence_lambda: float = 0.0,
    sira_lambda: float = 0.0,
    sira_recompute_every: int = 5,
    sira_margin: float = 0.0,
    sira_subset_frac: float = 0.1,
):
    """
    Train and evaluate the model for several epochs (vint or gnm models)
    """
    assert 0 <= alpha <= 1
    latest_path = os.path.join(project_folder, f"latest.pth")
    sira_vectors = None  # Will be computed after first K epochs

    for epoch in range(current_epoch, current_epoch + epochs):
        # Recompute SIRA steering vectors every K epochs
        if (sira_lambda > 0 and epoch > 0
                and epoch % sira_recompute_every == 0):
            print(f"\n=== SIRA: Recomputing steering vectors at epoch {epoch} ===")
            sira_vectors = recompute_sira_vectors(
                model=model,
                dataloader=dataloader,
                transform=transform,
                device=device,
                subset_frac=sira_subset_frac,
            )

        if train_model:
            print(
            f"Start ViNT Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train(
                model=model,
                optimizer=optimizer,
                dataloader=dataloader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                confidence_lambda=confidence_lambda,
                sira_vectors=sira_vectors,
                sira_lambda=sira_lambda,
                sira_margin=sira_margin,
            )

        avg_total_test_loss = []
        for dataset_type in test_dataloaders:
            print(
                f"Start {dataset_type} ViNT Testing Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            loader = test_dataloaders[dataset_type]

            test_dist_loss, test_action_loss, total_eval_loss = evaluate(
                eval_type=dataset_type,
                model=model,
                dataloader=loader,
                transform=transform,
                device=device,
                project_folder=project_folder,
                normalized=normalized,
                epoch=epoch,
                alpha=alpha,
                learn_angle=learn_angle,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                eval_fraction=eval_fraction,
                confidence_lambda=confidence_lambda,
            )

            avg_total_test_loss.append(total_eval_loss)

        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "avg_total_test_loss": np.mean(avg_total_test_loss),
            "scheduler": scheduler
        }
        # log average eval loss
        wandb.log({}, commit=False)

        if scheduler is not None:
            # scheduler calls based on the type of scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(np.mean(avg_total_test_loss))
            else:
                scheduler.step()
        wandb.log({
            "avg_total_test_loss": np.mean(avg_total_test_loss),
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(checkpoint, latest_path)
        torch.save(checkpoint, numbered_path)  # keep track of model at every epoch

    # Flush the last set of eval logs
    wandb.log({})
    print()

def train_eval_loop_nomad(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam, 
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    print_log_freq: int = 100,
    wandb_log_freq: int = 10,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    current_epoch: int = 0,
    alpha: float = 1e-4,
    use_wandb: bool = True,
    eval_fraction: float = 0.25,
    eval_freq: int = 1,
):
    """
    Train and evaluate the model for several epochs (vint or gnm models)

    Args:
        model: model to train
        optimizer: optimizer to use
        lr_scheduler: learning rate scheduler to use
        noise_scheduler: noise scheduler to use
        dataloader: dataloader for train dataset
        test_dataloaders: dict of dataloaders for testing
        transform: transform to apply to images
        goal_mask_prob: probability of masking the goal token during training
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        wandb_log_freq: frequency of logging to wandb
        print_log_freq: frequency of printing to console
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        use_wandb: whether to log to wandb or not
        eval_fraction: fraction of training data to use for evaluation
        eval_freq: frequency of evaluation
    """
    latest_path = os.path.join(project_folder, f"latest.pth")
    ema_model = EMAModel(model=model,power=0.75)
    
    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(
            f"Start ViNT DP Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            train_nomad(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                goal_mask_prob=goal_mask_prob,
                project_folder=project_folder,
                epoch=epoch,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                image_log_freq=image_log_freq,
                num_images_log=num_images_log,
                use_wandb=use_wandb,
                alpha=alpha,
            )
            lr_scheduler.step()

        numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
        torch.save(ema_model.averaged_model.state_dict(), numbered_path)
        numbered_path = os.path.join(project_folder, f"ema_latest.pth")
        print(f"Saved EMA model to {numbered_path}")

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(model.state_dict(), numbered_path)
        torch.save(model.state_dict(), latest_path)
        print(f"Saved model to {numbered_path}")

        # save optimizer
        numbered_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
        latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
        torch.save(optimizer.state_dict(), latest_optimizer_path)

        # save scheduler
        numbered_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
        latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
        torch.save(lr_scheduler.state_dict(), latest_scheduler_path)


        if (epoch + 1) % eval_freq == 0: 
            for dataset_type in test_dataloaders:
                print(
                    f"Start {dataset_type} ViNT DP Testing Epoch {epoch}/{current_epoch + epochs - 1}"
                )
                loader = test_dataloaders[dataset_type]
                evaluate_nomad(
                    eval_type=dataset_type,
                    ema_model=ema_model,
                    dataloader=loader,
                    transform=transform,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    goal_mask_prob=goal_mask_prob,
                    project_folder=project_folder,
                    epoch=epoch,
                    print_log_freq=print_log_freq,
                    num_images_log=num_images_log,
                    wandb_log_freq=wandb_log_freq,
                    use_wandb=use_wandb,
                    eval_fraction=eval_fraction,
                )
        wandb.log({
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        if lr_scheduler is not None:
            lr_scheduler.step()

        # log average eval loss
        wandb.log({}, commit=False)

        wandb.log({
            "lr": optimizer.param_groups[0]["lr"],
        }, commit=False)

        
    # Flush the last set of eval logs
    wandb.log({})
    print()

def load_model(model, model_type, checkpoint: dict) -> None:
    """Load model from checkpoint."""
    if model_type == "nomad":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    else:
        loaded_model = checkpoint["model"]
        try:
            state_dict = loaded_model.module.state_dict()
            model.load_state_dict(state_dict, strict=False)
        except AttributeError as e:
            state_dict = loaded_model.state_dict()
            model.load_state_dict(state_dict, strict=False)


def load_ema_model(ema_model, state_dict: dict) -> None:
    """Load model from checkpoint."""
    ema_model.load_state_dict(state_dict)


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    # print(table)
    print(f"Total Trainable Params: {total_params/1e6:.2f}M")
    return total_params