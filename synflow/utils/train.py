import os
import random

import numpy as np
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ChainedScheduler

def get_optimizer(cfg, model: torch.nn.Module):
    param_groups = list(model.parameters())
    if cfg["type"] == "adam":
        return torch.optim.Adam(
            param_groups,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(
                cfg["beta1"],
                cfg["beta2"],
            ),
        )
    elif cfg["type"] == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            betas=(
                cfg["beta1"],
                cfg["beta2"],
            ),
        )
    else:
        raise ValueError(f"Unknown optimizer type: {cfg['type']}")


def get_scheduler(cfg, optimizer: torch.optim.Optimizer):
    if cfg["type"] == "plateau":
        return {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=cfg["factor"],
                patience=cfg["patience"],
                min_lr=cfg["min_lr"],
            ),
            "monitor": "val/loss",
            "interval": "epoch",
            "frequency": 1
        }
    elif cfg["type"] == "cosine":
        schedulers = []
        
        # Warmup scheduler
        if "warmup_steps" in cfg and cfg["warmup_steps"] > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=1e-4,
                end_factor=1.0,
                total_iters=cfg["warmup_steps"]
            )
            schedulers.append(warmup)
        
        # Cosine scheduler
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=cfg["T_max"] - cfg.get("warmup_steps", 0),
            eta_min=cfg["eta_min"]
        )
        schedulers.append(cosine)
        
        return {
            "scheduler": ChainedScheduler(schedulers) if len(schedulers) > 1 else schedulers[0],
            "interval": "step",
            "frequency": 1
        }
    else:
        raise ValueError(f"Unknown scheduler type: {cfg['type']}")


def sum_weighted_losses(losses: dict[str, torch.Tensor], weights: dict[str, float] | None):
    """
    Args:
        losses:     Dict of scalar tensors.
        weights:    Dict of weights.
    """
    loss: torch.Tensor = torch.zeros_like(list(losses.values())[0])
    if weights is None:
        weights = {k: 1.0 for k in losses.keys()}
    for k in losses.keys():
        loss = loss + weights[k] * losses[k]
    return loss


def worker_init_fn(worker_id: int):
    os.sched_setaffinity(0, range(os.cpu_count() or 1))
    global_rank = rank_zero_only.rank
    process_seed = torch.initial_seed()

    base_seed = process_seed - worker_id
    print(
        f"Initializing random number generators of process {global_rank} worker {worker_id} with base seed {base_seed}"
    )
    ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
    # use 128 bits (4 x 32-bit words)
    np.random.seed(ss.generate_state(4))
    # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
    torch_ss, stdlib_ss = ss.spawn(2)
    torch.manual_seed(torch_ss.generate_state(1, dtype=np.uint64)[0])
    # use 128 bits expressed as an integer
    stdlib_seed = (stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
    random.seed(stdlib_seed)


def is_loss_nan_check(loss: torch.Tensor) -> bool:
    """check the validness of the current loss

    Args:
        loss: the loss from the model

    Returns:
        bool: if True, loss is not nan or inf
    """

    def is_nan(x):
        return torch.isnan(x).any() or torch.isinf(x).any()

    nan_flag = torch.tensor(
        1.0 if is_nan(loss) else 0.0,
        device=loss.device if torch.cuda.is_available() else None,
    )  # support cpu
    # avoid "Watchdog caught collective operation timeout" error
    if nan_flag.item() > 0.0:
        return True
    return False