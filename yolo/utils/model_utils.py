from typing import Any, Dict, List, Type, Union

import torch
import torch.distributed as dist
from omegaconf import ListConfig
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, _LRScheduler

from yolo.config.config import OptimizerConfig, SchedulerConfig
from yolo.model.yolo import YOLO


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    def update(self):
        """Update the shadow parameters using the current model parameters."""
        for name, param in self.model.named_parameters():
            assert name in self.shadow, "All model parameters should have a corresponding shadow parameter."
            new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply the shadow parameters to the model."""
        for name, param in self.model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore the original parameters from the shadow."""
        for name, param in self.model.named_parameters():
            self.shadow[name].copy_(param.data)


def create_optimizer(model: YOLO, optim_cfg: OptimizerConfig) -> Optimizer:
    """Create an optimizer for the given model parameters based on the configuration.

    Returns:
        An instance of the optimizer configured according to the provided settings.
    """
    optimizer_class: Type[Optimizer] = getattr(torch.optim, optim_cfg.type)

    bias_params = [p for name, p in model.named_parameters() if "bias" in name]
    norm_params = [p for name, p in model.named_parameters() if "weight" in name and "bn" in name]
    conv_params = [p for name, p in model.named_parameters() if "weight" in name and "bn" not in name]

    model_parameters = [
        {"params": bias_params, "nestrov": True, "momentum": 0.937},
        {"params": conv_params, "weight_decay": 0.0},
        {"params": norm_params, "weight_decay": 1e-5},
    ]
    return optimizer_class(model_parameters, **optim_cfg.args)


def create_scheduler(optimizer: Optimizer, schedule_cfg: SchedulerConfig) -> _LRScheduler:
    """Create a learning rate scheduler for the given optimizer based on the configuration.

    Returns:
        An instance of the scheduler configured according to the provided settings.
    """
    scheduler_class: Type[_LRScheduler] = getattr(torch.optim.lr_scheduler, schedule_cfg.type)
    schedule = scheduler_class(optimizer, **schedule_cfg.args)
    if hasattr(schedule_cfg, "warmup"):
        wepoch = schedule_cfg.warmup.epochs
        lambda1 = lambda epoch: 0.1 + 0.9 * (epoch + 1 / wepoch) if epoch < wepoch else 1
        lambda2 = lambda epoch: 10 - 9 * (epoch / wepoch) if epoch < wepoch else 1
        warmup_schedule = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2, lambda1])
        schedule = SequentialLR(optimizer, schedulers=[warmup_schedule, schedule], milestones=[2])
    return schedule


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def send_to_device(model: nn.Module, device: Union[str, int, List[int]]):
    if not isinstance(device, (List, ListConfig)):
        device = torch.device(device)
        print("runing man")
        return device, model.to(device)

    device = torch.device("cuda")
    world_size = dist.get_world_size()
    print("runing man")
    dist.init_process_group(
        backend="gloo" if torch.cuda.is_available() else "gloo", rank=dist.get_rank(), world_size=world_size
    )
    print(f"Initialized process group; rank: {dist.get_rank()}, size: {world_size}")

    model = model.cuda(device)
    model = DDP(model, device_ids=[device])
    return device, model.to(device)
