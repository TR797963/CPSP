from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from core.modeling import unwrap_model


class BestCheckpointTracker:
    def __init__(self, metric_name: str = 'mIoU', maximize: bool = True) -> None:
        self.metric_name = metric_name
        self.maximize = maximize
        self.best_value = float('-inf') if maximize else float('inf')
        self.best_epoch = -1
        self.best_path: Optional[str] = None

    def is_better(self, value: float) -> bool:
        return value > self.best_value if self.maximize else value < self.best_value

    def update(self, value: float, epoch: int, path: str) -> bool:
        if self.is_better(value):
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.best_path = path
            return True
        return False


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    run_dir: str,
    filename: str,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    path = Path(run_dir) / filename
    payload: Dict[str, Any] = {
        'epoch': epoch,
        'state_dict': unwrap_model(model).state_dict(),
    }
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, 'state_dict'):
        payload['scheduler'] = scheduler.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))
    return str(path)


def load_training_states(checkpoint: Dict[str, Any], optimizer: Optional[torch.optim.Optimizer], scheduler: Optional[Any]) -> Tuple[int, Dict[str, Any]]:
    start_epoch = int(checkpoint.get('epoch', 0))
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    return start_epoch, checkpoint
