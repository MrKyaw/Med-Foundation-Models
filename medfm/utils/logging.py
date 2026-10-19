"""Logging and experiment tracking utilities."""

import os
import time
from collections import defaultdict, deque
from typing import Dict, Any, Optional
import torch


class MetricLogger:
    """Log and track metrics during training."""

    def __init__(self, delimiter: str = "  "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs: Any):
        """Update metrics."""
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr: str):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter.avg:.4f}")
        return self.delimiter.join(loss_str)

    def log_every(self, iterable, print_freq: int, header: Optional[str] = None):
        """Log metrics at specified frequency."""
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = self.delimiter.join([
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ])
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(int(eta_seconds))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time),
                ))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(int(total_time))
        print(f"{header} Total time: {total_time_str}s")


class SmoothedValue:
    """Track a series of values and provide access to smoothed values."""

    def __init__(self, window_size: int = 20, fmt: Optional[str] = None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value: float, n: int = 1):
        """Update with new value."""
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self) -> float:
        """Get median value."""
        import statistics
        return statistics.median(self.deque)

    @property
    def avg(self) -> float:
        """Get average value."""
        return sum(self.deque) / len(self.deque) if self.deque else 0.0

    @property
    def global_avg(self) -> float:
        """Get global average."""
        return self.total / self.count if self.count > 0 else 0.0

    @property
    def max(self) -> float:
        """Get maximum value."""
        return max(self.deque) if self.deque else 0.0

    @property
    def value(self) -> float:
        """Get latest value."""
        return self.deque[-1] if self.deque else 0.0

    def __str__(self) -> str:
        if self.fmt:
            return self.fmt.format(
                median=self.median,
                avg=self.avg,
                global_avg=self.global_avg,
                max=self.max,
                value=self.value,
            )
        return f"{self.avg:.4f}"


def setup_logging(
    log_dir: str,
    experiment_name: str,
    rank: int = 0,
) -> str:
    """Setup logging directory and file.

    Args:
        log_dir: Base log directory
        experiment_name: Name of the experiment
        rank: Process rank (only main process logs)

    Returns:
        Path to log directory
    """
    if rank != 0:
        return ""

    log_path = os.path.join(log_dir, experiment_name)
    os.makedirs(log_path, exist_ok=True)

    # Create log file
    log_file = os.path.join(log_path, "log.txt")
    print(f"Logging to: {log_file}")

    return log_path


def log_metrics(metrics: Dict[str, float], step: int, log_dir: str):
    """Metrics to file.

    Args:
        metrics: Dictionary of metrics
        step: Current step
        log_dir: Log directory
    """
    import json

    log_file = os.path.join(log_dir, "metrics.jsonl")
    entry = {"step": step, **metrics}

    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


class TensorBoardLogger:
    """TensorBoard logger."""

    def __init__(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            raise ImportError("tensorboard is required. Install with: pip install tensorboard")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value."""
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag: str, values: Dict[str, float], step: int):
        """Log multiple scalars."""
        self.writer.add_scalars(tag, values, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log histogram."""
        self.writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """Log image."""
        self.writer.add_image(tag, image, step)

    def close(self):
        """Close logger."""
        self.writer.close()
