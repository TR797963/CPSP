import argparse
import copy
import csv
import json
import logging
import math
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


LOGGER_NAME = 'BasicIRSTDRefactor'


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    v = str(v).strip().lower()
    if v in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if v in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Cannot parse boolean from {v!r}.')


class DotDict(dict):
    """Simple dict with attribute access."""

    def __getattr__(self, item: str) -> Any:
        if item in self:
            return self[item]
        raise AttributeError(item)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def copy(self) -> 'DotDict':
        return DotDict(copy.deepcopy(dict(self)))


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()
        return super().default(obj)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


class CSVLogger:
    def __init__(self, csv_path: str) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames: Optional[List[str]] = None

    def append(self, row: Dict[str, Any]) -> None:
        flat = flatten_dict(row)
        if self.fieldnames is None:
            self.fieldnames = list(flat.keys())
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerow(flat)
            return
        missing = [k for k in flat.keys() if k not in self.fieldnames]
        if missing:
            self.fieldnames += missing
            existing: List[Dict[str, Any]] = []
            if self.csv_path.exists():
                with open(self.csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    existing = list(reader)
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                for item in existing:
                    writer.writerow(item)
                writer.writerow(flat)
            return
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(flat)


class JSONLLogger:
    def __init__(self, jsonl_path: str) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: Dict[str, Any]) -> None:
        with open(self.jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, cls=NumpyJSONEncoder, ensure_ascii=False) + '\n')


class ExperimentLogger:
    """Text + csv + jsonl logger."""

    def __init__(self, run_dir: str, txt_name: str = 'log.txt', csv_name: str = 'records.csv', jsonl_name: str = 'records.jsonl') -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.text_path = self.run_dir / txt_name
        self.csv_logger = CSVLogger(str(self.run_dir / csv_name))
        self.jsonl_logger = JSONLLogger(str(self.run_dir / jsonl_name))
        self.logger = logging.getLogger(f'{LOGGER_NAME}.{self.run_dir.name}.{id(self)}')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh = logging.FileHandler(self.text_path, encoding='utf-8')
        fh.setFormatter(formatter)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def log_record(self, row: Dict[str, Any], prefix: Optional[str] = None) -> None:
        row = copy.deepcopy(row)
        if prefix is not None:
            row['_prefix'] = prefix
        self.csv_logger.append(row)
        self.jsonl_logger.append(row)


class MetricHistory:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def append(self, row: Dict[str, Any]) -> None:
        self.rows.append(copy.deepcopy(row))

    def to_csv(self, path: str) -> None:
        logger = CSVLogger(path)
        for row in self.rows:
            logger.append(row)

    def to_json(self, path: str) -> None:
        save_json(self.rows, path)


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_text(lines: Sequence[str], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(str(line) + '\n')


def copy_file(src: str, dst: str) -> None:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def flatten_dict(d: Dict[str, Any], prefix: str = '', sep: str = '.') -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f'{prefix}{sep}{k}' if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key, sep))
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(v, cls=NumpyJSONEncoder, ensure_ascii=False)
        else:
            out[key] = v
    return out


def to_device(batch: Sequence[Any], device: torch.device) -> List[Any]:
    moved: List[Any] = []
    for item in batch:
        if torch.is_tensor(item):
            moved.append(item.to(device, non_blocking=True))
        else:
            moved.append(item)
    return moved


_DATASET_NORM_CFG = {
    'NUAA-SIRST': {'mean': 101.06385040283203, 'std': 34.619606018066406},
    'NUDT-SIRST': {'mean': 107.80905151367188, 'std': 33.02274703979492},
    'IRSTD-1K': {'mean': 87.4661865234375, 'std': 39.71953201293945},
    'NUDT-SIRST-Sea': {'mean': 43.62403869628906, 'std': 18.91838264465332},
    'SIRST4': {'mean': 62.10432052612305, 'std': 23.96998405456543},
    'IRDST-real': {'mean': 101.54053497314453, 'std': 56.49856185913086},
    'LimitIRTSTD-track2': {'mean': 64.21671295166016, 'std': 24.50885772705078},
    'SIRST-v1': {'mean': 101.06385040283203, 'std': 34.619606018066406},
}


def compute_dataset_norm_cfg(dataset_name: str, dataset_dir: str) -> Dict[str, float]:
    if dataset_name in _DATASET_NORM_CFG:
        return copy.deepcopy(_DATASET_NORM_CFG[dataset_name])

    base = Path(dataset_dir) / dataset_name
    train_txt = base / 'img_idx' / f'train_{dataset_name}.txt'
    test_txt = base / 'img_idx' / f'test_{dataset_name}.txt'
    if not train_txt.exists() and (base / 'img_idx' / 'train.txt').exists():
        train_txt = base / 'img_idx' / 'train.txt'
    img_list: List[str] = []
    if train_txt.exists():
        img_list.extend(train_txt.read_text(encoding='utf-8').splitlines())
    if test_txt.exists():
        img_list.extend(test_txt.read_text(encoding='utf-8').splitlines())
    img_dir = base / 'images'
    means: List[float] = []
    stds: List[float] = []
    for stem in img_list:
        img = read_gray_image(img_dir, stem)
        arr = np.array(img, dtype=np.float32)
        means.append(float(arr.mean()))
        stds.append(float(arr.std()))
    if not means:
        raise FileNotFoundError(f'Cannot compute mean/std for dataset={dataset_name}. No image list found in {base}.')
    return {'mean': float(np.mean(means)), 'std': float(np.mean(stds) + 1e-6)}


def normalized(img: np.ndarray, img_norm_cfg: Dict[str, float]) -> np.ndarray:
    return (img - img_norm_cfg['mean']) / max(img_norm_cfg['std'], 1e-6)


def denormalized(img: np.ndarray, img_norm_cfg: Dict[str, float]) -> np.ndarray:
    return img * img_norm_cfg['std'] + img_norm_cfg['mean']


def pad_image(img: np.ndarray, times: int = 32) -> np.ndarray:
    h, w = img.shape
    if h % times != 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0)), mode='constant')
    if w % times != 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w)), mode='constant')
    return img


def random_crop(img: np.ndarray, mask: np.ndarray, patch_size: int, pos_prob: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        mask = np.pad(mask, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode='constant')
        h, w = img.shape

    cur_prob = random.random()
    if pos_prob is None or cur_prob > pos_prob or mask.max() == 0:
        h_start = random.randint(0, h - patch_size)
        w_start = random.randint(0, w - patch_size)
    else:
        loc = np.where(mask > 0)
        idx = random.randint(0, max(len(loc[0]) - 1, 0)) if len(loc[0]) > 0 else 0
        h_start = random.randint(max(0, int(loc[0][idx]) - patch_size), min(int(loc[0][idx]), h - patch_size))
        w_start = random.randint(max(0, int(loc[1][idx]) - patch_size), min(int(loc[1][idx]), w - patch_size))
    h_end, w_end = h_start + patch_size, w_start + patch_size
    return img[h_start:h_end, w_start:w_end], mask[h_start:h_end, w_start:w_end]


class Augmentation:
    def __call__(self, image: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = image[::-1, :]
            target = target[::-1, :]
        if random.random() < 0.5:
            image = image[:, ::-1]
            target = target[:, ::-1]
        if random.random() < 0.5:
            image = image.transpose(1, 0)
            target = target.transpose(1, 0)
        return image, target


def read_gray_image(img_dir: Path, stem: str) -> Image.Image:
    for ext in ['.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff']:
        path = img_dir / f'{stem}{ext}'
        if path.exists():
            return Image.open(path).convert('I')
    raise FileNotFoundError(f'Cannot find image for stem={stem} under {img_dir}.')


def read_mask_image(mask_dir: Path, stem: str) -> Image.Image:
    for ext in ['.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff']:
        path = mask_dir / f'{stem}{ext}'
        if path.exists():
            return Image.open(path)
    raise FileNotFoundError(f'Cannot find mask for stem={stem} under {mask_dir}.')


def list_to_pretty_lines(items: Iterable[Dict[str, Any]], title: Optional[str] = None) -> List[str]:
    lines: List[str] = []
    if title is not None:
        lines.append(title)
    for item in items:
        lines.append(json.dumps(item, cls=NumpyJSONEncoder, ensure_ascii=False))
    return lines


def profile_model_safe(model: torch.nn.Module, input_size: Tuple[int, int, int, int], device: torch.device) -> Dict[str, Any]:
    """Try to compute Params/FLOPs. If THOP is unavailable, only return params.

    Notes:
    - FLOPs are optional; failure should not interrupt the main flow.
    - THOP registers temporary ``total_ops``/``total_params`` buffers.  It must
      therefore run on a deep copy so profiling can never change checkpoints
      subsequently saved from the caller's model.
    - For masked pruning we report *effective* params/FLOPs separately in pruning reports.
    """
    stats: Dict[str, Any] = {'params': None, 'flops': None, 'warning': None}
    stats['params'] = int(sum(p.numel() for p in model.parameters()))
    try:
        from thop import profile  # type: ignore

        # Never fall back to profiling ``model`` itself: if deepcopy or device
        # placement is unsupported, a params-only result is safer than silently
        # adding THOP buffers to the training model.
        profile_device = torch.device(device)
        profile_model = copy.deepcopy(model)
        profile_model = profile_model.to(profile_device)
        custom_ops = {}
        try:
            from compat.isnet import TorchvisionDeformConvPack

            def count_deform_conv(module: torch.nn.Module, inputs: Any, output: torch.Tensor) -> None:
                kernel_ops = (
                    int(module.in_channels)
                    // int(module.groups)
                    * int(module.kernel_size[0])
                    * int(module.kernel_size[1])
                )
                module.total_ops += torch.DoubleTensor(
                    [float(output.numel() * kernel_ops)]
                )

            # The internal offset Conv2d is counted by THOP separately. This
            # hook adds only the main deformable-convolution MACs.
            custom_ops[TorchvisionDeformConvPack] = count_deform_conv
        except Exception:
            pass
        dummy = torch.randn(*input_size, device=profile_device)
        profile_model.eval()
        with torch.no_grad():
            flops, _ = profile(
                profile_model,
                inputs=(dummy,),
                custom_ops=custom_ops,
                verbose=False,
            )
        stats['flops'] = float(flops)
    except Exception as exc:  # pragma: no cover - optional dependency / model specifics
        stats['warning'] = f'FLOPs profiling skipped: {exc}'
    return stats


@torch.no_grad()
def measure_inference_speed(model: torch.nn.Module, input_size: Tuple[int, int, int, int], device: torch.device, repeats: int = 50) -> Dict[str, float]:
    model.eval()
    dummy = torch.randn(*input_size, device=device)
    for _ in range(10):
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    ms = elapsed * 1000.0 / max(repeats, 1)
    fps = max(repeats, 1) / max(elapsed, 1e-9)
    return {'latency_ms': float(ms), 'fps': float(fps)}


def plot_curve(series: Dict[str, Sequence[float]], out_path: str, title: str, xlabel: str, ylabel: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for name, values in series.items():
        if not values:
            continue
        plt.plot(range(1, len(values) + 1), values, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.4)
    if len(series) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bar(labels: Sequence[str], values: Sequence[float], out_path: str, title: str, xlabel: str, ylabel: str, rotate_xticks: bool = True) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(8, len(labels) * 0.35), 5))
    plt.bar(range(len(labels)), values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.4, axis='y')
    plt.xticks(range(len(labels)), labels, rotation=45 if rotate_xticks else 0, ha='right' if rotate_xticks else 'center')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_hist(values: Sequence[float], out_path: str, title: str, xlabel: str, ylabel: str = 'Count', bins: int = 20) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_simple_csv(rows: Sequence[Dict[str, Any]], path: str) -> None:
    logger = CSVLogger(path)
    for row in rows:
        logger.append(row)


def namespace_to_dict(ns: Any) -> Dict[str, Any]:
    if isinstance(ns, dict):
        return copy.deepcopy(ns)
    return {k: copy.deepcopy(v) for k, v in vars(ns).items()}


def select_device(device_str: str) -> torch.device:
    if device_str == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    if device_str.startswith('cuda') and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device('cpu')


def safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    return float(np.mean(values)) if len(values) > 0 else float(default)


def safe_std(values: Sequence[float], default: float = 0.0) -> float:
    return float(np.std(values)) if len(values) > 0 else float(default)


def resolve_best_metric(metrics: Dict[str, float], metric_name: str) -> float:
    if metric_name not in metrics:
        raise KeyError(f'Metric {metric_name} not found in {list(metrics.keys())}.')
    return float(metrics[metric_name])
