from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from core.utils import Augmentation, compute_dataset_norm_cfg, normalized, pad_image, random_crop, read_gray_image, read_mask_image, str2bool


def _read_index_file(path: Path) -> List[str]:
    return [
        item.strip()
        for item in path.read_text(encoding='utf-8').splitlines()
        if item.strip()
    ]


def _validate_index_subset(
    requested: List[str],
    official: List[str],
    configured_path: str,
    split: str,
) -> List[str]:
    if not requested:
        raise ValueError(
            f'Configured analysis index {configured_path!r} is empty.'
        )
    seen = set()
    duplicates = []
    for item in requested:
        if item in seen and item not in duplicates:
            duplicates.append(item)
        seen.add(item)
    if duplicates:
        raise ValueError(
            f'Configured analysis index {configured_path!r} contains duplicate '
            f'IDs: {duplicates[:20]}'
        )
    official_ids = set(official)
    outside = [item for item in requested if item not in official_ids]
    if outside:
        raise ValueError(
            f'Configured analysis index {configured_path!r} contains IDs '
            f'outside the official {split!r} split: {outside[:20]}'
        )
    return requested


class IRSTDDataset(Dataset):
    """Generic dataset wrapper compatible with BasicIRSTD folder layout.

    Expected layout:
    datasets/<dataset_name>/
      images/*.png|bmp|jpg
      masks/*.png|bmp|jpg
      img_idx/train_<dataset>.txt or train.txt
      img_idx/test_<dataset>.txt or test.txt
    """

    def __init__(
        self,
        dataset_dir: str,
        dataset_name: str,
        split: str,
        patch_size: int = 256,
        img_norm_cfg: Optional[Dict[str, float]] = None,
        pos_prob: float = 0.5,
        return_name: bool = False,
        deterministic_eval: bool = False,
        index_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert split in {'train', 'test', 'val'}
        self.dataset_name = dataset_name
        self.base_dir = Path(dataset_dir) / dataset_name
        self.split = 'test' if split == 'val' else split
        self.patch_size = patch_size
        self.return_name = return_name
        self.pos_prob = pos_prob
        self.deterministic_eval = bool(deterministic_eval)
        self.index_path = index_path
        self.augmentation = Augmentation()
        self.img_norm_cfg = img_norm_cfg or compute_dataset_norm_cfg(dataset_name, dataset_dir)
        self.indices = self._load_index_list()

    def _official_index_path(self) -> Path:
        img_idx_dir = self.base_dir / 'img_idx'
        primary = img_idx_dir / f'{self.split}_{self.dataset_name}.txt'
        fallback = img_idx_dir / f'{self.split}.txt'
        if primary.exists():
            return primary
        if fallback.exists():
            return fallback
        raise FileNotFoundError(
            f'Cannot find split list for {self.dataset_name}/{self.split} '
            f'under {img_idx_dir}.'
        )

    def _load_index_list(self) -> List[str]:
        img_idx_dir = self.base_dir / 'img_idx'
        official_path = self._official_index_path()
        official_indices = _read_index_file(official_path)
        if self.index_path is not None:
            configured_value = str(self.index_path).strip()
            if not configured_value:
                raise ValueError('Configured analysis index path is empty.')
            configured = Path(configured_value).expanduser()
            candidates = [configured]
            if not configured.is_absolute():
                candidates.extend(
                    [self.base_dir / configured, img_idx_dir / configured]
                )
            path = next((candidate for candidate in candidates if candidate.exists()), None)
            if path is None:
                checked = ', '.join(str(candidate) for candidate in candidates)
                raise FileNotFoundError(
                    f'Cannot find configured index_path={self.index_path!r}. '
                    f'Checked: {checked}'
                )
            return _validate_index_subset(
                _read_index_file(path),
                official_indices,
                str(path),
                self.split,
            )
        return official_indices

    def __len__(self) -> int:
        return len(self.indices)

    def _read_pair(self, stem: str) -> Tuple[np.ndarray, np.ndarray]:
        img = np.array(read_gray_image(self.base_dir / 'images', stem), dtype=np.float32)
        mask = np.array(read_mask_image(self.base_dir / 'masks', stem), dtype=np.float32) / 255.0
        if mask.ndim > 2:
            mask = mask[..., 0]
        return normalized(img, self.img_norm_cfg), mask

    def __getitem__(self, index: int) -> Tuple[Any, ...]:
        stem = self.indices[index]
        img, mask = self._read_pair(stem)
        if self.split == 'train' and not self.deterministic_eval:
            img, mask = random_crop(img, mask, self.patch_size, pos_prob=self.pos_prob)
            img, mask = self.augmentation(img, mask)
            h, w = img.shape
        else:
            h, w = img.shape
            img = pad_image(img)
            mask = pad_image(mask)

        img = torch.from_numpy(np.ascontiguousarray(img[np.newaxis, ...])).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask[np.newaxis, ...])).float()
        size = torch.tensor([h, w], dtype=torch.long)
        if self.return_name or self.split != 'train':
            return img, mask, size, stem
        return img, mask


class InferenceImageDataset(Dataset):
    def __init__(self, dataset_dir: str, dataset_name: str, img_norm_cfg: Optional[Dict[str, float]] = None) -> None:
        super().__init__()
        self.base_dir = Path(dataset_dir) / dataset_name
        self.img_norm_cfg = img_norm_cfg or compute_dataset_norm_cfg(dataset_name, dataset_dir)
        img_idx_dir = self.base_dir / 'img_idx'
        test_list = img_idx_dir / f'test_{dataset_name}.txt'
        if not test_list.exists():
            test_list = img_idx_dir / 'test.txt'
        self.indices = [x.strip() for x in test_list.read_text(encoding='utf-8').splitlines() if x.strip()]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        stem = self.indices[index]
        img = np.array(read_gray_image(self.base_dir / 'images', stem), dtype=np.float32)
        h, w = img.shape
        img = normalized(img, self.img_norm_cfg)
        img = pad_image(img)
        tensor = torch.from_numpy(np.ascontiguousarray(img[np.newaxis, ...])).float()
        return tensor, torch.tensor([h, w], dtype=torch.long), stem


def _evaluation_num_workers(num_workers: int) -> int:
    """Honor the single-process setting for test and analysis loaders too."""
    count = int(num_workers)
    if count < 0:
        raise ValueError('num_workers must be >= 0.')
    return 0 if count == 0 else max(1, count // 2)


def build_dataloaders(cfg: Any, return_name_for_train: bool = False) -> Tuple[DataLoader, DataLoader]:
    evaluation_workers = _evaluation_num_workers(cfg.num_workers)
    train_set = IRSTDDataset(
        dataset_dir=cfg.dataset_dir,
        dataset_name=cfg.dataset,
        split='train',
        patch_size=cfg.patch_size,
        img_norm_cfg=cfg.img_norm_cfg,
        return_name=return_name_for_train,
    )
    val_set = IRSTDDataset(
        dataset_dir=cfg.dataset_dir,
        dataset_name=cfg.dataset,
        split='test',
        patch_size=cfg.patch_size,
        img_norm_cfg=cfg.img_norm_cfg,
        return_name=True,
    )
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=evaluation_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader


def build_analysis_loader(cfg: Any, subset_size: Optional[int] = None) -> DataLoader:
    evaluation_workers = _evaluation_num_workers(cfg.num_workers)
    # Analysis is a method-design input, so the default must not consume the
    # official test split.  A train-index sample is still evaluated with the
    # deterministic pad-only path (no crop or random augmentation).
    analysis_split = str(getattr(cfg, 'analysis_split', 'train')).strip().lower()
    if analysis_split not in {'train', 'test'}:
        raise ValueError(
            f'analysis_split must be train or test; got {analysis_split!r}.'
        )
    analysis_allow_test = str2bool(
        getattr(cfg, 'analysis_allow_test', False)
    )
    if analysis_split == 'test' and not analysis_allow_test:
        raise ValueError(
            f'Refusing analysis_split={analysis_split!r}: this resolves to the '
            'official test index and risks evaluation leakage. Set '
            '--analysis_allow_test true only for an explicitly authorized '
            'diagnostic.'
        )
    analysis_index = getattr(cfg, 'analysis_index', None)
    dataset = IRSTDDataset(
        dataset_dir=cfg.dataset_dir,
        dataset_name=cfg.dataset,
        split=analysis_split,
        patch_size=cfg.patch_size,
        img_norm_cfg=cfg.img_norm_cfg,
        return_name=True,
        deterministic_eval=True,
        index_path=analysis_index,
    )
    if subset_size is not None and subset_size > 0 and subset_size < len(dataset):
        analysis_seed = int(getattr(cfg, 'analysis_seed', 42))
        g = torch.Generator().manual_seed(analysis_seed)
        perm = torch.randperm(len(dataset), generator=g).tolist()
        dataset = Subset(dataset, perm[:subset_size])
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=evaluation_workers, pin_memory=True)
