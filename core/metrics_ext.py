from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from skimage import measure


@dataclass
class BatchConfusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    intersection: int = 0
    union: int = 0
    labeled: int = 0
    correct: int = 0


class SegmentationMeter:
    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self.conf = BatchConfusion()

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_bin = (pred > self.threshold).long()
        target_bin = (target > 0.5).long()
        assert pred_bin.shape == target_bin.shape, f'Shape mismatch: {pred_bin.shape} vs {target_bin.shape}'

        tp = int(((pred_bin == 1) & (target_bin == 1)).sum().item())
        fp = int(((pred_bin == 1) & (target_bin == 0)).sum().item())
        tn = int(((pred_bin == 0) & (target_bin == 0)).sum().item())
        fn = int(((pred_bin == 0) & (target_bin == 1)).sum().item())
        inter = tp
        union = tp + fp + fn
        labeled = int((target_bin == 1).sum().item())
        correct = int(((pred_bin == target_bin) & (target_bin == 1)).sum().item())

        self.conf.tp += tp
        self.conf.fp += fp
        self.conf.tn += tn
        self.conf.fn += fn
        self.conf.intersection += inter
        self.conf.union += union
        self.conf.labeled += labeled
        self.conf.correct += correct

    def get(self) -> Dict[str, float]:
        eps = 1e-12
        pix_acc = self.conf.correct / max(self.conf.labeled, 1)
        miou = self.conf.intersection / max(self.conf.union, 1)
        precision = self.conf.tp / max(self.conf.tp + self.conf.fp, 1)
        recall = self.conf.tp / max(self.conf.tp + self.conf.fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, eps)
        return {
            'pixAcc': float(pix_acc),
            'mIoU': float(miou),
            'IoU_pct': float(miou * 100.0),
            'precision': float(precision),
            'recall': float(recall),
            'F1': float(f1),
        }


class PDFAAccumulator:
    """PD/FA metric adapted from BasicIRSTD.

    Engineering note:
    - The implementation follows the common centroid-distance matching rule.
    - When there are no targets in the current evaluation set, PD is defined as 0.
    """

    def __init__(self, match_distance: float = 3.0) -> None:
        self.match_distance = match_distance
        self.reset()

    def reset(self) -> None:
        self.false_alarm_pixels = 0.0
        self.false_alarm_objects = 0
        self.total_pixels = 0
        self.detected_targets = 0
        self.total_targets = 0

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor, size: Tuple[int, int]) -> None:
        pred_np = np.array(pred.detach().cpu()).astype('int64')
        target_np = np.array(target.detach().cpu()).astype('int64')

        pred_cc = measure.label(pred_np, connectivity=2)
        pred_props = list(measure.regionprops(pred_cc))
        gt_cc = measure.label(target_np, connectivity=2)
        gt_props = list(measure.regionprops(gt_cc))

        self.total_targets += len(gt_props)
        true_positive_mask = np.zeros_like(pred_np, dtype=np.int64)

        for gt in gt_props:
            gt_centroid = np.array(list(gt.centroid))
            matched = False
            for idx, pred_obj in enumerate(list(pred_props)):
                pred_centroid = np.array(list(pred_obj.centroid))
                dist = np.linalg.norm(pred_centroid - gt_centroid)
                if dist < self.match_distance:
                    true_positive_mask[pred_obj.coords[:, 0], pred_obj.coords[:, 1]] = 1
                    del pred_props[idx]
                    self.detected_targets += 1
                    matched = True
                    break
            if not matched:
                continue

        self.false_alarm_pixels += float(np.maximum(pred_np - true_positive_mask, 0).sum())
        self.false_alarm_objects += len(pred_props)
        self.total_pixels += int(size[0] * size[1])

    def get(self) -> Dict[str, float]:
        fa = self.false_alarm_pixels / max(self.total_pixels, 1)
        pd = self.detected_targets / max(self.total_targets, 1)
        return {
            'PD': float(pd),
            'FA': float(fa),
            'Pd_pct': float(pd * 100.0),
            'Fa_x1e6': float(fa * 1e6),
            'detected_targets': int(self.detected_targets),
            'total_targets': int(self.total_targets),
            'false_alarm_pixels': int(round(self.false_alarm_pixels)),
            'false_alarm_objects': int(self.false_alarm_objects),
            'total_pixels': int(self.total_pixels),
        }


@torch.no_grad()
def combine_metric_dicts(*metric_dicts: Dict[str, float]) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    for d in metric_dicts:
        merged.update(d)
    return merged
