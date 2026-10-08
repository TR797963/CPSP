import copy
import math
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

try:
    import torch_pruning as tp
except Exception:
    tp = None

from core.config import (
    normalize_layer_min_remaining_rules,
    normalize_layer_prune_cap_rules,
    parse_layer_min_remaining_rule,
    parse_layer_prune_cap_rule,
)
from core.modeling import get_module_by_name, named_modules_dict, unwrap_model
from core.utils import save_json, save_simple_csv
from pruning.auto_select import LayerMeta, SelectionBundle


@dataclass
class LayerPlan:
    layer_name: str
    delete_count: int
    keep_count: int
    prune_ratio: float
    deleted_indices: List[int]
    kept_indices: List[int]
    scores: List[float]
    q_values: Optional[List[float]] = None
    u_values: Optional[List[float]] = None
    gain_values: Optional[List[float]] = None
    priority_values: Optional[List[float]] = None
    layer_budget_weight: Optional[float] = None
    budget_before_correction: Optional[float] = None
    budget_after_correction: Optional[int] = None
    max_delete_cap: Optional[int] = None
    min_delete_floor: Optional[int] = None
    high_q_retention: Optional[float] = None
    hook_module_name: Optional[str] = None
    effective_max_prune_ratio: Optional[float] = None
    matched_layer_prune_cap_rules: Optional[List[str]] = None
    effective_layer_prune_cap_rule: Optional[str] = None
    effective_min_remaining_channels: Optional[int] = None
    matched_layer_min_remaining_rules: Optional[List[str]] = None
    effective_layer_min_remaining_rule: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'layer_name': self.layer_name,
            'delete_count': self.delete_count,
            'keep_count': self.keep_count,
            'prune_ratio': self.prune_ratio,
            'deleted_indices': self.deleted_indices,
            'kept_indices': self.kept_indices,
            'scores': self.scores,
            'q_values': self.q_values,
            'u_values': self.u_values,
            'gain_values': self.gain_values,
            'priority_values': self.priority_values,
            'layer_budget_weight': self.layer_budget_weight,
            'budget_before_correction': self.budget_before_correction,
            'budget_after_correction': self.budget_after_correction,
            'max_delete_cap': self.max_delete_cap,
            'min_delete_floor': self.min_delete_floor,
            'high_q_retention': self.high_q_retention,
            'hook_module_name': self.hook_module_name,
            'effective_max_prune_ratio': self.effective_max_prune_ratio,
            'matched_layer_prune_cap_rules': self.matched_layer_prune_cap_rules,
            'effective_layer_prune_cap_rule': self.effective_layer_prune_cap_rule,
            'effective_min_remaining_channels': self.effective_min_remaining_channels,
            'matched_layer_min_remaining_rules': self.matched_layer_min_remaining_rules,
            'effective_layer_min_remaining_rule': self.effective_layer_min_remaining_rule,
        }


class ChannelMaskApplier:
    """Apply physical channel pruning with torch-pruning / DepGraph.

    This upgraded backend keeps the existing pruning planner / budget allocator / logging API,
    but replaces forward-hook masking with real structural pruning. After `set_masks_from_plans`,
    the model held by this applier is already compacted in-place.

    Compatibility notes:
    - `masks` is intentionally preserved as an exported bookkeeping view of the original plan so
      existing logs and finetune entry points can be reused.
    - `rebind()` becomes a lightweight model-reference refresh for device move / DataParallel wrap.
    - For dependency-coupled branches, the final compact structure is governed by DepGraph. When a
      previously pruned dependency also changes another selected layer, later pruning requests are
      reconciled against the layer's remaining live channels.
    """

    def __init__(self, model: nn.Module, bundle: SelectionBundle, default_input_size: Tuple[int, int, int, int] = (1, 1, 256, 256)) -> None:
        self.model = unwrap_model(model)
        self.bundle = bundle
        self.default_input_size = tuple(int(x) for x in default_input_size)
        self.handles: List[Any] = []
        self.masks: Dict[str, torch.Tensor] = {}
        self.hook_targets: Dict[str, str] = {}
        self.active_index_map: Dict[str, List[int]] = {}
        self.pruned = False

    def _require_tp(self) -> None:
        if tp is None:
            raise ImportError(
                'torch_pruning is required for the physical pruning backend. '
                'Please install it (for example via requirements_refactor.txt) before running sweep_pruning.'
            )

    def _resolve_hook_module_name(self, meta: LayerMeta) -> str:
        return meta.matched_bn if meta.matched_bn is not None else meta.name

    def _infer_example_input_size(self, input_size: Optional[Tuple[int, int, int, int]] = None) -> Tuple[int, int, int, int]:
        if input_size is not None:
            return tuple(int(x) for x in input_size)
        first_conv = next((m for m in self.model.modules() if isinstance(m, nn.Conv2d)), None)
        in_channels = int(first_conv.in_channels) if first_conv is not None else int(self.default_input_size[1])
        n, _, h, w = self.default_input_size
        return (int(n), in_channels, int(h), int(w))

    def _build_masks_from_plan(self, plans: Dict[str, Any], device: torch.device) -> None:
        self.masks = {}
        self.hook_targets = {}
        self.active_index_map = {}
        for meta in self.bundle.selected_layers:
            if meta.name not in plans:
                continue
            plan_obj = plans[meta.name]
            plan = plan_obj if isinstance(plan_obj, LayerPlan) else LayerPlan(**plan_obj)
            keep = set(plan.kept_indices)
            mask = torch.zeros(meta.out_channels, device=device, dtype=torch.float32)
            if keep:
                mask[list(sorted(keep))] = 1.0
            self.hook_targets[meta.name] = plan.hook_module_name or self._resolve_hook_module_name(meta)
            self.masks[meta.name] = mask
            self.active_index_map[meta.name] = list(range(meta.out_channels))

    def _refresh_model_reference(self, new_model: nn.Module) -> None:
        self.model = unwrap_model(new_model)

    def _translate_original_delete_to_current(self, layer_name: str, deleted_indices: Sequence[int]) -> List[int]:
        live_original = self.active_index_map.get(layer_name)
        if live_original is None:
            module = get_module_by_name(self.model, layer_name)
            if not isinstance(module, nn.Conv2d):
                return []
            live_original = list(range(int(module.out_channels)))
            self.active_index_map[layer_name] = live_original
        delete_set = set(int(x) for x in deleted_indices)
        return [cur_idx for cur_idx, orig_idx in enumerate(live_original) if orig_idx in delete_set]

    def _apply_direct_bookkeeping(self, layer_name: str, current_delete_indices: Sequence[int]) -> None:
        if layer_name not in self.active_index_map:
            return
        delete_set = set(int(x) for x in current_delete_indices)
        self.active_index_map[layer_name] = [orig for cur, orig in enumerate(self.active_index_map[layer_name]) if cur not in delete_set]

    def _reconcile_selected_layer_maps(self) -> None:
        named = named_modules_dict(self.model)
        for meta in self.bundle.selected_layers:
            module = named.get(meta.name)
            if not isinstance(module, nn.Conv2d):
                continue
            current_out = int(module.out_channels)
            live = self.active_index_map.get(meta.name)
            if live is None:
                self.active_index_map[meta.name] = list(range(current_out))
                continue
            if len(live) > current_out:
                self.active_index_map[meta.name] = live[:current_out]
            elif len(live) < current_out:
                # Unexpected expansion should not happen, but keep bookkeeping consistent if it does.
                max_orig = max(live) + 1 if live else 0
                self.active_index_map[meta.name] = live + list(range(max_orig, max_orig + (current_out - len(live))))

    def set_masks_from_plans(
        self,
        plans: Dict[str, Any],
        device: torch.device,
        input_size: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        self._require_tp()
        self.clear()
        self._build_masks_from_plan(plans, device=device)
        self.model.eval()
        example_input_size = self._infer_example_input_size(input_size)
        example_inputs = torch.randn(*example_input_size, device=device)
        dg = tp.DependencyGraph().build_dependency(self.model.to(device), example_inputs=example_inputs)

        for meta in self.bundle.selected_layers:
            if meta.name not in plans:
                continue
            module = get_module_by_name(self.model, meta.name)
            if not isinstance(module, nn.Conv2d):
                continue
            plan_obj = plans[meta.name]
            plan = plan_obj if isinstance(plan_obj, LayerPlan) else LayerPlan(**plan_obj)
            current_delete = self._translate_original_delete_to_current(meta.name, plan.deleted_indices)
            if not current_delete:
                continue
            current_delete = [idx for idx in current_delete if idx < int(module.out_channels)]
            if not current_delete:
                continue
            pruning_fn = tp.prune_conv_out_channels
            try:
                group = dg.get_pruning_group(module, pruning_fn, idxs=current_delete)
            except Exception as exc:
                raise RuntimeError(f'Failed to build DepGraph pruning group for {meta.name}: {exc}') from exc
            if not dg.check_pruning_group(group):
                continue
            group.prune()
            self._apply_direct_bookkeeping(meta.name, current_delete)
            self._reconcile_selected_layer_maps()

        self.pruned = True

    def clear(self, keep_masks: bool = False) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if not keep_masks:
            self.masks = {}
            self.hook_targets = {}
            self.active_index_map = {}
            self.pruned = False

    def rebind(self, new_model: nn.Module) -> None:
        self._refresh_model_reference(new_model)

    def export_mask_dict(self) -> Dict[str, List[float]]:
        return {name: mask.detach().cpu().tolist() for name, mask in self.masks.items()}


def _l1_scores_for_conv(conv: nn.Conv2d) -> np.ndarray:
    weight = conv.weight.detach().abs().flatten(1)
    return weight.sum(dim=1).cpu().numpy()



def _bn_gamma_scores(bn: nn.BatchNorm2d) -> np.ndarray:
    return bn.weight.detach().abs().cpu().numpy()



def compute_layer_scores(model: nn.Module, bundle: SelectionBundle, score_source: str) -> Dict[str, np.ndarray]:
    model = unwrap_model(model)
    scores: Dict[str, np.ndarray] = {}
    for meta in bundle.selected_layers:
        if score_source in {'l1', 'ours_l1'}:
            conv = get_module_by_name(model, meta.name)
            assert isinstance(conv, nn.Conv2d)
            scores[meta.name] = _l1_scores_for_conv(conv)
        elif score_source in {'slim', 'ours_slim'}:
            if meta.matched_bn is None:
                raise RuntimeError(f'Layer {meta.name} has no matched BN, cannot use slim score.')
            bn = get_module_by_name(model, meta.matched_bn)
            assert isinstance(bn, nn.BatchNorm2d)
            scores[meta.name] = _bn_gamma_scores(bn)
        else:
            raise ValueError(f'Unknown score source: {score_source}')
    return scores



def _round_to_int(values: Sequence[float], mode: str) -> List[int]:
    out: List[int] = []
    for value in values:
        if mode == 'floor':
            out.append(int(math.floor(value)))
        elif mode == 'ceil':
            out.append(int(math.ceil(value)))
        else:
            out.append(int(round(value)))
    return out



def _safe_delete_cap(channels: int, min_remaining: int, max_ratio: Optional[float] = None) -> int:
    cap = max(0, int(channels - min_remaining))
    if max_ratio is not None:
        cap = min(cap, int(math.floor(channels * max_ratio)))
    return max(cap, 0)


def resolve_layer_prune_cap(
    layer_name: str,
    max_layer_prune_ratio: float,
    layer_prune_cap_rules: Any,
) -> Dict[str, Any]:
    """Resolve the strictest fnmatch cap for one layer.

    The global cap always participates in the minimum. Rule order is retained
    for reproducibility; if equally strict rules match, the first one is the
    reported controlling rule while all matches remain recorded.
    """
    normalized_rules = normalize_layer_prune_cap_rules(layer_prune_cap_rules)
    effective_ratio = float(max_layer_prune_ratio)
    effective_rule = 'max_layer_prune_ratio'
    matched_rules: List[str] = []
    for rule in normalized_rules:
        pattern, ratio, canonical = parse_layer_prune_cap_rule(rule)
        if not fnmatchcase(layer_name, pattern):
            continue
        matched_rules.append(canonical)
        if ratio < effective_ratio:
            effective_ratio = ratio
            effective_rule = canonical
    return {
        'effective_max_prune_ratio': float(effective_ratio),
        'matched_layer_prune_cap_rules': matched_rules,
        'effective_layer_prune_cap_rule': effective_rule,
    }


def resolve_layer_min_remaining(
    layer_name: str,
    min_remaining_channels_per_layer: int,
    layer_min_remaining_rules: Any,
) -> Dict[str, Any]:
    """Resolve the strictest (largest) minimum remaining width for one layer."""
    normalized_rules = normalize_layer_min_remaining_rules(layer_min_remaining_rules)
    effective_minimum = int(min_remaining_channels_per_layer)
    effective_rule = 'min_remaining_channels_per_layer'
    matched_rules: List[str] = []
    for rule in normalized_rules:
        pattern, minimum, canonical = parse_layer_min_remaining_rule(rule)
        if not fnmatchcase(layer_name, pattern):
            continue
        matched_rules.append(canonical)
        if minimum > effective_minimum:
            effective_minimum = minimum
            effective_rule = canonical
    return {
        'effective_min_remaining_channels': int(effective_minimum),
        'matched_layer_min_remaining_rules': matched_rules,
        'effective_layer_min_remaining_rule': effective_rule,
    }



def _layer_prunability(q_values: np.ndarray, s_values: np.ndarray, u_values: np.ndarray, metric: str) -> float:
    eps = 1e-8
    q_norm = 1.0 - np.clip(q_values, 0.0, 1.0)
    s_norm = 1.0 - (s_values - s_values.min()) / max(s_values.max() - s_values.min(), eps)
    u_norm = 1.0 - (u_values - u_values.min()) / max(u_values.max() - u_values.min(), eps)
    if metric == 'mean_inverse_q':
        return float(np.mean(q_norm))
    if metric == 'mean_inverse_s':
        return float(np.mean(s_norm))
    if metric == 'mean_inverse_u':
        return float(np.mean(u_norm))
    return float(np.mean((q_norm + s_norm + u_norm) / 3.0))



def _compute_high_q_retention(q_values: np.ndarray, kept_indices: Sequence[int], threshold: float = 0.7) -> float:
    high_idx = np.where(q_values >= threshold)[0]
    if len(high_idx) == 0:
        return 1.0
    kept = set(kept_indices)
    kept_high = sum(int(idx in kept) for idx in high_idx.tolist())
    return float(kept_high / max(len(high_idx), 1))



def _sort_indices_by_score(scores: np.ndarray) -> np.ndarray:
    return np.argsort(scores, kind='stable')


class BaselinePruner:
    """Baseline per-layer pruning.

    Important: l1/slim stay *per-layer fixed-ratio* baselines. They do not use global budget.
    Safety clipping is only used to avoid deleting all channels in a layer.
    """

    def __init__(self, model: nn.Module, bundle: SelectionBundle, cfg: Any, method: str) -> None:
        assert method in {'l1', 'slim'}
        self.model = unwrap_model(model)
        self.bundle = bundle
        self.cfg = cfg
        self.method = method
        self.score_source = 'l1' if method == 'l1' else 'slim'

    def plan(self, prune_ratio: float) -> Dict[str, Any]:
        scores = compute_layer_scores(self.model, self.bundle, self.score_source)
        plans: Dict[str, LayerPlan] = {}
        rows: List[Dict[str, Any]] = []
        for meta in self.bundle.selected_layers:
            s = scores[meta.name]
            delete_cap = max(0, meta.out_channels - int(self.cfg.min_remaining_channels_per_layer))
            delete_count = min(int(round(prune_ratio * meta.out_channels)), delete_cap)
            sorted_idx = _sort_indices_by_score(s)
            deleted = sorted_idx[:delete_count].tolist()
            kept = sorted(sorted_idx[delete_count:].tolist())
            plan = LayerPlan(
                layer_name=meta.name,
                delete_count=delete_count,
                keep_count=len(kept),
                prune_ratio=float(delete_count / max(meta.out_channels, 1)),
                deleted_indices=deleted,
                kept_indices=kept,
                scores=s.astype(float).tolist(),
                hook_module_name=meta.matched_bn if meta.matched_bn is not None else meta.name,
            )
            plans[meta.name] = plan
            rows.append({
                'layer_name': meta.name,
                'out_channels': meta.out_channels,
                'delete_count': delete_count,
                'keep_count': len(kept),
                'actual_prune_ratio': plan.prune_ratio,
                'score_source': self.score_source,
                'matched_bn': meta.matched_bn,
            })
        result = {
            'method': self.method,
            'score_source': self.score_source,
            'prune_ratio': float(prune_ratio),
            'plans': {k: v.to_dict() for k, v in plans.items()},
            'layer_rows': rows,
        }
        return result


class DepGraphGlobalPruner:
    """Reproducible generic DepGraph baseline with global L1 ranking.

    DependencyGraph is used by ``ChannelMaskApplier`` for physical group
    removal.  This planner intentionally excludes target-aware q statistics so
    it remains a clean generic structured-pruning baseline.
    """

    def __init__(self, model: nn.Module, bundle: SelectionBundle, cfg: Any) -> None:
        self.model = unwrap_model(model)
        self.bundle = bundle
        self.cfg = cfg

    def plan(self, prune_ratio: float) -> Dict[str, Any]:
        ratio = float(prune_ratio)
        scores = compute_layer_scores(self.model, self.bundle, 'l1')
        global_max_ratio = float(self.cfg.max_layer_prune_ratio)
        global_min_remaining = int(self.cfg.min_remaining_channels_per_layer)
        cap_rules = normalize_layer_prune_cap_rules(
            getattr(self.cfg, 'layer_prune_cap_rules', [])
        )
        min_remaining_rules = normalize_layer_min_remaining_rules(
            getattr(self.cfg, 'layer_min_remaining_rules', [])
        )
        candidates: List[Tuple[float, int, int]] = []
        caps: Dict[int, int] = {}
        cap_metadata: Dict[int, Dict[str, Any]] = {}
        floors: Dict[int, int] = {}
        normalized_scores: Dict[str, np.ndarray] = {}
        total_channels = 0
        for layer_idx, meta in enumerate(self.bundle.selected_layers):
            normalized = _minmax_normalize(scores[meta.name])
            normalized_scores[meta.name] = normalized
            cap_info = resolve_layer_prune_cap(
                meta.name,
                global_max_ratio,
                cap_rules,
            )
            min_info = resolve_layer_min_remaining(
                meta.name,
                global_min_remaining,
                min_remaining_rules,
            )
            cap = _safe_delete_cap(
                meta.out_channels,
                min_info['effective_min_remaining_channels'],
                cap_info['effective_max_prune_ratio'],
            )
            floor = min(
                cap,
                int(math.floor(meta.out_channels * float(self.cfg.min_layer_prune_ratio))),
            )
            caps[layer_idx] = cap
            cap_metadata[layer_idx] = {**cap_info, **min_info}
            floors[layer_idx] = floor
            total_channels += meta.out_channels
            for channel_idx, value in enumerate(normalized.tolist()):
                candidates.append((float(value), layer_idx, channel_idx))

        target_delete = min(
            int(round(ratio * total_channels)),
            int(sum(caps.values())),
        )
        selected = {idx: set() for idx in range(len(self.bundle.selected_layers))}
        for layer_idx, floor in floors.items():
            if floor <= 0:
                continue
            meta = self.bundle.selected_layers[layer_idx]
            order = np.argsort(normalized_scores[meta.name], kind='stable')[:floor]
            selected[layer_idx].update(order.tolist())

        selected_count = sum(len(values) for values in selected.values())
        for _, layer_idx, channel_idx in sorted(candidates, key=lambda item: item[0]):
            if selected_count >= target_delete:
                break
            if channel_idx in selected[layer_idx]:
                continue
            if len(selected[layer_idx]) >= caps[layer_idx]:
                continue
            selected[layer_idx].add(channel_idx)
            selected_count += 1

        plans: Dict[str, LayerPlan] = {}
        rows: List[Dict[str, Any]] = []
        for layer_idx, meta in enumerate(self.bundle.selected_layers):
            deleted = sorted(selected[layer_idx])
            deleted_set = set(deleted)
            kept = [idx for idx in range(meta.out_channels) if idx not in deleted_set]
            plan = LayerPlan(
                layer_name=meta.name,
                delete_count=len(deleted),
                keep_count=len(kept),
                prune_ratio=float(len(deleted) / max(meta.out_channels, 1)),
                deleted_indices=deleted,
                kept_indices=kept,
                scores=scores[meta.name].astype(float).tolist(),
                priority_values=normalized_scores[meta.name].astype(float).tolist(),
                max_delete_cap=caps[layer_idx],
                min_delete_floor=floors[layer_idx],
                hook_module_name=meta.matched_bn if meta.matched_bn is not None else meta.name,
                effective_max_prune_ratio=cap_metadata[layer_idx]['effective_max_prune_ratio'],
                matched_layer_prune_cap_rules=cap_metadata[layer_idx]['matched_layer_prune_cap_rules'],
                effective_layer_prune_cap_rule=cap_metadata[layer_idx]['effective_layer_prune_cap_rule'],
                effective_min_remaining_channels=cap_metadata[layer_idx]['effective_min_remaining_channels'],
                matched_layer_min_remaining_rules=cap_metadata[layer_idx]['matched_layer_min_remaining_rules'],
                effective_layer_min_remaining_rule=cap_metadata[layer_idx]['effective_layer_min_remaining_rule'],
            )
            plans[meta.name] = plan
            rows.append({
                'layer_name': meta.name,
                'candidate_count': meta.out_channels,
                'delete_count': len(deleted),
                'keep_count': len(kept),
                'actual_prune_ratio': plan.prune_ratio,
                'max_delete_cap': caps[layer_idx],
                'effective_max_prune_ratio': cap_metadata[layer_idx]['effective_max_prune_ratio'],
                'matched_layer_prune_cap_rules': cap_metadata[layer_idx]['matched_layer_prune_cap_rules'],
                'effective_layer_prune_cap_rule': cap_metadata[layer_idx]['effective_layer_prune_cap_rule'],
                'effective_min_remaining_channels': cap_metadata[layer_idx]['effective_min_remaining_channels'],
                'matched_layer_min_remaining_rules': cap_metadata[layer_idx]['matched_layer_min_remaining_rules'],
                'effective_layer_min_remaining_rule': cap_metadata[layer_idx]['effective_layer_min_remaining_rule'],
            })

        return {
            'method': 'depgraph',
            'score_source': 'l1_global',
            'prune_ratio': ratio,
            'global_candidate_channels': total_channels,
            'global_target_delete_after_cap': target_delete,
            'global_actual_delete': selected_count,
            'max_layer_prune_ratio': global_max_ratio,
            'layer_prune_cap_rules': cap_rules,
            'min_remaining_channels_per_layer': global_min_remaining,
            'layer_min_remaining_rules': min_remaining_rules,
            'plans': {name: plan.to_dict() for name, plan in plans.items()},
            'layer_rows': rows,
            'note': 'Generic global L1 ranking with DepGraph dependency-consistent physical pruning.',
        }


def _minmax_normalize(values: np.ndarray, neutral: float = 0.5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo <= 1e-8:
        return np.full_like(values, float(neutral), dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def _collect_conv_output_hw(model: nn.Module, layer_names: Sequence[str], patch_size: int) -> Dict[str, Tuple[int, int]]:
    """Collect output spatial sizes once for a cheap per-channel FLOPs proxy."""
    model = unwrap_model(model)
    wanted = set(layer_names)
    shapes: Dict[str, Tuple[int, int]] = {}
    handles: List[Any] = []
    for name, module in model.named_modules():
        if name not in wanted or not isinstance(module, nn.Conv2d):
            continue

        def hook_fn(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor, n: str = name) -> None:
            if torch.is_tensor(output) and output.ndim == 4:
                shapes[n] = (int(output.shape[-2]), int(output.shape[-1]))

        handles.append(module.register_forward_hook(hook_fn))

    first_conv = next((m for m in model.modules() if isinstance(m, nn.Conv2d)), None)
    in_channels = int(first_conv.in_channels) if first_conv is not None else 1
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')
    was_training = model.training
    model.eval()
    try:
        dummy = torch.zeros(1, in_channels, int(patch_size), int(patch_size), device=device)
        with torch.no_grad():
            _ = model(dummy)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return shapes


class OursGlobalBudgetPruner:
    """Target-aware global structured pruning planner.

    The recommended backend ranks every eligible channel in one global pool.
    DepGraph remains responsible for dependency-consistent physical removal.  A
    legacy adaptive per-layer allocator is retained as a compatibility fallback.
    """

    def __init__(self, model: nn.Module, bundle: SelectionBundle, cfg: Any, ocp_mapping: Dict[str, Any], variant: str = 'ours_slim') -> None:
        assert variant in {'ours', 'ours_slim', 'ours_l1'}
        self.model = unwrap_model(model)
        self.bundle = bundle
        self.cfg = cfg
        self.ocp_mapping = ocp_mapping
        self.variant = variant
        self.score_source = 'ours_l1' if variant == 'ours_l1' else 'ours_slim'

    def _plan_cost_aware_global(self, global_prune_ratio: float) -> Dict[str, Any]:
        ratio = float(global_prune_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f'global_prune_ratio must be in [0, 1], got {ratio}')

        scores = compute_layer_scores(self.model, self.bundle, self.score_source)
        spatial_shapes = _collect_conv_output_hw(
            self.model,
            self.bundle.selected_names(),
            int(getattr(self.cfg, 'patch_size', 256)),
        )
        beta = float(np.clip(getattr(self.cfg, 'target_aware_beta', 0.5), 0.0, 1.0))
        alpha = float(np.clip(getattr(self.cfg, 'structural_gain_alpha', 0.5), 0.0, 1.0))
        global_max_ratio = float(self.cfg.max_layer_prune_ratio)
        global_min_remaining = int(self.cfg.min_remaining_channels_per_layer)
        cap_rules = normalize_layer_prune_cap_rules(
            getattr(self.cfg, 'layer_prune_cap_rules', [])
        )
        min_remaining_rules = normalize_layer_min_remaining_rules(
            getattr(self.cfg, 'layer_min_remaining_rules', [])
        )

        layer_items: List[Dict[str, Any]] = []
        all_param_gain: List[float] = []
        all_flops_gain: List[float] = []
        total_channels = 0
        for meta in self.bundle.selected_layers:
            conv = get_module_by_name(self.model, meta.name)
            if not isinstance(conv, nn.Conv2d):
                raise TypeError(f'Expected Conv2d for {meta.name}, got {type(conv)}')
            raw_s = scores[meta.name].astype(np.float32)
            s = _minmax_normalize(raw_s)
            q = np.asarray(
                self.ocp_mapping['mappings'][meta.name]['projected_q_values'],
                dtype=np.float32,
            )
            if len(q) != meta.out_channels:
                raise RuntimeError(
                    f'q length mismatch for {meta.name}: len(q)={len(q)} '
                    f'out_channels={meta.out_channels}'
                )
            q = np.clip(q, 0.0, 1.0)
            removal_cost = (1.0 - beta) * s + beta * q

            kh, kw = (int(x) for x in conv.kernel_size)
            direct_params = float((conv.in_channels // conv.groups) * kh * kw)
            if conv.bias is not None:
                direct_params += 1.0
            if meta.matched_bn is not None:
                direct_params += 2.0
            oh, ow = spatial_shapes.get(meta.name, (1, 1))
            direct_flops = float(oh * ow * (conv.in_channels // conv.groups) * kh * kw)
            param_gain = np.full(meta.out_channels, direct_params, dtype=np.float32)
            flops_gain = np.full(meta.out_channels, direct_flops, dtype=np.float32)
            all_param_gain.extend(param_gain.tolist())
            all_flops_gain.extend(flops_gain.tolist())

            cap_info = resolve_layer_prune_cap(
                meta.name,
                global_max_ratio,
                cap_rules,
            )
            min_info = resolve_layer_min_remaining(
                meta.name,
                global_min_remaining,
                min_remaining_rules,
            )
            max_delete_cap = _safe_delete_cap(
                meta.out_channels,
                min_info['effective_min_remaining_channels'],
                cap_info['effective_max_prune_ratio'],
            )
            min_delete_floor = min(
                max_delete_cap,
                int(math.floor(meta.out_channels * float(self.cfg.min_layer_prune_ratio))),
            )
            total_channels += meta.out_channels
            layer_items.append({
                'meta': meta,
                'raw_s': raw_s,
                's': s,
                'q': q,
                'u': removal_cost,
                'param_gain': param_gain,
                'flops_gain': flops_gain,
                'max_delete_cap': int(max_delete_cap),
                'min_delete_floor': int(min_delete_floor),
                'cap_info': {**cap_info, **min_info},
            })

        max_param_gain = max(max(all_param_gain, default=1.0), 1e-12)
        max_flops_gain = max(max(all_flops_gain, default=1.0), 1e-12)
        all_candidates: List[Tuple[float, int, int, float]] = []
        removable_candidates: List[Tuple[float, int, int, float]] = []
        for layer_idx, item in enumerate(layer_items):
            p_norm = item['param_gain'] / max_param_gain
            f_norm = item['flops_gain'] / max_flops_gain
            gain = alpha * p_norm + (1.0 - alpha) * f_norm
            priority = item['u'] / (gain + 1e-8)
            item['gain'] = gain.astype(np.float32)
            item['priority'] = priority.astype(np.float32)
            order = np.argsort(priority, kind='stable')
            removable = set(order[:item['max_delete_cap']].tolist())
            for channel_idx in range(int(item['meta'].out_channels)):
                candidate = (
                    float(priority[channel_idx]),
                    layer_idx,
                    channel_idx,
                    float(gain[channel_idx]),
                )
                all_candidates.append(candidate)
                if channel_idx in removable:
                    removable_candidates.append(candidate)

        total_structural_gain = float(sum(x[3] for x in all_candidates))
        removable_structural_gain = float(sum(x[3] for x in removable_candidates))
        requested_target_gain = ratio * total_structural_gain
        target_gain = min(requested_target_gain, removable_structural_gain)

        selected_by_layer = {idx: set() for idx in range(len(layer_items))}
        selected_gain = 0.0
        # Honour an optional minimum layer floor before filling the shared pool.
        for layer_idx, item in enumerate(layer_items):
            floor = int(item['min_delete_floor'])
            if floor <= 0:
                continue
            order = np.argsort(item['priority'], kind='stable')[:floor]
            for channel_idx in order.tolist():
                if channel_idx not in selected_by_layer[layer_idx]:
                    selected_by_layer[layer_idx].add(channel_idx)
                    selected_gain += float(item['gain'][channel_idx])

        for priority, layer_idx, channel_idx, gain in sorted(removable_candidates, key=lambda x: x[0]):
            if selected_gain >= target_gain:
                break
            if channel_idx in selected_by_layer[layer_idx]:
                continue
            selected_by_layer[layer_idx].add(channel_idx)
            selected_gain += gain

        plans: Dict[str, LayerPlan] = {}
        layer_rows: List[Dict[str, Any]] = []
        total_selected_gain = max(selected_gain, 1e-12)
        for layer_idx, item in enumerate(layer_items):
            meta: LayerMeta = item['meta']
            deleted = sorted(selected_by_layer[layer_idx])
            deleted_set = set(deleted)
            kept = [idx for idx in range(meta.out_channels) if idx not in deleted_set]
            delete_count = len(deleted)
            layer_gain = float(sum(float(item['gain'][idx]) for idx in deleted))
            retention = _compute_high_q_retention(item['q'], kept)
            plan = LayerPlan(
                layer_name=meta.name,
                delete_count=delete_count,
                keep_count=len(kept),
                prune_ratio=float(delete_count / max(meta.out_channels, 1)),
                deleted_indices=deleted,
                kept_indices=kept,
                scores=item['raw_s'].astype(float).tolist(),
                q_values=item['q'].astype(float).tolist(),
                u_values=item['u'].astype(float).tolist(),
                gain_values=item['gain'].astype(float).tolist(),
                priority_values=item['priority'].astype(float).tolist(),
                layer_budget_weight=float(layer_gain / total_selected_gain),
                budget_before_correction=None,
                budget_after_correction=delete_count,
                max_delete_cap=int(item['max_delete_cap']),
                min_delete_floor=int(item['min_delete_floor']),
                high_q_retention=float(retention),
                hook_module_name=meta.matched_bn if meta.matched_bn is not None else meta.name,
                effective_max_prune_ratio=item['cap_info']['effective_max_prune_ratio'],
                matched_layer_prune_cap_rules=item['cap_info']['matched_layer_prune_cap_rules'],
                effective_layer_prune_cap_rule=item['cap_info']['effective_layer_prune_cap_rule'],
                effective_min_remaining_channels=item['cap_info']['effective_min_remaining_channels'],
                matched_layer_min_remaining_rules=item['cap_info']['matched_layer_min_remaining_rules'],
                effective_layer_min_remaining_rule=item['cap_info']['effective_layer_min_remaining_rule'],
            )
            plans[meta.name] = plan
            layer_rows.append({
                'layer_name': meta.name,
                'candidate_count': int(meta.out_channels),
                'delete_count': delete_count,
                'keep_count': len(kept),
                'actual_prune_ratio': plan.prune_ratio,
                'selected_structural_gain': layer_gain,
                'selected_gain_share': float(layer_gain / total_selected_gain),
                'max_delete_cap': int(item['max_delete_cap']),
                'min_delete_floor': int(item['min_delete_floor']),
                'effective_max_prune_ratio': item['cap_info']['effective_max_prune_ratio'],
                'matched_layer_prune_cap_rules': item['cap_info']['matched_layer_prune_cap_rules'],
                'effective_layer_prune_cap_rule': item['cap_info']['effective_layer_prune_cap_rule'],
                'effective_min_remaining_channels': item['cap_info']['effective_min_remaining_channels'],
                'matched_layer_min_remaining_rules': item['cap_info']['matched_layer_min_remaining_rules'],
                'effective_layer_min_remaining_rule': item['cap_info']['effective_layer_min_remaining_rule'],
                'high_q_retention': float(retention),
                'mean_q': float(np.mean(item['q'])),
                'mean_importance_normalized': float(np.mean(item['s'])),
                'mean_removal_cost': float(np.mean(item['u'])),
                'mean_structural_gain': float(np.mean(item['gain'])),
                'mapped_ocp_layer': self.ocp_mapping['mappings'][meta.name].get('mapped_ocp_layer'),
                'mapping_rule': self.ocp_mapping['mappings'][meta.name].get('rule'),
            })

        actual_delete = int(sum(plan.delete_count for plan in plans.values()))
        return {
            'method': 'ours',
            'variant': self.variant,
            'planner': 'cost_aware_global',
            'score_source': self.score_source,
            'global_prune_ratio': ratio,
            'target_aware_beta': beta,
            'structural_gain_alpha': alpha,
            'global_candidate_channels': int(total_channels),
            'global_target_delete_before_cap': int(round(ratio * total_channels)),
            'global_target_delete_after_cap': actual_delete,
            'global_actual_delete': actual_delete,
            'max_layer_prune_ratio': global_max_ratio,
            'layer_prune_cap_rules': cap_rules,
            'min_remaining_channels_per_layer': global_min_remaining,
            'layer_min_remaining_rules': min_remaining_rules,
            'total_structural_gain': total_structural_gain,
            'removable_structural_gain': removable_structural_gain,
            'requested_target_structural_gain': requested_target_gain,
            'target_structural_gain_after_cap': target_gain,
            'actual_selected_structural_gain': float(selected_gain),
            'actual_structural_gain_ratio': float(selected_gain / max(total_structural_gain, 1e-12)),
            'plans': {k: v.to_dict() for k, v in plans.items()},
            'layer_rows': layer_rows,
            'note': (
                'Single global candidate pool with target-aware removal cost and '
                'parameter/FLOPs gain; DepGraph performs physical dependency-consistent pruning.'
            ),
        }

    def plan(self, global_prune_ratio: float) -> Dict[str, Any]:
        if str(getattr(self.cfg, 'global_planner', 'cost_aware_global')) == 'cost_aware_global':
            return self._plan_cost_aware_global(global_prune_ratio)

        scores = compute_layer_scores(self.model, self.bundle, self.score_source)
        global_max_ratio = float(self.cfg.max_layer_prune_ratio)
        global_min_remaining = int(self.cfg.min_remaining_channels_per_layer)
        cap_rules = normalize_layer_prune_cap_rules(
            getattr(self.cfg, 'layer_prune_cap_rules', [])
        )
        min_remaining_rules = normalize_layer_min_remaining_rules(
            getattr(self.cfg, 'layer_min_remaining_rules', [])
        )
        layer_items: List[Dict[str, Any]] = []
        total_channels = 0
        for meta in self.bundle.selected_layers:
            s = scores[meta.name].astype(np.float32)
            q = np.array(self.ocp_mapping['mappings'][meta.name]['projected_q_values'], dtype=np.float32)
            if len(q) != meta.out_channels:
                raise RuntimeError(f'q length mismatch for {meta.name}: len(q)={len(q)} out_channels={meta.out_channels}')
            u = s * q
            cap_info = resolve_layer_prune_cap(
                meta.name,
                global_max_ratio,
                cap_rules,
            )
            min_info = resolve_layer_min_remaining(
                meta.name,
                global_min_remaining,
                min_remaining_rules,
            )
            max_delete_cap = _safe_delete_cap(
                meta.out_channels,
                min_info['effective_min_remaining_channels'],
                cap_info['effective_max_prune_ratio'],
            )
            min_delete_floor = min(max_delete_cap, int(math.floor(meta.out_channels * float(self.cfg.min_layer_prune_ratio))))
            prunability = _layer_prunability(q, s, u, str(self.cfg.layer_prunability_metric))
            total_channels += meta.out_channels
            layer_items.append({
                'meta': meta,
                's': s,
                'q': q,
                'u': u,
                'candidate_count': int(meta.out_channels),
                'max_delete_cap': int(max_delete_cap),
                'min_delete_floor': int(min_delete_floor),
                'prunability': float(prunability),
                'cap_info': {**cap_info, **min_info},
            })

        target_delete = int(round(float(global_prune_ratio) * total_channels))
        floor_sum = sum(item['min_delete_floor'] for item in layer_items)
        target_delete = max(target_delete, floor_sum)
        pre_cap_target = target_delete
        cap_sum = sum(item['max_delete_cap'] for item in layer_items)
        if target_delete > cap_sum:
            target_delete = cap_sum
        remaining_budget = max(0, target_delete - floor_sum)

        smoothing = float(self.cfg.budget_smoothing)
        raw_weights = np.array([max(item['prunability'], 0.0) + smoothing for item in layer_items], dtype=np.float64)
        if raw_weights.sum() <= 0:
            raw_weights[:] = 1.0
        weight_sum = float(raw_weights.sum())
        desired_extra = [remaining_budget * float(w / weight_sum) for w in raw_weights.tolist()]
        rounded_extra = _round_to_int(desired_extra, str(self.cfg.budget_rounding_mode))

        budgets_before = []
        budgets_after = []
        residual_slots: List[Tuple[float, int, int]] = []
        allocated = 0
        for idx, item in enumerate(layer_items):
            budget_before = item['min_delete_floor'] + rounded_extra[idx]
            budget_after = min(max(item['min_delete_floor'], budget_before), item['max_delete_cap'])
            budgets_before.append(float(budget_before))
            budgets_after.append(int(budget_after))
            allocated += int(budget_after)
            residual_capacity = item['max_delete_cap'] - int(budget_after)
            if residual_capacity > 0:
                # Use best remaining low-u channels later for residual correction.
                remaining_sorted = _sort_indices_by_score(item['u'])[int(budget_after):]
                for local_rank, ch_idx in enumerate(remaining_sorted.tolist()):
                    residual_slots.append((float(item['u'][ch_idx]), idx, ch_idx))

        # Correction to ensure exact global target after caps/rounding.
        if allocated < target_delete:
            residual_slots = sorted(residual_slots, key=lambda x: x[0])
            needed = target_delete - allocated
            used_per_layer = {idx: 0 for idx in range(len(layer_items))}
            for _, layer_idx, _ch_idx in residual_slots:
                if needed <= 0:
                    break
                if budgets_after[layer_idx] + used_per_layer[layer_idx] >= layer_items[layer_idx]['max_delete_cap']:
                    continue
                used_per_layer[layer_idx] += 1
                needed -= 1
            for layer_idx, extra in used_per_layer.items():
                budgets_after[layer_idx] += extra
        elif allocated > target_delete:
            # Remove over-allocation from highest-u among currently selected deletions.
            removable: List[Tuple[float, int]] = []
            for idx, item in enumerate(layer_items):
                if budgets_after[idx] <= item['min_delete_floor']:
                    continue
                selected = _sort_indices_by_score(item['u'])[:budgets_after[idx]]
                if len(selected) == 0:
                    continue
                worst_selected_u = float(item['u'][selected[-1]])
                removable.append((worst_selected_u, idx))
            removable = sorted(removable, reverse=True)
            extra_to_remove = allocated - target_delete
            ptr = 0
            while extra_to_remove > 0 and removable:
                _, idx = removable[ptr % len(removable)]
                if budgets_after[idx] > layer_items[idx]['min_delete_floor']:
                    budgets_after[idx] -= 1
                    extra_to_remove -= 1
                ptr += 1

        plans: Dict[str, LayerPlan] = {}
        layer_rows: List[Dict[str, Any]] = []
        for idx, item in enumerate(layer_items):
            meta: LayerMeta = item['meta']
            delete_count = int(budgets_after[idx])
            sorted_idx = _sort_indices_by_score(item['u'])
            deleted = sorted_idx[:delete_count].tolist()
            kept = sorted(sorted_idx[delete_count:].tolist())
            retention = _compute_high_q_retention(item['q'], kept)
            plan = LayerPlan(
                layer_name=meta.name,
                delete_count=delete_count,
                keep_count=len(kept),
                prune_ratio=float(delete_count / max(meta.out_channels, 1)),
                deleted_indices=deleted,
                kept_indices=kept,
                scores=item['s'].astype(float).tolist(),
                q_values=item['q'].astype(float).tolist(),
                u_values=item['u'].astype(float).tolist(),
                layer_budget_weight=float(raw_weights[idx] / weight_sum),
                budget_before_correction=float(budgets_before[idx]),
                budget_after_correction=int(delete_count),
                max_delete_cap=int(item['max_delete_cap']),
                min_delete_floor=int(item['min_delete_floor']),
                high_q_retention=float(retention),
                hook_module_name=meta.matched_bn if meta.matched_bn is not None else meta.name,
                effective_max_prune_ratio=item['cap_info']['effective_max_prune_ratio'],
                matched_layer_prune_cap_rules=item['cap_info']['matched_layer_prune_cap_rules'],
                effective_layer_prune_cap_rule=item['cap_info']['effective_layer_prune_cap_rule'],
                effective_min_remaining_channels=item['cap_info']['effective_min_remaining_channels'],
                matched_layer_min_remaining_rules=item['cap_info']['matched_layer_min_remaining_rules'],
                effective_layer_min_remaining_rule=item['cap_info']['effective_layer_min_remaining_rule'],
            )
            plans[meta.name] = plan
            layer_rows.append({
                'layer_name': meta.name,
                'candidate_count': int(item['candidate_count']),
                'global_target_delete': int(target_delete),
                'delete_count': int(delete_count),
                'keep_count': len(kept),
                'actual_prune_ratio': plan.prune_ratio,
                'prunability_weight': float(raw_weights[idx] / weight_sum),
                'budget_before_correction': float(budgets_before[idx]),
                'budget_after_correction': int(delete_count),
                'max_delete_cap': int(item['max_delete_cap']),
                'min_delete_floor': int(item['min_delete_floor']),
                'effective_max_prune_ratio': item['cap_info']['effective_max_prune_ratio'],
                'matched_layer_prune_cap_rules': item['cap_info']['matched_layer_prune_cap_rules'],
                'effective_layer_prune_cap_rule': item['cap_info']['effective_layer_prune_cap_rule'],
                'effective_min_remaining_channels': item['cap_info']['effective_min_remaining_channels'],
                'matched_layer_min_remaining_rules': item['cap_info']['matched_layer_min_remaining_rules'],
                'effective_layer_min_remaining_rule': item['cap_info']['effective_layer_min_remaining_rule'],
                'high_q_retention': float(retention),
                'mean_q': float(np.mean(item['q'])),
                'mean_s': float(np.mean(item['s'])),
                'mean_u': float(np.mean(item['u'])),
                'mapped_ocp_layer': self.ocp_mapping['mappings'][meta.name].get('mapped_ocp_layer'),
                'mapping_rule': self.ocp_mapping['mappings'][meta.name].get('rule'),
            })

        result = {
            'method': 'ours',
            'variant': self.variant,
            'score_source': self.score_source,
            'global_prune_ratio': float(global_prune_ratio),
            'global_candidate_channels': int(total_channels),
            'global_target_delete_before_cap': int(pre_cap_target),
            'global_target_delete_after_cap': int(target_delete),
            'global_actual_delete': int(sum(x.delete_count for x in plans.values())),
            'adaptive_budget_strategy': str(self.cfg.adaptive_budget_strategy),
            'max_layer_prune_ratio': global_max_ratio,
            'layer_prune_cap_rules': cap_rules,
            'min_remaining_channels_per_layer': global_min_remaining,
            'layer_min_remaining_rules': min_remaining_rules,
            'plans': {k: v.to_dict() for k, v in plans.items()},
            'layer_rows': layer_rows,
            'note': 'ours uses global budget + adaptive inter-layer budget allocation; l1/slim do not.',
        }
        return result



def export_pruning_result(plan_result: Dict[str, Any], out_dir: str, stem: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(plan_result, str(out / f'{stem}.json'))
    rows = plan_result.get('layer_rows', [])
    if rows:
        save_simple_csv(rows, str(out / f'{stem}.csv'))



def masked_model_effective_stats(model: nn.Module, mask_applier: ChannelMaskApplier, input_size: Tuple[int, int, int, int], device: torch.device) -> Dict[str, Any]:
    """Measure compact-model params/FLOPs after physical pruning.

    The previous mask backend reported approximate compact-model cost while keeping the dense graph.
    After the DepGraph upgrade the model has already been structurally pruned, so these numbers describe
    the current compact model itself (subject only to the usual profiling convention for FLOPs).
    """
    model = unwrap_model(model)
    hook_shapes: Dict[str, Tuple[int, ...]] = {}
    handles = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            def hook_fn(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor, n: str = name) -> None:
                if torch.is_tensor(output):
                    hook_shapes[n] = tuple(int(x) for x in output.shape)
            handles.append(module.register_forward_hook(hook_fn))

    dummy = torch.randn(*input_size, device=device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        _ = model(dummy)
    for handle in handles:
        handle.remove()

    total_params = 0
    total_flops = 0.0
    layer_rows: List[Dict[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            kh, kw = module.kernel_size
            out_c = int(module.out_channels)
            in_c = int(module.in_channels)
            params = out_c * (in_c // module.groups) * kh * kw + (out_c if module.bias is not None else 0)
            total_params += params
            shape = hook_shapes.get(name)
            flops = 0.0
            if shape is not None and len(shape) == 4:
                _, _, oh, ow = shape
                flops = float(oh * ow * out_c * (in_c // module.groups) * kh * kw)
                total_flops += flops
            layer_rows.append({
                'layer_name': name,
                'out_channels': out_c,
                'in_channels': in_c,
                'dense_params': int(params),
                'effective_params': int(params),
                'dense_flops': flops,
                'effective_flops': flops,
            })
        elif isinstance(module, nn.BatchNorm2d):
            c = int(module.num_features)
            total_params += 2 * c

    return {
        'dense_params_approx': int(total_params),
        'effective_params_approx': int(total_params),
        'dense_flops_approx': float(total_flops),
        'effective_flops_approx': float(total_flops),
        'note': 'Physical compact-model stats after DepGraph pruning.',
        'layer_rows': layer_rows,
    }


def clone_model_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return copy.deepcopy(unwrap_model(model).state_dict())
