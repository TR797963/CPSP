import json
import re
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from core.modeling import get_module_by_name, named_modules_dict
from core.utils import list_to_pretty_lines, save_json, save_text


ROLE_KEYWORDS = {
    'stem': ['stem', 'input', 'conv0_0', 'conv_in'],
    'encoder': ['encoder', 'backbone', 'layer', 'down', 'enc', 'conv1_', 'conv2_', 'conv3_', 'conv4_'],
    'decoder': ['decoder', 'up', 'dec', 'conv0_', 'conv1_', 'conv2_'],
    'fuse': ['fuse', 'merge', 'concat', 'fusion', 'final', 'conv0_4_final'],
    'head': ['head', 'final', 'cls', 'pred', 'out', 'mask', 'segmentation', 'classifier'],
}


@dataclass
class LayerMeta:
    name: str
    type: str
    depth_index: int
    parent_name: str
    role: str
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int]
    groups: int
    has_bn: bool
    matched_bn: Optional[str]
    selected: bool
    reason: str
    stage_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelectionBundle:
    all_layers: List[LayerMeta]
    selected_layers: List[LayerMeta]
    skipped_layers: List[LayerMeta]

    def selected_names(self) -> List[str]:
        return [x.name for x in self.selected_layers]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'selected': [x.to_dict() for x in self.selected_layers],
            'skipped': [x.to_dict() for x in self.skipped_layers],
            'all': [x.to_dict() for x in self.all_layers],
        }


def _split_parent_child(module_name: str) -> Tuple[str, str]:
    if '.' not in module_name:
        return '', module_name
    parent, child = module_name.rsplit('.', 1)
    return parent, child


_STAGE_PATTERNS = [
    re.compile(r'conv(\d+)_(\d+)'),
    re.compile(r'layer(\d+)'),
    re.compile(r'encoder(\d+)'),
    re.compile(r'decoder(\d+)'),
    re.compile(r'up(\d+)'),
    re.compile(r'down(\d+)'),
]


def infer_stage_id(module_name: str) -> str:
    lower = module_name.lower()
    for pat in _STAGE_PATTERNS:
        m = pat.search(lower)
        if m:
            return '_'.join(m.groups())
    digits = re.findall(r'\d+', lower)
    if digits:
        return digits[0]
    return 'na'



def infer_role(module_name: str, index: int, total_conv_layers: int) -> str:
    lower = module_name.lower()
    if index == 0:
        return 'stem'
    if index >= total_conv_layers - 2:
        return 'head'
    for role, words in ROLE_KEYWORDS.items():
        if any(word in lower for word in words):
            return role
    return 'encoder' if index < total_conv_layers // 2 else 'decoder'



def _collect_bn_candidates(model: nn.Module) -> Dict[str, List[Tuple[str, nn.BatchNorm2d]]]:
    parent_to_bn: Dict[str, List[Tuple[str, nn.BatchNorm2d]]] = defaultdict(list)
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            parent, _ = _split_parent_child(name)
            parent_to_bn[parent].append((name, module))
    return parent_to_bn



def match_bn_for_conv(model: nn.Module) -> Dict[str, Optional[str]]:
    """Heuristic Conv->BN matcher.

    Engineering approximation:
    - Prefer sibling bn with the same suffix number, e.g. conv2 -> bn2.
    - Otherwise choose the first sibling BatchNorm2d with matching num_features.
    - This is explicit and logged; layers without reliable BN match are skipped for slim/ours_slim.
    """
    parent_to_bn = _collect_bn_candidates(model)
    mapping: Dict[str, Optional[str]] = {}
    for conv_name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        parent, child = _split_parent_child(conv_name)
        candidates = parent_to_bn.get(parent, [])
        matched = None
        suffix_digits = ''.join(re.findall(r'\d+', child))
        if suffix_digits:
            for bn_name, bn in candidates:
                if bn_name.endswith(suffix_digits) and bn.num_features == module.out_channels:
                    matched = bn_name
                    break
        if matched is None:
            for bn_name, bn in candidates:
                if bn.num_features == module.out_channels:
                    matched = bn_name
                    break
        mapping[conv_name] = matched
    return mapping



def _module_children_names(model: nn.Module, parent_name: str) -> List[str]:
    parent = get_module_by_name(model, parent_name) if parent_name else model
    return [name for name, _ in parent.named_children()]



def _exclude_reason_for_conv(
    module_name: str,
    module: nn.Conv2d,
    depth_index: int,
    total_conv_layers: int,
    cfg: Any,
    role: str,
    matched_bn: Optional[str],
    require_bn: bool,
) -> str:
    lower = module_name.lower()
    if 'shortcut' in lower or 'downsample' in lower or 'proj' in lower:
        return 'shape_sensitive_shortcut_like'
    if cfg.exclude_first_conv and depth_index == 0:
        return 'excluded_first_conv'
    if cfg.exclude_last_head and role == 'head':
        return 'excluded_head'
    if module.out_channels < int(cfg.min_prunable_channels):
        return f'out_channels<{cfg.min_prunable_channels}'
    if module.groups != 1 and not bool(cfg.allow_group_conv_pruning):
        return f'groups={module.groups}_not_allowed'
    if module.groups == module.in_channels == module.out_channels and not bool(cfg.allow_depthwise_pruning):
        return 'depthwise_not_allowed'
    if require_bn and matched_bn is None:
        return 'bn_not_matched'
    if module.kernel_size == (1, 1) and role == 'head' and cfg.exclude_last_head:
        return '1x1_head_excluded'
    # Conservative skip for very late final predictor convs.
    lower = module_name.lower()
    if cfg.exclude_last_head and any(k in lower for k in ['final', 'pred', 'cls', 'mask', 'out']) and module.out_channels <= max(4, int(cfg.min_prunable_channels)):
        return 'predictor_like_layer'
    return ''



def _select_ocp_subset(candidates: List[LayerMeta], cfg: Any) -> List[LayerMeta]:
    strategy = str(cfg.ocp_layer_selection_strategy)
    max_layers = int(cfg.max_ocp_layers)
    if strategy == 'all_safe_layers':
        return candidates[:max_layers] if max_layers > 0 else candidates

    if strategy == 'topk_safe_layers':
        ranked = sorted(candidates, key=lambda x: (x.out_channels, -x.depth_index), reverse=True)
        return ranked[:max_layers]

    # per_stage_representative: one or two representative layers per stage, preferring encoder/fuse layers with BN.
    stage_to_layers: 'OrderedDict[str, List[LayerMeta]]' = OrderedDict()
    for meta in candidates:
        stage_to_layers.setdefault(meta.stage_id, []).append(meta)

    chosen: List[LayerMeta] = []
    for stage_id, metas in stage_to_layers.items():
        metas = sorted(metas, key=lambda x: (0 if x.role in {'encoder', 'fuse', 'decoder'} else 1, -x.out_channels, x.depth_index))
        chosen.append(metas[0])
    if max_layers > 0 and len(chosen) > max_layers:
        chosen = sorted(chosen, key=lambda x: (x.depth_index, -x.out_channels))[:max_layers]
    return chosen



def _iterate_conv_meta(model: nn.Module) -> List[Tuple[str, nn.Conv2d]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]



def auto_select_prunable_layers(model: nn.Module, cfg: Any, method: str = 'ours') -> SelectionBundle:
    convs = _iterate_conv_meta(model)
    bn_map = match_bn_for_conv(model)
    require_bn = method in {'slim', 'ours', 'ours_slim'} and bool(cfg.slim_bn_only)
    all_layers: List[LayerMeta] = []
    selected: List[LayerMeta] = []
    skipped: List[LayerMeta] = []

    total_conv_layers = len(convs)
    for idx, (name, module) in enumerate(convs):
        role = infer_role(name, idx, total_conv_layers)
        parent_name, _ = _split_parent_child(name)
        matched_bn = bn_map.get(name)
        reason = _exclude_reason_for_conv(name, module, idx, total_conv_layers, cfg, role, matched_bn, require_bn)
        meta = LayerMeta(
            name=name,
            type=module.__class__.__name__,
            depth_index=idx,
            parent_name=parent_name,
            role=role,
            in_channels=int(module.in_channels),
            out_channels=int(module.out_channels),
            kernel_size=tuple(int(x) for x in module.kernel_size),
            stride=tuple(int(x) for x in module.stride),
            groups=int(module.groups),
            has_bn=matched_bn is not None,
            matched_bn=matched_bn,
            selected=(reason == ''),
            reason=reason if reason else 'selected',
            stage_id=infer_stage_id(name),
        )
        all_layers.append(meta)
        if meta.selected:
            selected.append(meta)
        else:
            skipped.append(meta)

    if cfg.pruning_layers:
        manual = set(cfg.pruning_layers)
        for meta in all_layers:
            if meta.name in manual:
                meta.selected = True
                meta.reason = 'manual_override_selected'
            else:
                meta.selected = False
                meta.reason = 'manual_override_not_selected'
        selected = [x for x in all_layers if x.selected]
        skipped = [x for x in all_layers if not x.selected]

    return SelectionBundle(all_layers=all_layers, selected_layers=selected, skipped_layers=skipped)


def filter_selection_bundle_by_forward(
    model: nn.Module,
    bundle: SelectionBundle,
    *,
    input_size: Tuple[int, int, int, int],
    device: torch.device,
) -> SelectionBundle:
    """Exclude selected modules that are not executed by the real forward graph.

    Some upstream backbones retain constructor modules that are never used in
    ``forward``. Module enumeration alone cannot distinguish those dead
    branches, while DependencyGraph correctly omits them. Filtering before
    planning keeps the global budget and the physical graph consistent instead
    of silently dropping an invalid pruning request later.
    """

    executed = set()
    handles = []
    for meta in bundle.selected_layers:
        module = get_module_by_name(model, meta.name)

        def _record(_module: nn.Module, _inputs: Any, _output: Any, name: str = meta.name) -> None:
            executed.add(name)

        handles.append(module.register_forward_hook(_record))

    was_training = model.training
    try:
        model.to(device)
        model.eval()
        example = torch.zeros(*input_size, device=device)
        with torch.inference_mode():
            model(example)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    for meta in bundle.all_layers:
        if meta.selected and meta.name not in executed:
            meta.selected = False
            meta.reason = 'not_in_forward_graph'
    selected = [meta for meta in bundle.all_layers if meta.selected]
    skipped = [meta for meta in bundle.all_layers if not meta.selected]
    return SelectionBundle(
        all_layers=bundle.all_layers,
        selected_layers=selected,
        skipped_layers=skipped,
    )



def auto_select_ocp_layers(model: nn.Module, cfg: Any) -> SelectionBundle:
    convs = _iterate_conv_meta(model)
    bn_map = match_bn_for_conv(model)
    all_layers: List[LayerMeta] = []
    selected_candidates: List[LayerMeta] = []
    skipped: List[LayerMeta] = []
    total_conv_layers = len(convs)

    for idx, (name, module) in enumerate(convs):
        role = infer_role(name, idx, total_conv_layers)
        parent_name, _ = _split_parent_child(name)
        matched_bn = bn_map.get(name)
        reason = ''
        if cfg.exclude_first_conv and idx == 0:
            reason = 'excluded_first_conv'
        elif role == 'head' and cfg.exclude_last_head:
            reason = 'excluded_head'
        elif module.out_channels < max(8, int(cfg.min_prunable_channels)):
            reason = 'too_few_channels'
        elif min(module.kernel_size) <= 1 and role == 'head':
            reason = 'head_or_predictor'
        lower = name.lower()
        if 'shortcut' in lower or 'downsample' in lower or 'proj' in lower:
            reason = 'auxiliary_or_shortcut_like'
        if any(k in lower for k in ['aux', 'shortcut']) and role == 'head':
            reason = 'auxiliary_or_shortcut_like'
        meta = LayerMeta(
            name=name,
            type=module.__class__.__name__,
            depth_index=idx,
            parent_name=parent_name,
            role=role,
            in_channels=int(module.in_channels),
            out_channels=int(module.out_channels),
            kernel_size=tuple(int(x) for x in module.kernel_size),
            stride=tuple(int(x) for x in module.stride),
            groups=int(module.groups),
            has_bn=matched_bn is not None,
            matched_bn=matched_bn,
            selected=(reason == ''),
            reason=reason if reason else 'candidate',
            stage_id=infer_stage_id(name),
        )
        all_layers.append(meta)
        if meta.selected:
            selected_candidates.append(meta)
        else:
            skipped.append(meta)

    if cfg.ocp_layers:
        manual = set(cfg.ocp_layers)
        for meta in all_layers:
            if meta.name in manual:
                meta.selected = True
                meta.reason = 'manual_override_selected'
            else:
                meta.selected = False
                meta.reason = 'manual_override_not_selected'
        selected = [x for x in all_layers if x.selected]
        skipped = [x for x in all_layers if not x.selected]
        return SelectionBundle(all_layers=all_layers, selected_layers=selected, skipped_layers=skipped)

    chosen = _select_ocp_subset(selected_candidates, cfg)
    chosen_names = {x.name for x in chosen}
    final_selected: List[LayerMeta] = []
    final_skipped: List[LayerMeta] = []
    for meta in all_layers:
        if meta.name in chosen_names:
            meta.selected = True
            meta.reason = 'selected_by_strategy'
            final_selected.append(meta)
        else:
            if meta.reason == 'candidate':
                meta.reason = 'candidate_but_not_selected_by_strategy'
            meta.selected = False
            final_skipped.append(meta)
    return SelectionBundle(all_layers=all_layers, selected_layers=final_selected, skipped_layers=final_skipped)



def save_selection_bundle(bundle: SelectionBundle, out_dir: str, stem: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(bundle.to_dict(), str(out / f'{stem}.json'))
    save_text(list_to_pretty_lines([x.to_dict() for x in bundle.all_layers], title=stem), str(out / f'{stem}.txt'))



def build_selection_summary(selected: SelectionBundle) -> Dict[str, Any]:
    return {
        'num_all_layers': len(selected.all_layers),
        'num_selected_layers': len(selected.selected_layers),
        'num_skipped_layers': len(selected.skipped_layers),
        'selected_names': [x.name for x in selected.selected_layers],
        'skipped_with_reason': [{x.name: x.reason} for x in selected.skipped_layers],
    }



def pruning_dry_run_check(model: nn.Module, bundle: SelectionBundle, method: str) -> Dict[str, Any]:
    """Dry-run safety check before applying pruning.

    We verify that:
    - the module still exists,
    - channel count is compatible,
    - slim/ours_slim layers have BN mapping,
    - no selected layer has ambiguous zero channels.
    """
    modules = named_modules_dict(model)
    checked_layers = []
    kept = []
    removed = []
    for meta in bundle.selected_layers:
        ok = True
        reason = 'ok'
        if meta.name not in modules:
            ok = False
            reason = 'module_not_found'
        else:
            mod = modules[meta.name]
            if not isinstance(mod, nn.Conv2d):
                ok = False
                reason = 'not_conv2d'
            elif mod.out_channels <= 0:
                ok = False
                reason = 'invalid_out_channels'
        if method in {'slim', 'ours', 'ours_slim'} and meta.matched_bn is None:
            ok = False
            reason = 'bn_missing_for_slim_family'
        row = {'name': meta.name, 'ok': ok, 'reason': reason, 'stage_id': meta.stage_id, 'role': meta.role}
        checked_layers.append(row)
        if ok:
            kept.append(meta.name)
        else:
            removed.append(row)
    return {
        'checked_layers': checked_layers,
        'kept_layers': kept,
        'removed_layers': removed,
    }
