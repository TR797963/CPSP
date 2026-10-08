from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from analysis.reporting import plot_polarization_stats
from core.modeling import get_module_by_name, unwrap_model
from core.utils import AverageMeter, save_json, save_simple_csv, safe_mean
from pruning.auto_select import LayerMeta, SelectionBundle


@dataclass
class OCPBatchResult:
    pol_loss: torch.Tensor
    pre_loss: torch.Tensor
    total_extra_loss: torch.Tensor
    avg_q: float
    avg_P: float
    avg_E_ent: float
    valid_samples: int
    invalid_samples: int
    per_layer: Dict[str, Dict[str, float]]


class FeatureHookManager:
    def __init__(self, model: nn.Module, layer_names: Sequence[str]) -> None:
        self.model = unwrap_model(model)
        self.layer_names = list(layer_names)
        self.handles: List[Any] = []
        self.features: Dict[str, torch.Tensor] = {}

    def register(self) -> None:
        self.remove()
        for layer_name in self.layer_names:
            module = get_module_by_name(self.model, layer_name)

            def hook_fn(_module: nn.Module, _inputs: Tuple[torch.Tensor, ...], output: torch.Tensor, name: str = layer_name) -> None:
                self.features[name] = output

            self.handles.append(module.register_forward_hook(hook_fn))

    def clear(self) -> None:
        self.features = {}

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.clear()


class OCPComputer:
    """Engineering-friendly OCP implementation.

    Notes / approximations (also documented in README):
    1. OCP uses forward hooks on a *small automatically selected set* of safe Conv2d layers.
    2. GT mask is resized with area interpolation for downsampling and bilinear
       interpolation for upsampling, then clamped to [0, 1].  This preserves
       tiny-target mass at deep feature resolutions.
    3. Empty-mask / extremely tiny target samples are skipped based on `min_valid_target_pixels`.
    4. If a pruning layer has no direct OCP stats later, q is mapped via explicit same-layer / same-stage / nearest-layer rules.
    """

    def __init__(self, model: nn.Module, layer_names: Sequence[str], cfg: Any) -> None:
        self.cfg = cfg
        self.layer_names = list(layer_names)
        self.hooks = FeatureHookManager(model, self.layer_names)
        self.hooks.register()

    def close(self) -> None:
        self.hooks.remove()

    def clear(self) -> None:
        self.hooks.clear()

    def _compute_single_layer(self, feat: torch.Tensor, mask: torch.Tensor, layer_name: str) -> Dict[str, Any]:
        eps = 1e-6
        b, c, h, w = feat.shape
        # Area downsampling preserves the fractional mass of tiny targets much
        # better than point-sampled bilinear interpolation, which can erase a
        # 1-2 pixel object entirely at deep feature resolutions.  Bilinear is
        # still appropriate when the feature map is larger than the mask.
        if h <= mask.shape[-2] and w <= mask.shape[-1]:
            resized_mask = F.interpolate(mask.float(), size=(h, w), mode='area')
        else:
            resized_mask = F.interpolate(
                mask.float(), size=(h, w), mode='bilinear', align_corners=False
            )
        resized_mask = resized_mask.clamp_(0.0, 1.0)
        target_pixels = resized_mask.flatten(1).sum(dim=1)
        # Tiny infrared targets may contribute much less than one pixel after
        # bilinear downsampling.  Keep every non-empty soft target by default;
        # the configurable floor only filters genuinely empty / numerically
        # degenerate samples.
        valid = target_pixels > float(self.cfg.min_valid_target_pixels)
        valid_count = int(valid.sum().item())
        invalid_count = int((~valid).sum().item())

        if valid_count == 0:
            zero = feat.sum() * 0.0
            return {
                'layer_name': layer_name,
                'pol_loss': zero,
                'pre_loss': zero,
                'mean_q': 0.0,
                'mean_P': 0.0,
                'mean_E_ent': 0.0,
                'valid_samples': 0,
                'invalid_samples': invalid_count,
                'q_matrix': None,
                'e_t': None,
                'e_b': None,
            }

        feat_abs = feat.abs()[valid]
        mask_v = resized_mask[valid]
        inv_mask_v = 1.0 - mask_v
        feat_flat = feat_abs.flatten(2)
        mask_flat = mask_v.flatten(2)
        inv_mask_flat = inv_mask_v.flatten(2)

        target_den = mask_flat.sum(dim=-1) + eps
        bg_den = inv_mask_flat.sum(dim=-1) + eps

        e_t = (feat_flat * mask_flat).sum(dim=-1) / target_den
        e_b = (feat_flat * inv_mask_flat).sum(dim=-1) / bg_den
        q = e_t / (e_t + e_b + eps)
        ent = (e_t * e_b) / ((e_t + e_b) ** 2 + eps)
        pol_loss = (4.0 * q * (1.0 - q)).mean()
        # Preserve the layer-level target energy instead of forcing every
        # channel to stay active.  Penalising log(e_t) channel-by-channel fights
        # polarization because background-specialised channels are expected to
        # have low target energy.  The layer mean follows the intended CP idea:
        # allow specialisation while preventing the whole layer from collapsing.
        layer_target_energy = e_t.mean(dim=1)
        pre_loss = -(torch.log(layer_target_energy + eps)).mean()
        p_value = torch.abs(2.0 * q - 1.0).mean()
        e_ent = ent.mean()

        if self.cfg.ocp_loss_clip is not None and self.cfg.ocp_loss_clip > 0:
            pol_loss = torch.clamp(pol_loss, max=float(self.cfg.ocp_loss_clip))
            pre_loss = torch.clamp(pre_loss, max=float(self.cfg.ocp_loss_clip))

        return {
            'layer_name': layer_name,
            'pol_loss': pol_loss,
            'pre_loss': pre_loss,
            'mean_q': float(q.mean().detach().cpu().item()),
            'mean_P': float(p_value.detach().cpu().item()),
            'mean_E_ent': float(e_ent.detach().cpu().item()),
            'valid_samples': valid_count,
            'invalid_samples': invalid_count,
            'q_matrix': q,
            'e_t': e_t,
            'e_b': e_b,
        }

    def compute_batch(self, mask: torch.Tensor, epoch: int, global_step: int) -> OCPBatchResult:
        enabled = bool(self.cfg.ocp_enable) and epoch >= int(self.cfg.ocp_warmup_epochs) and (global_step % max(int(self.cfg.ocp_compute_interval), 1) == 0)
        if not enabled or not self.hooks.features:
            zero = mask.sum() * 0.0
            return OCPBatchResult(
                pol_loss=zero,
                pre_loss=zero,
                total_extra_loss=zero,
                avg_q=0.0,
                avg_P=0.0,
                avg_E_ent=0.0,
                valid_samples=0,
                invalid_samples=int(mask.shape[0]),
                per_layer={},
            )

        layer_results: List[Dict[str, Any]] = []
        for layer_name, feat in self.hooks.features.items():
            if not torch.is_tensor(feat) or feat.ndim != 4:
                continue
            layer_results.append(self._compute_single_layer(feat, mask, layer_name))

        valid_layers = [x for x in layer_results if x['valid_samples'] > 0]
        if not valid_layers:
            zero = mask.sum() * 0.0
            return OCPBatchResult(
                pol_loss=zero,
                pre_loss=zero,
                total_extra_loss=zero,
                avg_q=0.0,
                avg_P=0.0,
                avg_E_ent=0.0,
                valid_samples=0,
                invalid_samples=int(mask.shape[0]),
                per_layer={x['layer_name']: {'valid_samples': 0, 'invalid_samples': x['invalid_samples']} for x in layer_results},
            )

        pol_loss = torch.stack([x['pol_loss'] for x in valid_layers]).mean()
        pre_loss = torch.stack([x['pre_loss'] for x in valid_layers]).mean()
        total_extra = float(self.cfg.lambda_pol) * pol_loss + float(self.cfg.lambda_pre) * pre_loss

        per_layer: Dict[str, Dict[str, float]] = {}
        avg_q_list: List[float] = []
        avg_P_list: List[float] = []
        avg_E_list: List[float] = []
        valid_samples = 0
        invalid_samples = 0
        for item in layer_results:
            per_layer[item['layer_name']] = {
                'mean_q': item['mean_q'],
                'mean_P': item['mean_P'],
                'mean_E_ent': item['mean_E_ent'],
                'valid_samples': item['valid_samples'],
                'invalid_samples': item['invalid_samples'],
            }
            if item['valid_samples'] > 0:
                avg_q_list.append(item['mean_q'])
                avg_P_list.append(item['mean_P'])
                avg_E_list.append(item['mean_E_ent'])
            valid_samples += int(item['valid_samples'])
            invalid_samples += int(item['invalid_samples'])

        return OCPBatchResult(
            pol_loss=pol_loss,
            pre_loss=pre_loss,
            total_extra_loss=total_extra,
            avg_q=safe_mean(avg_q_list, 0.0),
            avg_P=safe_mean(avg_P_list, 0.0),
            avg_E_ent=safe_mean(avg_E_list, 0.0),
            valid_samples=valid_samples,
            invalid_samples=invalid_samples,
            per_layer=per_layer,
        )


class LayerAccumulator:
    def __init__(self, layer_name: str, out_channels: int) -> None:
        self.layer_name = layer_name
        self.out_channels = int(out_channels)
        self.sum_et = np.zeros(self.out_channels, dtype=np.float64)
        self.sum_eb = np.zeros(self.out_channels, dtype=np.float64)
        self.sum_q = np.zeros(self.out_channels, dtype=np.float64)
        self.valid_samples = 0
        self.invalid_samples = 0
        self.mean_P_list: List[float] = []
        self.mean_E_ent_list: List[float] = []

    def update(self, e_t: torch.Tensor, e_b: torch.Tensor, q: torch.Tensor) -> None:
        # e_t/e_b/q shape: [valid_samples, C]
        e_t_np = e_t.detach().cpu().numpy()
        e_b_np = e_b.detach().cpu().numpy()
        q_np = q.detach().cpu().numpy()
        self.sum_et += e_t_np.mean(axis=0)
        self.sum_eb += e_b_np.mean(axis=0)
        self.sum_q += q_np.mean(axis=0)
        self.valid_samples += 1
        self.mean_P_list.append(float(np.mean(np.abs(2.0 * q_np - 1.0))))
        self.mean_E_ent_list.append(float(np.mean((e_t_np * e_b_np) / ((e_t_np + e_b_np) ** 2 + 1e-6))))

    def add_invalid(self, n: int) -> None:
        self.invalid_samples += int(n)

    def finalize(self) -> Dict[str, Any]:
        denom = max(self.valid_samples, 1)
        mean_et = self.sum_et / denom
        mean_eb = self.sum_eb / denom
        mean_q = self.sum_q / denom
        return {
            'layer_name': self.layer_name,
            'valid_samples': int(self.valid_samples),
            'invalid_samples': int(self.invalid_samples),
            'mean_e_t': mean_et.tolist(),
            'mean_e_b': mean_eb.tolist(),
            'q_values': mean_q.tolist(),
            'mean_q': float(np.mean(mean_q)) if len(mean_q) else 0.0,
            'mean_P': float(np.mean(self.mean_P_list)) if self.mean_P_list else 0.0,
            'mean_E_ent': float(np.mean(self.mean_E_ent_list)) if self.mean_E_ent_list else 0.0,
        }


@torch.no_grad()
def collect_ocp_statistics(model: nn.Module, layer_bundle: SelectionBundle, data_loader: torch.utils.data.DataLoader, device: torch.device, cfg: Any, out_dir: Optional[str] = None) -> Dict[str, Any]:
    layer_names = layer_bundle.selected_names()
    ocp = OCPComputer(model, layer_names, cfg)
    model.eval()
    accumulators: Dict[str, LayerAccumulator] = {meta.name: LayerAccumulator(meta.name, meta.out_channels) for meta in layer_bundle.selected_layers}

    for batch_idx, batch in enumerate(data_loader):
        img, mask, size, name = batch
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        ocp.clear()
        _ = model(img)
        for layer_name, feat in ocp.hooks.features.items():
            if layer_name not in accumulators or feat.ndim != 4:
                continue
            res = ocp._compute_single_layer(feat, mask, layer_name)
            if res['valid_samples'] > 0 and res['q_matrix'] is not None:
                accumulators[layer_name].update(res['e_t'], res['e_b'], res['q_matrix'])
            else:
                accumulators[layer_name].add_invalid(int(mask.shape[0]))

    ocp.close()
    layers: Dict[str, Any] = {}
    layer_rows: List[Dict[str, Any]] = []
    global_q: List[float] = []
    global_p: List[float] = []
    global_e: List[float] = []
    for meta in layer_bundle.selected_layers:
        finalized = accumulators[meta.name].finalize()
        layers[meta.name] = finalized
        layer_rows.append({
            'layer_name': meta.name,
            'role': meta.role,
            'stage_id': meta.stage_id,
            'valid_samples': finalized['valid_samples'],
            'invalid_samples': finalized['invalid_samples'],
            'mean_q': finalized['mean_q'],
            'mean_P': finalized['mean_P'],
            'mean_E_ent': finalized['mean_E_ent'],
            'out_channels': meta.out_channels,
        })
        global_q.extend(finalized['q_values'])
        global_p.append(finalized['mean_P'])
        global_e.append(finalized['mean_E_ent'])

    result = {
        'config_note': {
            'mask_resize_mode': 'area_downsample_bilinear_upsample',
            'mask_resize_align_corners': False,
            'min_valid_target_pixels': int(cfg.min_valid_target_pixels),
            'ocp_mapping_strategy': str(cfg.ocp_mapping_strategy),
            'engineering_approximation': 'forward-hook + resized GT mask + representative layers only',
        },
        'enabled_layers': layer_names,
        'layers': layers,
        'layer_rows': layer_rows,
        'global': {
            'mean_q': float(np.mean(global_q)) if global_q else 0.0,
            'mean_P': float(np.mean(global_p)) if global_p else 0.0,
            'mean_E_ent': float(np.mean(global_e)) if global_e else 0.0,
            'num_layers': len(layer_names),
            'num_channels': int(sum(len(v['q_values']) for v in layers.values())),
        },
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_json(result, str(out / 'ocp_stats.json'))
        save_simple_csv(layer_rows, str(out / 'ocp_stats.csv'))
        plot_polarization_stats(result, str(out))
    return result



def _nearest_by_depth(candidates: List[Dict[str, Any]], depth_index: int) -> Dict[str, Any]:
    return sorted(candidates, key=lambda x: abs(int(x['depth_index']) - int(depth_index)))[0]



def project_q_vector(q_values: Sequence[float], target_channels: int) -> List[float]:
    """Resize q-vector to target channel count.

    Engineering approximation used for pruning-layer mapping when OCP hook channels differ from
    pruning-layer channels. This is explicit, logged, and never silent.
    """
    q = np.array(list(q_values), dtype=np.float32)
    if len(q) == target_channels:
        return q.tolist()
    if len(q) == 0:
        return [0.5] * int(target_channels)
    if target_channels <= 1:
        return [float(np.mean(q))]
    src = np.linspace(0.0, 1.0, num=len(q))
    dst = np.linspace(0.0, 1.0, num=target_channels)
    resized = np.interp(dst, src, q)
    return resized.tolist()



def build_ocp_layer_mapping(pruning_bundle: SelectionBundle, ocp_bundle: SelectionBundle, ocp_stats: Dict[str, Any], strategy: str, out_dir: Optional[str] = None) -> Dict[str, Any]:
    ocp_meta = {meta.name: meta for meta in ocp_bundle.selected_layers}
    ocp_rows = {meta.name: {'name': meta.name, 'stage_id': meta.stage_id, 'role': meta.role, 'depth_index': meta.depth_index, 'out_channels': meta.out_channels} for meta in ocp_bundle.selected_layers}
    mappings: Dict[str, Any] = {}
    for p_meta in pruning_bundle.selected_layers:
        mapping: Dict[str, Any] = {'pruning_layer': p_meta.name, 'pruning_stage_id': p_meta.stage_id, 'pruning_out_channels': p_meta.out_channels}
        if p_meta.name in ocp_stats.get('layers', {}):
            src_name = p_meta.name
            mapping.update({'mapped_ocp_layer': src_name, 'rule': 'same_layer', 'source_out_channels': ocp_rows[src_name]['out_channels']})
        else:
            candidates = []
            if strategy in {'stage_then_nearest', 'stage_share'}:
                candidates = [row for row in ocp_rows.values() if row['stage_id'] == p_meta.stage_id]
                if candidates:
                    src = _nearest_by_depth(candidates, p_meta.depth_index)
                    mapping.update({'mapped_ocp_layer': src['name'], 'rule': 'same_stage_nearest', 'source_out_channels': src['out_channels']})
                else:
                    src = _nearest_by_depth(list(ocp_rows.values()), p_meta.depth_index)
                    mapping.update({'mapped_ocp_layer': src['name'], 'rule': 'nearest_depth_fallback', 'source_out_channels': src['out_channels']})
            elif strategy == 'nearest':
                src = _nearest_by_depth(list(ocp_rows.values()), p_meta.depth_index)
                mapping.update({'mapped_ocp_layer': src['name'], 'rule': 'nearest_depth', 'source_out_channels': src['out_channels']})
            elif strategy == 'same_layer':
                mapping.update({'mapped_ocp_layer': None, 'rule': 'same_layer_required_but_missing', 'source_out_channels': None})
            else:
                src = _nearest_by_depth(list(ocp_rows.values()), p_meta.depth_index)
                mapping.update({'mapped_ocp_layer': src['name'], 'rule': 'nearest_depth_default', 'source_out_channels': src['out_channels']})

        src_name = mapping.get('mapped_ocp_layer')
        if src_name is not None and src_name in ocp_stats.get('layers', {}):
            raw_q = ocp_stats['layers'][src_name]['q_values']
            projected = project_q_vector(raw_q, p_meta.out_channels)
            mapping['projected_q_values'] = projected
            mapping['projection'] = {
                'from_channels': len(raw_q),
                'to_channels': p_meta.out_channels,
                'mode': 'identity' if len(raw_q) == p_meta.out_channels else 'linear_interp',
            }
        else:
            mapping['projected_q_values'] = [0.5] * int(p_meta.out_channels)
            mapping['projection'] = {
                'from_channels': 0,
                'to_channels': p_meta.out_channels,
                'mode': 'fallback_constant_0.5',
            }
        mappings[p_meta.name] = mapping

    result = {
        'strategy': strategy,
        'mappings': mappings,
        'note': 'Mapping is an explicit engineering approximation: same-layer -> same-stage nearest -> nearest-depth, with q resizing if channel counts differ.',
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_json(result, str(out / 'ocp_layer_mapping.json'))
        save_simple_csv(list(mappings.values()), str(out / 'ocp_layer_mapping.csv'))
    return result
