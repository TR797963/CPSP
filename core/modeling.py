from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, Optional

import torch
from torch import nn

from loss import SoftIoULoss


@dataclass
class ModelBundle:
    model: nn.Module
    criterion: nn.Module
    model_name: str
    mode: str



def _lazy_build_model(model_name: str, mode: str) -> nn.Module:
    """Lazy per-model import.

    This avoids importing optional dependencies for models that are not used. For example ACM/ALCNet
    import torchvision backbones, while the default DNANet path does not need them.
    """
    if model_name == 'DNANet':
        mod = import_module('model.DNANet.model_DNANet')
        return mod.DNANet(mode='train' if mode == 'train' else 'test')
    if model_name == 'ACM':
        mod = import_module('model.ACM.model_ACM')
        return mod.ASKCResUNet()
    if model_name == 'ALCNet':
        mod = import_module('model.ACM.model_ALCnet')
        return mod.ASKCResNetFPN()
    if model_name == 'ISNet':
        from compat.isnet import build_isnet
        return build_isnet(mode='train' if mode == 'train' else 'test')
    if model_name == 'RISTDnet':
        from compat.ristd import build_ristdnet
        return build_ristdnet()
    if model_name == 'UIUNet':
        mod = import_module('model.UIUNet.model_UIUNet')
        return mod.UIUNET(mode='train' if mode == 'train' else 'test')
    if model_name == 'U-Net':
        mod = import_module('model.Unet.model_Unet')
        return mod.U_Net()
    if model_name == 'ISTDU-Net':
        mod = import_module('model.ISTDUNet.model_ISTDUNet')
        return mod.ISTDU_Net()
    if model_name == 'RDIAN':
        mod = import_module('model.RDIAN.model_RDIAN')
        return mod.RDIAN()
    if model_name == 'ResUNet':
        mod = import_module('model.ResUNet.model_ResUNet')
        return mod.ResUNet()
    raise ValueError(f'Unsupported model: {model_name}')



def build_model(model_name: str, mode: str = 'train') -> ModelBundle:
    criterion: nn.Module = SoftIoULoss()
    model = _lazy_build_model(model_name, mode)
    if model_name == 'ISNet':
        from compat.isnet import ISNetStableLoss
        criterion = ISNetStableLoss()
    return ModelBundle(model=model, criterion=criterion, model_name=model_name, mode=mode)


@torch.no_grad()
def extract_primary_prediction(
    raw_output: Any, model_name: Optional[str] = None
) -> torch.Tensor:
    if torch.is_tensor(raw_output):
        return raw_output
    if isinstance(raw_output, dict):
        for key in ['pred', 'prediction', 'seg', 'mask', 'out']:
            if key in raw_output and torch.is_tensor(raw_output[key]):
                return raw_output[key]
        for value in raw_output.values():
            if torch.is_tensor(value):
                return value
    if isinstance(raw_output, (list, tuple)):
        tensors = [item for item in raw_output if torch.is_tensor(item)]
        if not tensors:
            raise TypeError(f'No tensor-like prediction found in output type={type(raw_output)}')
        # Output ordering is backbone-specific. UIUNet returns d0 first and
        # ISNet returns (segmentation, edge), whereas DNANet's deepest fused
        # prediction is the last item.
        if model_name in {'UIUNet', 'ISNet'}:
            return tensors[0]
        return tensors[-1]
    raise TypeError(f'Unsupported raw output type: {type(raw_output)}')



def compute_detection_loss(bundle: ModelBundle, raw_output: Any, target: torch.Tensor) -> torch.Tensor:
    return bundle.criterion(raw_output, target)



def _strip_known_prefixes_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    prefixes=('module.', 'model.')
) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict

    out = state_dict
    changed = True
    while changed:
        changed = False
        keys = list(out.keys())
        if not keys:
            break
        for prefix in prefixes:
            if all(k.startswith(prefix) for k in keys):
                out = {k[len(prefix):]: v for k, v in out.items()}
                changed = True
                break
    return out


def state_dict_strip_module(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # 保留旧函数名，避免其他调用点改动
    return _strip_known_prefixes_from_state_dict(state_dict)


def _drop_thop_profiler_buffers(
    state_dict: Dict[str, torch.Tensor],
) -> tuple:
    """Drop only buffers whose final key component is a known THOP field."""
    profiler_buffer_names = {'total_ops', 'total_params'}
    kept = {}
    dropped = []
    for key, value in state_dict.items():
        final_component = key.rsplit('.', 1)[-1] if isinstance(key, str) else None
        if final_component in profiler_buffer_names:
            dropped.append(key)
        else:
            kept[key] = value
    return kept, dropped


def _adapt_legacy_isnet_checkpoint(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> tuple:
    """Supply only fixed legacy gradient tensors absent from old checkpoints."""
    uses_external_dcn = any(
        getattr(module, 'dcn_backend', None) == 'torchvision_deform_conv2d'
        for module in model.modules()
    )
    if not uses_external_dcn:
        return state_dict, None

    adapted = dict(state_dict)
    model_state = model.state_dict()
    supplied = []
    for key in ('grad.weight_h', 'grad.weight_v'):
        if key in model_state and key not in adapted:
            adapted[key] = model_state[key]
            supplied.append(key)
    return adapted, {
        'type': 'isnet_external_torchvision_dcn',
        'dropped_keys': [],
        'supplied_fixed_buffers': supplied,
    }


# Backward-compatible private name for older callers/tests.
_adapt_legacy_isnet_dcn_checkpoint = _adapt_legacy_isnet_checkpoint



def load_model_weights(model: nn.Module, checkpoint_path: str, device: torch.device, strict: bool = True) -> Dict[str, Any]:
    # Project checkpoints may include optimizer/scheduler metadata in addition
    # to tensors, so the required pickle mode is explicit on modern Torch.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 常见 checkpoint 格式兼容
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        elif 'model' in ckpt and isinstance(ckpt['model'], dict):
            state_dict = ckpt['model']
        elif 'net' in ckpt and isinstance(ckpt['net'], dict):
            state_dict = ckpt['net']
        else:
            # 直接就是 state_dict
            state_dict = ckpt
    else:
        state_dict = ckpt

    state_dict = state_dict_strip_module(state_dict)
    state_dict, dropped_profiler_keys = _drop_thop_profiler_buffers(state_dict)
    state_dict, compatibility_adaptation = _adapt_legacy_isnet_checkpoint(model, state_dict)

    incompat = model.load_state_dict(state_dict, strict=strict)
    missing = list(getattr(incompat, 'missing_keys', []))
    unexpected = list(getattr(incompat, 'unexpected_keys', []))

    return {
        'checkpoint': ckpt,
        'missing_keys': missing,
        'unexpected_keys': unexpected,
        'dropped_profiler_keys': dropped_profiler_keys,
        'dropped_profiler_key_count': len(dropped_profiler_keys),
        'compatibility_adaptation': compatibility_adaptation,
    }


def maybe_data_parallel(model: nn.Module) -> nn.Module:
    if torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model



def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, 'module') else model



def named_modules_dict(model: nn.Module) -> Dict[str, nn.Module]:
    return {name: module for name, module in model.named_modules()}



def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    modules = named_modules_dict(model)
    if module_name not in modules:
        raise KeyError(f'Module {module_name} not found. Available example keys: {list(modules.keys())[:20]}')
    return modules[module_name]
