import argparse
import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from core.utils import DotDict, namespace_to_dict, save_json, str2bool, timestamp


DEFAULT_CONFIG_PATH = 'configs/cpsp.yaml'


def parse_layer_prune_cap_rule(value: Any) -> Tuple[str, float, str]:
    """Parse and canonicalize one ``PATTERN=RATIO`` layer-cap rule."""
    if not isinstance(value, str):
        raise ValueError(
            'layer_prune_cap_rules entries must be strings in PATTERN=RATIO format, '
            f'got {type(value).__name__}'
        )
    raw = value.strip()
    if raw.count('=') != 1:
        raise ValueError(
            'Invalid layer_prune_cap_rules entry '
            f'{value!r}: expected exactly one PATTERN=RATIO separator'
        )
    pattern, ratio_text = (part.strip() for part in raw.split('=', 1))
    if not pattern or not ratio_text:
        raise ValueError(
            f'Invalid layer_prune_cap_rules entry {value!r}: pattern and ratio must be non-empty'
        )
    try:
        ratio = float(ratio_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'Invalid layer_prune_cap_rules entry {value!r}: ratio must be a number in [0, 1]'
        ) from exc
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError(
            f'Invalid layer_prune_cap_rules entry {value!r}: ratio must be finite and in [0, 1]'
        )
    canonical = f'{pattern}={ratio:.12g}'
    return pattern, ratio, canonical


def normalize_layer_prune_cap_rules(value: Any) -> List[str]:
    """Return a validated canonical rule list for CLI, YAML, and direct configs."""
    if value is None:
        return []
    entries = [value] if isinstance(value, str) else value
    if not isinstance(entries, (list, tuple)):
        raise ValueError(
            'layer_prune_cap_rules must be a list of PATTERN=RATIO strings'
        )
    return [parse_layer_prune_cap_rule(entry)[2] for entry in entries]


def parse_layer_min_remaining_rule(value: Any) -> Tuple[str, int, str]:
    """Parse and canonicalize one ``PATTERN=INT`` minimum-width rule."""
    if not isinstance(value, str):
        raise ValueError(
            'layer_min_remaining_rules entries must be strings in PATTERN=INT format, '
            f'got {type(value).__name__}'
        )
    raw = value.strip()
    if raw.count('=') != 1:
        raise ValueError(
            'Invalid layer_min_remaining_rules entry '
            f'{value!r}: expected exactly one PATTERN=INT separator'
        )
    pattern, minimum_text = (part.strip() for part in raw.split('=', 1))
    if not pattern or not minimum_text:
        raise ValueError(
            f'Invalid layer_min_remaining_rules entry {value!r}: pattern and integer must be non-empty'
        )
    try:
        minimum = int(minimum_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f'Invalid layer_min_remaining_rules entry {value!r}: minimum must be an integer >= 1'
        ) from exc
    if str(minimum) != minimum_text and minimum_text not in {f'+{minimum}'}:
        raise ValueError(
            f'Invalid layer_min_remaining_rules entry {value!r}: minimum must be an integer >= 1'
        )
    if minimum < 1:
        raise ValueError(
            f'Invalid layer_min_remaining_rules entry {value!r}: minimum must be >= 1'
        )
    canonical = f'{pattern}={minimum}'
    return pattern, minimum, canonical


def normalize_layer_min_remaining_rules(value: Any) -> List[str]:
    """Return a validated canonical per-layer minimum-width rule list."""
    if value is None:
        return []
    entries = [value] if isinstance(value, str) else value
    if not isinstance(entries, (list, tuple)):
        raise ValueError(
            'layer_min_remaining_rules must be a list of PATTERN=INT strings'
        )
    return [parse_layer_min_remaining_rule(entry)[2] for entry in entries]


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH, help='YAML config path.')
    parser.add_argument('--model', type=str, default='DNANet')
    parser.add_argument('--dataset', type=str, default='NUAA-SIRST')
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./experiments')
    parser.add_argument('--experiment_name', type=str, default=None)
    parser.add_argument('--mode', type=str, default='baseline_train', choices=['baseline_train', 'slim_train', 'ocp_train', 'eval', 'analyze_polarization', 'sweep_pruning', 'compare_pruning'])
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--save_best_metric', type=str, default='mIoU')
    parser.add_argument('--save_every_epoch_ckpt', type=str2bool, default=False)
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers; 0 uses reliable single-process loading for train/test/analysis.')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--optimizer_name', type=str, default='Adam', choices=['Adam', 'Adagrad', 'SGD'])
    parser.add_argument('--scheduler_name', type=str, default='MultiStepLR', choices=['MultiStepLR', 'CosineAnnealingLR', 'None'])
    parser.add_argument('--scheduler_milestones', type=int, nargs='*', default=[60, 80])
    parser.add_argument('--scheduler_gamma', type=float, default=0.1)
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-5)
    parser.add_argument('--img_norm_cfg_mean', type=float, default=None)
    parser.add_argument('--img_norm_cfg_std', type=float, default=None)
    parser.add_argument('--img_norm_cfg', type=float, nargs=2, default=None, help='Optional mean std pair.')
    parser.add_argument('--save_predictions', type=str2bool, default=False)
    parser.add_argument('--inference_speed_repeats', type=int, default=50)
    parser.add_argument('--sanity_analysis_samples', type=int, default=8)
    parser.add_argument('--sanity_val_samples', type=int, default=8)
    parser.add_argument(
        '--analysis_split',
        type=str,
        default='train',
        choices=['train', 'test'],
        help=(
            'Dataset split used for OCP/pruning statistics. The default uses '
            'the training index with deterministic evaluation preprocessing, '
            'so the official test set is not consumed during method design.'
        ),
    )
    parser.add_argument(
        '--analysis_allow_test',
        type=str2bool,
        default=False,
        help=(
            'Explicitly allow analysis_split=test. Disabled by default '
            'because using the official test index for method design risks '
            'evaluation leakage.'
        ),
    )
    parser.add_argument(
        '--analysis_seed',
        type=int,
        default=42,
        help='Independent seed used only for deterministic analysis subsampling.',
    )
    parser.add_argument(
        '--analysis_index',
        type=str,
        default=None,
        help=(
            'Optional explicit image-index text file for analysis. Relative '
            'paths are checked from the current directory, dataset directory, '
            'and dataset img_idx directory, in that order.'
        ),
    )
    parser.add_argument('--pruning_bundle_json', type=str, default=None,
                        help='Path to auto_prunable_layers.json used to rebuild a pruned model for evaluation.')
    parser.add_argument('--pruning_plan_json', type=str, default=None,
                        help='Path to pruning_plan.json used to rebuild a pruned model for evaluation.')


def _add_ocp_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--ocp_enable', type=str2bool, default=False)
    parser.add_argument('--ocp_layers', type=str, nargs='*', default=[])
    parser.add_argument('--lambda_pol', type=float, default=1e-2)
    parser.add_argument('--lambda_pre', type=float, default=1e-3)
    parser.add_argument('--ocp_stat_subset', type=int, default=256)
    parser.add_argument('--ocp_warmup_epochs', type=int, default=10)
    parser.add_argument('--ocp_compute_interval', type=int, default=1)
    parser.add_argument('--min_valid_target_pixels', type=float, default=1e-3)
    parser.add_argument('--ocp_loss_clip', type=float, default=5.0)
    parser.add_argument('--ocp_mapping_strategy', type=str, default='stage_then_nearest', choices=['same_layer', 'nearest', 'stage_then_nearest', 'stage_share'])
    parser.add_argument('--auto_select_ocp_layers', type=str2bool, default=True)
    parser.add_argument('--max_ocp_layers', type=int, default=6)
    parser.add_argument('--ocp_layer_selection_strategy', type=str, default='per_stage_representative', choices=['topk_safe_layers', 'per_stage_representative', 'all_safe_layers'])
    parser.add_argument('--analysis_compare_with', type=str, default=None)


def _add_pruning_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--pruning_method', type=str, default='ours', choices=['l1', 'slim', 'depgraph', 'ours', 'ours_l1', 'ours_slim'])
    parser.add_argument('--pruning_rates', type=float, nargs='*', default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument('--pruning_layers', type=str, nargs='*', default=[])
    parser.add_argument('--finetune_epochs', type=int, default=40)
    parser.add_argument('--finetune_lr', type=float, default=1e-4)
    parser.add_argument(
        '--plan_only',
        type=str2bool,
        default=False,
        help=(
            'Build and physically apply each pruning plan, evaluate the '
            'pre-finetune model, and profile it without running finetuning.'
        ),
    )
    parser.add_argument('--pruning_score_source', type=str, default='ours_slim', choices=['l1', 'slim', 'ours_l1', 'ours_slim'])
    parser.add_argument('--slim_bn_only', type=str2bool, default=True)
    parser.add_argument('--slim_regularization_enable', type=str2bool, default=False)
    parser.add_argument('--slim_lambda', type=float, default=1e-4)
    parser.add_argument('--collapse_metric', type=str, default='mIoU')
    parser.add_argument('--collapse_ratio_threshold', type=float, default=0.15)

    parser.add_argument('--auto_select_pruning_layers', type=str2bool, default=True)
    parser.add_argument('--min_prunable_channels', type=int, default=16)
    parser.add_argument('--exclude_first_conv', type=str2bool, default=True)
    parser.add_argument('--exclude_last_head', type=str2bool, default=True)
    parser.add_argument('--allow_group_conv_pruning', type=str2bool, default=False)
    parser.add_argument('--allow_depthwise_pruning', type=str2bool, default=False)
    parser.add_argument('--pruning_layer_selection_strategy', type=str, default='safe_bn_conv', choices=['safe_bn_conv', 'safe_conv_only', 'all_safe_layers'])
    parser.add_argument('--dry_run_pruning_plan', type=str2bool, default=True)

    parser.add_argument('--global_prune_ratio', type=float, nargs='*', default=[0.5])
    parser.add_argument('--adaptive_budget_strategy', type=str, default='weighted_waterfill', choices=['weighted_waterfill', 'global_rank_then_cap', 'two_stage'])
    parser.add_argument('--global_planner', type=str, default='cost_aware_global',
                        choices=['cost_aware_global', 'adaptive_layer_budget'],
                        help='Global candidate ranking (recommended) or the legacy per-layer budget allocator.')
    parser.add_argument('--target_aware_beta', type=float, default=0.5,
                        help='Weight of target-carrying responsibility in the global removal cost.')
    parser.add_argument('--structural_gain_alpha', type=float, default=0.0,
                        help='Weight of parameter reduction versus FLOPs reduction in structural gain.')
    parser.add_argument('--min_remaining_channels_per_layer', type=int, default=8)
    parser.add_argument('--max_layer_prune_ratio', type=float, default=0.8)
    parser.add_argument(
        '--layer_prune_cap_rules',
        type=str,
        nargs='*',
        default=[],
        metavar='PATTERN=RATIO',
        help=(
            'Optional fnmatch-style per-layer maximum prune ratios. Multiple '
            'matching rules use the strictest (smallest) ratio, and the final '
            'cap never exceeds max_layer_prune_ratio.'
        ),
    )
    parser.add_argument(
        '--layer_min_remaining_rules',
        type=str,
        nargs='*',
        default=[],
        metavar='PATTERN=INT',
        help=(
            'Optional fnmatch-style per-layer minimum remaining channels. '
            'Multiple matching rules use the strictest (largest) value, and '
            'the final minimum is never below min_remaining_channels_per_layer.'
        ),
    )
    parser.add_argument('--min_layer_prune_ratio', type=float, default=0.0)
    parser.add_argument('--layer_prunability_metric', type=str, default='combined', choices=['mean_inverse_q', 'mean_inverse_s', 'mean_inverse_u', 'combined'])
    parser.add_argument('--budget_smoothing', type=float, default=1.0)
    parser.add_argument('--budget_rounding_mode', type=str, default='round', choices=['floor', 'round', 'ceil'])
    parser.add_argument('--dry_run_budget_check', type=str2bool, default=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Refactored BasicIRSTD training / OCP / pruning entry parser.')
    _add_common_args(parser)
    _add_ocp_args(parser)
    _add_pruning_args(parser)
    return parser


def _normalize_config_values(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if cfg.get('img_norm_cfg') is not None and isinstance(cfg['img_norm_cfg'], (list, tuple)) and len(cfg['img_norm_cfg']) == 2:
        cfg['img_norm_cfg'] = {'mean': float(cfg['img_norm_cfg'][0]), 'std': float(cfg['img_norm_cfg'][1])}
    if cfg.get('img_norm_cfg_mean') is not None and cfg.get('img_norm_cfg_std') is not None:
        cfg['img_norm_cfg'] = {'mean': float(cfg['img_norm_cfg_mean']), 'std': float(cfg['img_norm_cfg_std'])}
    for key in [
        'pruning_rates',
        'global_prune_ratio',
        'scheduler_milestones',
        'pruning_layers',
        'ocp_layers',
        'layer_prune_cap_rules',
        'layer_min_remaining_rules',
    ]:
        if key in cfg and cfg[key] is None:
            cfg[key] = []
    cfg['layer_prune_cap_rules'] = normalize_layer_prune_cap_rules(
        cfg.get('layer_prune_cap_rules', [])
    )
    cfg['layer_min_remaining_rules'] = normalize_layer_min_remaining_rules(
        cfg.get('layer_min_remaining_rules', [])
    )
    if isinstance(cfg.get('global_prune_ratio'), (float, int)):
        cfg['global_prune_ratio'] = [float(cfg['global_prune_ratio'])]
    if isinstance(cfg.get('pruning_rates'), (float, int)):
        cfg['pruning_rates'] = [float(cfg['pruning_rates'])]
    if cfg.get('experiment_name') in (None, ''):
        cfg['experiment_name'] = f"{cfg.get('dataset', 'dataset')}_{cfg.get('model', 'model')}_{cfg.get('mode', 'mode')}_{timestamp()}"
    return cfg


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    yaml_path = Path(path)
    if not yaml_path.exists():
        return {}
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data


def parse_args(argv: Optional[List[str]] = None) -> DotDict:
    parser = build_parser()
    # First parse only --config to load yaml defaults.
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH)
    config_ns, _ = config_parser.parse_known_args(argv)
    yaml_cfg = load_yaml_config(config_ns.config)
    parser.set_defaults(**yaml_cfg)
    ns = parser.parse_args(argv)
    cfg = _normalize_config_values(namespace_to_dict(ns))
    cfg['config'] = ns.config
    return DotDict(cfg)


def build_run_dir(cfg: DotDict, subdir: Optional[str] = None) -> str:
    run_dir = Path(cfg.save_dir) / cfg.dataset / cfg.model / cfg.experiment_name
    if subdir:
        run_dir = run_dir / subdir
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def dump_effective_config(cfg: DotDict, run_dir: str, filename: str = 'config.effective.json') -> None:
    path = Path(run_dir) / filename
    save_json(dict(cfg), str(path))
    if cfg.get('config') and Path(cfg.config).exists():
        base = load_yaml_config(cfg.config)
        save_json(base, str(Path(run_dir) / 'config.base.json'))


def clone_cfg(cfg: DotDict, **updates: Any) -> DotDict:
    new_cfg = DotDict(copy.deepcopy(dict(cfg)))
    for k, v in updates.items():
        new_cfg[k] = v
    return _normalize_config_values(new_cfg)
