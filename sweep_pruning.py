from pathlib import Path
from typing import Any, Dict, List

import torch
from analysis.reporting import plot_pruning_compare, plot_pruning_summary
from core.utils import plot_bar
from core.config import build_run_dir, clone_cfg, dump_effective_config, parse_args
from core.data import build_analysis_loader, build_dataloaders
from core.modeling import build_model, load_model_weights
from core.trainer import evaluate_checkpoint, evaluate_model, fit_model
from core.utils import ExperimentLogger, profile_model_safe, save_json, save_simple_csv, seed_everything, select_device
from ocp.ocp_core import build_ocp_layer_mapping, collect_ocp_statistics
from pruning.auto_select import auto_select_ocp_layers, auto_select_prunable_layers, build_selection_summary, filter_selection_bundle_by_forward, pruning_dry_run_check, save_selection_bundle
from pruning.masked_pruning import BaselinePruner, ChannelMaskApplier, DepGraphGlobalPruner, OursGlobalBudgetPruner, export_pruning_result, masked_model_effective_stats


BASELINE_METHODS = {'l1', 'slim', 'depgraph'}





def _plot_point_details(plan_result: Dict[str, Any], point_dir: Path) -> None:
    rows = plan_result.get('layer_rows', [])
    if not rows:
        return
    labels = [row['layer_name'] for row in rows]
    deletes = [float(row.get('delete_count', 0.0)) for row in rows]
    plot_bar(labels, deletes, str(point_dir / 'deleted_channel_distribution.png'), 'Deleted channels per layer', 'Layer', 'Deleted channels')
    if 'prunability_weight' in rows[0]:
        weights = [float(row.get('prunability_weight', 0.0)) for row in rows]
        budgets = [float(row.get('budget_after_correction', 0.0)) for row in rows]
        ret = [float(row.get('high_q_retention', 0.0)) for row in rows]
        plot_bar(labels, weights, str(point_dir / 'layer_prunability_scores.png'), 'Layer prunability weights', 'Layer', 'Weight')
        plot_bar(labels, budgets, str(point_dir / 'adaptive_budget_allocation.png'), 'Adaptive budget allocation', 'Layer', 'Delete count')
        plot_bar(labels, ret, str(point_dir / 'high_q_retention_per_layer.png'), 'High-q retention per layer', 'Layer', 'Retention')

def _collapse_flag(dense_metric: float, pruned_metric: float, threshold: float) -> bool:
    if dense_metric <= 0:
        return False
    return pruned_metric < dense_metric * (1.0 - threshold)


def _finetune_result_fields(
    plan_only: bool,
    finetune_summary: Dict[str, Any],
    dense_metric: float,
    collapse_ratio_threshold: float,
) -> Dict[str, Any]:
    """Return unambiguous result fields for screened and completed points.

    A plan-only point is useful for structural screening, but it is not a
    completed recovery experiment.  Keeping all post-finetune fields null
    prevents downstream collectors from accidentally ranking it alongside
    fully finetuned checkpoints.
    """
    if plan_only:
        return {
            'best_metric_after_finetune': None,
            'best_epoch_after_finetune': None,
            'best_checkpoint_path': None,
            'collapse_flag': None,
        }

    if finetune_summary is None:
        raise ValueError('finetune_summary is required when plan_only is false.')
    best_metric = float(finetune_summary['best_metric'])
    return {
        'best_metric_after_finetune': best_metric,
        'best_epoch_after_finetune': int(finetune_summary['best_epoch']),
        'best_checkpoint_path': finetune_summary['best_path'],
        'collapse_flag': bool(
            _collapse_flag(dense_metric, best_metric, collapse_ratio_threshold)
        ),
    }


def _weighted_high_q_retention(
    plan_result: Dict[str, Any],
    threshold: float = 0.7,
) -> Dict[str, Any]:
    """Aggregate retention by high-q channel count, not by layer count."""

    exact_total = 0
    exact_kept = 0
    fallback_weight = 0
    fallback_sum = 0.0
    for plan in plan_result.get('plans', {}).values():
        q_values = plan.get('q_values')
        kept_indices = plan.get('kept_indices')
        deleted_indices = plan.get('deleted_indices', [])
        if isinstance(q_values, list):
            high_q = {idx for idx, value in enumerate(q_values) if float(value) >= threshold}
            if isinstance(kept_indices, list):
                kept = {int(idx) for idx in kept_indices}
            else:
                deleted = {int(idx) for idx in deleted_indices}
                kept = set(range(len(q_values))) - deleted
            exact_total += len(high_q)
            exact_kept += len(high_q.intersection(kept))
            continue
        retention = plan.get('high_q_retention')
        candidate_count = int(plan.get('delete_count', 0)) + int(plan.get('keep_count', 0))
        if retention is not None and candidate_count > 0:
            fallback_weight += candidate_count
            fallback_sum += float(retention) * candidate_count
    if exact_total > 0:
        return {
            'value': float(exact_kept / exact_total),
            'weighting': 'high_q_channel_count',
            'high_q_channels': int(exact_total),
            'retained_high_q_channels': int(exact_kept),
        }
    if fallback_weight > 0:
        return {
            'value': float(fallback_sum / fallback_weight),
            'weighting': 'candidate_channel_count_fallback',
            'high_q_channels': None,
            'retained_high_q_channels': None,
        }
    return {
        'value': None,
        'weighting': None,
        'high_q_channels': None,
        'retained_high_q_channels': None,
    }


def _build_finetune_cfg(cfg: Any) -> Any:
    """Clone sweep config while preserving checkpoint-retention policy."""

    return clone_cfg(
        cfg,
        pretrained=None,
        resume=None,
        ocp_enable=False,
        save_predictions=False,
        mode='baseline_train',
    )


if __name__ == '__main__':
    cfg = parse_args()
    assert cfg.pretrained is not None, '--pretrained checkpoint is required for pruning sweep.'
    seed_everything(int(cfg.seed))
    sweep_dir = build_run_dir(cfg, subdir=f'pruning_{cfg.pruning_method}')
    logger = ExperimentLogger(sweep_dir)
    dump_effective_config(cfg, sweep_dir)
    logger.info(f'Start pruning sweep method={cfg.pruning_method} pretrained={cfg.pretrained}')

    train_loader, val_loader = build_dataloaders(cfg)
    analysis_loader = build_analysis_loader(cfg, subset_size=int(cfg.ocp_stat_subset))
    device = select_device(cfg.device)

    # Dense reference metrics used for collapse estimation.
    dense_eval_bundle = build_model(cfg.model, mode='test')
    dense_eval = evaluate_checkpoint(dense_eval_bundle, cfg.pretrained, val_loader, cfg, str(Path(sweep_dir) / 'dense_reference'), logger)
    dense_metric = float(dense_eval['metrics'][cfg.collapse_metric])
    dense_params = dense_eval['profile'].get('params')
    dense_flops = dense_eval['profile'].get('flops')

    # Auto-select pruning layers once and save them.
    dry_model_bundle = build_model(cfg.model, mode='test')
    load_model_weights(dry_model_bundle.model, cfg.pretrained, device, strict=True)
    score_family = cfg.pruning_method if cfg.pruning_method in BASELINE_METHODS else ('ours_l1' if cfg.pruning_method == 'ours_l1' else 'ours_slim')
    prunable_bundle = auto_select_prunable_layers(dry_model_bundle.model, cfg, method=score_family)
    prunable_bundle = filter_selection_bundle_by_forward(
        dry_model_bundle.model,
        prunable_bundle,
        input_size=(1, 1, int(cfg.patch_size), int(cfg.patch_size)),
        device=device,
    )
    save_selection_bundle(prunable_bundle, sweep_dir, 'auto_prunable_layers')
    save_json(build_selection_summary(prunable_bundle), str(Path(sweep_dir) / 'auto_prunable_layers.summary.json'))
    dry_run = pruning_dry_run_check(dry_model_bundle.model, prunable_bundle, method=score_family)
    save_json(dry_run, str(Path(sweep_dir) / 'pruning_plan_checked.json'))

    ocp_bundle = None
    ocp_stats = None
    ocp_mapping = None
    if cfg.pruning_method in {'ours', 'ours_l1', 'ours_slim'}:
        ocp_bundle = auto_select_ocp_layers(dry_model_bundle.model, cfg)
        ocp_bundle = filter_selection_bundle_by_forward(
            dry_model_bundle.model,
            ocp_bundle,
            input_size=(1, 1, int(cfg.patch_size), int(cfg.patch_size)),
            device=device,
        )
        save_selection_bundle(ocp_bundle, sweep_dir, 'auto_ocp_layers')
        ocp_stats = collect_ocp_statistics(dry_model_bundle.model.to(device), ocp_bundle, analysis_loader, device, cfg, out_dir=str(Path(sweep_dir) / 'polarization'))
        ocp_mapping = build_ocp_layer_mapping(prunable_bundle, ocp_bundle, ocp_stats, strategy=str(cfg.ocp_mapping_strategy), out_dir=str(Path(sweep_dir) / 'polarization'))
        logger.info(f'Ours will use global-budget pruning with OCP mapping strategy={cfg.ocp_mapping_strategy}.')

    point_values = list(cfg.pruning_rates) if cfg.pruning_method in BASELINE_METHODS else list(cfg.global_prune_ratio)
    summary_rows: List[Dict[str, Any]] = []
    method_key = cfg.pruning_method if cfg.pruning_method in BASELINE_METHODS else 'ours'

    for point in point_values:
        point_value = float(point)
        point_tag = f'prune_{point_value:.2f}' if cfg.pruning_method in BASELINE_METHODS else f'global_{point_value:.2f}'
        point_dir = Path(sweep_dir) / point_tag
        point_dir.mkdir(parents=True, exist_ok=True)
        point_logger = ExperimentLogger(str(point_dir), txt_name='point.log.txt', csv_name='point.records.csv', jsonl_name='point.records.jsonl')
        point_logger.info(f'Running pruning point={point_tag}')

        model_bundle = build_model(cfg.model, mode='train')
        load_model_weights(model_bundle.model, cfg.pretrained, device, strict=True)

        if cfg.pruning_method in {'l1', 'slim'}:
            pruner = BaselinePruner(model_bundle.model, prunable_bundle, cfg, method=str(cfg.pruning_method))
            plan_result = pruner.plan(point_value)
        elif cfg.pruning_method == 'depgraph':
            pruner = DepGraphGlobalPruner(model_bundle.model, prunable_bundle, cfg)
            plan_result = pruner.plan(point_value)
        else:
            variant = 'ours_l1' if cfg.pruning_method == 'ours_l1' else 'ours_slim'
            pruner = OursGlobalBudgetPruner(model_bundle.model, prunable_bundle, cfg, ocp_mapping, variant=variant)
            plan_result = pruner.plan(point_value)

        export_pruning_result(plan_result, str(point_dir), 'pruning_plan')
        _plot_point_details(plan_result, point_dir)

        mask_applier = ChannelMaskApplier(model_bundle.model, prunable_bundle, default_input_size=(1, 1, int(cfg.patch_size), int(cfg.patch_size)))
        mask_applier.set_masks_from_plans(
            plan_result['plans'],
            device=device,
            input_size=(1, 1, int(cfg.patch_size), int(cfg.patch_size)),
        )
        save_json(mask_applier.export_mask_dict(), str(point_dir / 'channel_masks.json'))

        model_bundle.model = model_bundle.model.to(device)
        mask_applier.rebind(model_bundle.model)
        metric_before = evaluate_model(model_bundle, val_loader, device, threshold=float(cfg.threshold), save_predictions=False)
        effective_stats = masked_model_effective_stats(model_bundle.model, mask_applier, (1, 1, int(cfg.patch_size), int(cfg.patch_size)), device)
        save_json(effective_stats, str(point_dir / 'effective_model_stats.json'))
        save_simple_csv(effective_stats.get('layer_rows', []), str(point_dir / 'effective_model_stats.csv'))
        physical_profile = profile_model_safe(
            model_bundle.model,
            (1, 1, int(cfg.patch_size), int(cfg.patch_size)),
            device,
        )
        save_json(physical_profile, str(point_dir / 'physical_model_profile.json'))

        finetune_summary = None
        if bool(cfg.plan_only):
            point_logger.info(
                'Plan-only screening requested: physical pruning, '
                'pre-finetune evaluation, and profiling are complete; '
                'finetuning is skipped.'
            )
        else:
            finetune_cfg = _build_finetune_cfg(cfg)
            finetune_summary = fit_model(
                model_bundle=model_bundle,
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=finetune_cfg,
                run_dir=str(point_dir / 'finetune'),
                logger=point_logger,
                ocp_layer_names=None,
                fixed_mask_applier=mask_applier,
                for_finetune=True,
            )

        retention_summary = _weighted_high_q_retention(plan_result)

        finetune_fields = _finetune_result_fields(
            plan_only=bool(cfg.plan_only),
            finetune_summary=finetune_summary,
            dense_metric=dense_metric,
            collapse_ratio_threshold=float(cfg.collapse_ratio_threshold),
        )
        row = {
            'method': method_key,
            'plan_only': bool(cfg.plan_only),
            'completion_status': 'plan_only' if bool(cfg.plan_only) else 'finetuned',
            'prune_rate': point_value if cfg.pruning_method in BASELINE_METHODS else None,
            'global_prune_ratio': point_value if cfg.pruning_method not in BASELINE_METHODS else None,
            'metric_before_finetune': float(metric_before[cfg.collapse_metric]),
            'dense_metric': dense_metric,
            'params_before': dense_params,
            'flops_before': dense_flops,
            'params_after': physical_profile.get('params')
            if physical_profile.get('params') is not None
            else effective_stats.get('effective_params_approx'),
            'flops_after': physical_profile.get('flops')
            if physical_profile.get('flops') is not None
            else effective_stats.get('effective_flops_approx'),
            'physical_model_profile': physical_profile,
            'effective_model_stats': {
                'params': effective_stats.get('effective_params_approx'),
                'flops': effective_stats.get('effective_flops_approx'),
            },
            'high_q_retention': retention_summary.get('value'),
            'high_q_retention_summary': retention_summary,
        }
        row.update(finetune_fields)
        if cfg.pruning_method not in BASELINE_METHODS:
            row.update({
                'global_candidate_channels': plan_result.get('global_candidate_channels'),
                'global_target_delete': plan_result.get('global_target_delete_after_cap'),
                'global_actual_delete': plan_result.get('global_actual_delete'),
            })
        summary_rows.append(row)
        save_json(row, str(point_dir / 'result_summary.json'))
        point_logger.log_record(row, prefix='point_summary')
        if bool(cfg.plan_only):
            point_logger.info(
                f'Finished plan-only point={point_tag}: '
                f'before={row["metric_before_finetune"]:.6f}, '
                f'params_after={row["params_after"]}, flops_after={row["flops_after"]}'
            )
        else:
            point_logger.info(
                f'Finished point={point_tag}: '
                f'before={row["metric_before_finetune"]:.6f}, '
                f'best_after_ft={row["best_metric_after_finetune"]:.6f}, '
                f'collapse={row["collapse_flag"]}'
            )

    x_key = 'prune_rate' if cfg.pruning_method in BASELINE_METHODS else 'global_prune_ratio'
    save_simple_csv(summary_rows, str(Path(sweep_dir) / 'collapse_scan_summary.csv'))
    save_json(summary_rows, str(Path(sweep_dir) / 'collapse_scan_summary.json'))
    if bool(cfg.plan_only):
        logger.info(
            'Plan-only sweep: skipped post-finetune metric plots because '
            'best_metric_after_finetune is intentionally null.'
        )
    else:
        plot_pruning_summary(summary_rows, sweep_dir, method_name=method_key, x_key=x_key)
        plot_pruning_compare({method_key: summary_rows}, str(Path(sweep_dir) / 'pruning_compare_all.png'))
    logger.info(f'Sweep finished. Summary rows={len(summary_rows)} collapse_scan_summary.csv saved to {sweep_dir}')
