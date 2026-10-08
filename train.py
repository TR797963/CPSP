from pathlib import Path

from analysis.reporting import plot_polarization_stats
from core.config import build_run_dir, dump_effective_config, parse_args
from core.data import build_analysis_loader, build_dataloaders
from core.modeling import build_model, load_model_weights, unwrap_model
from core.trainer import fit_model
from core.utils import ExperimentLogger, save_json, seed_everything, select_device
from ocp.ocp_core import build_ocp_layer_mapping, collect_ocp_statistics
from pruning.auto_select import auto_select_ocp_layers, auto_select_prunable_layers, build_selection_summary, pruning_dry_run_check, save_selection_bundle


if __name__ == '__main__':
    cfg = parse_args()
    cfg.ocp_enable = bool(cfg.mode == 'ocp_train' or cfg.ocp_enable)
    cfg.slim_regularization_enable = bool(
        cfg.mode == 'slim_train' or cfg.slim_regularization_enable
    )
    seed_everything(int(cfg.seed))

    run_dir = build_run_dir(cfg)
    logger = ExperimentLogger(run_dir)
    dump_effective_config(cfg, run_dir)
    logger.info(f'Starting mode={cfg.mode}, run_dir={run_dir}')

    train_loader, val_loader = build_dataloaders(cfg)
    analysis_loader = build_analysis_loader(cfg, subset_size=int(cfg.ocp_stat_subset))

    # Always save auto-selected candidate sets for reproducibility.
    dry_model_bundle = build_model(cfg.model, mode='train')
    dry_model = dry_model_bundle.model
    prunable_bundle = auto_select_prunable_layers(dry_model, cfg, method=str(cfg.pruning_method))
    save_selection_bundle(prunable_bundle, run_dir, 'auto_prunable_layers')
    save_json(build_selection_summary(prunable_bundle), str(Path(run_dir) / 'auto_prunable_layers.summary.json'))
    dry_run = pruning_dry_run_check(dry_model, prunable_bundle, method=str(cfg.pruning_method))
    save_json(dry_run, str(Path(run_dir) / 'pruning_plan_checked.json'))

    ocp_bundle = auto_select_ocp_layers(dry_model, cfg)
    save_selection_bundle(ocp_bundle, run_dir, 'auto_ocp_layers')
    save_json(build_selection_summary(ocp_bundle), str(Path(run_dir) / 'auto_ocp_layers.summary.json'))

    model_bundle = build_model(cfg.model, mode='train')
    summary = fit_model(
        model_bundle=model_bundle,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        run_dir=run_dir,
        logger=logger,
        ocp_layer_names=ocp_bundle.selected_names() if bool(cfg.ocp_enable) else None,
        fixed_mask_applier=None,
        for_finetune=False,
    )

    logger.info(f'Training complete. best_epoch={summary["best_epoch"]} best_metric={summary["best_metric"]:.6f}')

    if bool(cfg.ocp_enable):
        best_model_bundle = build_model(cfg.model, mode='test')
        device = select_device(cfg.device)
        best_model_bundle.model = best_model_bundle.model.to(device)
        load_model_weights(best_model_bundle.model, summary['best_path'], device, strict=True)
        stats = collect_ocp_statistics(best_model_bundle.model, ocp_bundle, analysis_loader, device, cfg, out_dir=str(Path(run_dir) / 'polarization'))
        mapping = build_ocp_layer_mapping(prunable_bundle, ocp_bundle, stats, strategy=str(cfg.ocp_mapping_strategy), out_dir=str(Path(run_dir) / 'polarization'))
        logger.info(f'OCP analysis complete. global_mean_q={stats["global"]["mean_q"]:.6f}, global_mean_P={stats["global"]["mean_P"]:.6f}, global_mean_E_ent={stats["global"]["mean_E_ent"]:.6f}')
