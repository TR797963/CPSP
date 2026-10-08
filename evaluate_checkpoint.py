import json

from core.config import build_run_dir, dump_effective_config, parse_args
from core.data import build_dataloaders
from core.modeling import build_model
from core.trainer import evaluate_checkpoint
from core.utils import ExperimentLogger, seed_everything
from pruning.auto_select import LayerMeta, SelectionBundle
from pruning.masked_pruning import ChannelMaskApplier


def load_selection_bundle(path: str) -> SelectionBundle:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return SelectionBundle(
        all_layers=[LayerMeta(**x) for x in data['all']],
        selected_layers=[LayerMeta(**x) for x in data['selected']],
        skipped_layers=[LayerMeta(**x) for x in data['skipped']],
    )


def maybe_rebuild_pruned_model(model_bundle, cfg, logger):
    if not cfg.pruning_bundle_json or not cfg.pruning_plan_json:
        logger.info('No pruning metadata provided. Evaluate as dense/original model.')
        return model_bundle

    bundle = load_selection_bundle(cfg.pruning_bundle_json)
    with open(cfg.pruning_plan_json, 'r', encoding='utf-8') as f:
        plan_result = json.load(f)

    # 注意：这里 patch_size 要和你训练/剪枝时一致
    input_size = (1, 1, int(cfg.patch_size), int(cfg.patch_size))

    applier = ChannelMaskApplier(
        model_bundle.model,
        bundle,
        default_input_size=input_size,
    )
    applier.set_masks_from_plans(
        plan_result['plans'],
        device='cpu',          # 这里只是为了重建结构，先在 CPU 上做即可
        input_size=input_size,
    )
    logger.info(
        f'Pruned model structure rebuilt from plan: {cfg.pruning_plan_json}'
    )
    return model_bundle


if __name__ == '__main__':
    cfg = parse_args()
    assert cfg.pretrained is not None, '--pretrained checkpoint is required for evaluation.'
    seed_everything(int(cfg.seed))
    run_dir = build_run_dir(cfg, subdir='evaluation')
    logger = ExperimentLogger(run_dir)
    dump_effective_config(cfg, run_dir)

    _, val_loader = build_dataloaders(cfg)
    model_bundle = build_model(cfg.model, mode='test')
    model_bundle = maybe_rebuild_pruned_model(model_bundle, cfg, logger)

    result = evaluate_checkpoint(model_bundle, cfg.pretrained, val_loader, cfg, run_dir, logger)
    logger.info(f'Evaluation done. metrics={result["metrics"]}')