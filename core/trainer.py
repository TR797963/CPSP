from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import nn
from PIL import Image
import numpy as np

from analysis.reporting import plot_training_history, save_training_history
from core.checkpoint import BestCheckpointTracker, load_training_states, save_checkpoint
from core.metrics_ext import PDFAAccumulator, SegmentationMeter, combine_metric_dicts
from core.modeling import ModelBundle, compute_detection_loss, extract_primary_prediction, load_model_weights, maybe_data_parallel, unwrap_model
from core.utils import AverageMeter, ExperimentLogger, plot_curve, profile_model_safe, measure_inference_speed, resolve_best_metric, save_json, select_device, to_device
from ocp.ocp_core import OCPComputer


class SchedulerStub:
    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, _state: Dict[str, Any]) -> None:
        return

    def step(self) -> None:
        return


def checkpoint_selection_eligible(
    epoch_number: int,
    *,
    ocp_enabled: bool,
    ocp_warmup_epochs: int,
) -> bool:
    """Only rank CP checkpoints after at least one CP-active epoch.

    ``epoch_number`` is one-based, while :class:`OCPComputer` enables its
    auxiliary losses when the zero-based epoch reaches the warm-up length.
    The first eligible CP checkpoint is therefore ``warmup + 1``.
    """

    if not ocp_enabled:
        return True
    return int(epoch_number) > max(int(ocp_warmup_epochs), 0)


def compute_bn_sparsity_loss(model: nn.Module) -> torch.Tensor:
    """Network-Slimming regularizer over trainable BN scale parameters."""
    gammas = [
        module.weight.reshape(-1)
        for module in unwrap_model(model).modules()
        if isinstance(module, nn.BatchNorm2d) and module.affine and module.weight is not None
    ]
    if not gammas:
        parameter = next(model.parameters())
        return parameter.sum() * 0.0
    return torch.cat(gammas).abs().mean()



def build_optimizer_and_scheduler(model: nn.Module, cfg: Any, for_finetune: bool = False) -> Tuple[torch.optim.Optimizer, Any]:
    lr = float(cfg.finetune_lr if for_finetune else cfg.lr)
    if cfg.optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif cfg.optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    elif cfg.optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f'Unsupported optimizer {cfg.optimizer_name}')

    if cfg.scheduler_name == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(cfg.scheduler_milestones), gamma=float(cfg.scheduler_gamma))
    elif cfg.scheduler_name == 'CosineAnnealingLR':
        t_max = int(cfg.finetune_epochs if for_finetune else cfg.epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(t_max, 1), eta_min=float(cfg.scheduler_min_lr))
    else:
        scheduler = SchedulerStub()
    return optimizer, scheduler


@torch.no_grad()
def evaluate_model(
    model_bundle: ModelBundle,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    save_predictions: bool = False,
    predictions_dir: Optional[str] = None,
) -> Dict[str, float]:
    model = model_bundle.model
    model.eval()
    seg_meter = SegmentationMeter(threshold=threshold)
    pdfa_meter = PDFAAccumulator(match_distance=3.0)

    for batch in data_loader:
        img, mask, size, name = batch
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        raw = model(img)
        pred = extract_primary_prediction(raw, model_name=model_bundle.model_name)
        h, w = int(size[0][0].item()), int(size[0][1].item())
        pred = pred[:, :, :h, :w]
        mask = mask[:, :, :h, :w]
        seg_meter.update(pred, mask)
        pdfa_meter.update((pred[0, 0] > threshold).float(), mask[0, 0], (h, w))

        if save_predictions and predictions_dir is not None:
            Path(predictions_dir).mkdir(parents=True, exist_ok=True)
            arr = pred[0, 0].detach().cpu().clamp(0.0, 1.0).numpy()
            img_save = Image.fromarray((arr * 255.0).astype(np.uint8))
            img_save.save(str(Path(predictions_dir) / f'{name[0]}.png'))

    return combine_metric_dicts(seg_meter.get(), pdfa_meter.get())



def _maybe_load_checkpoint(model_bundle: ModelBundle, optimizer: Optional[torch.optim.Optimizer], scheduler: Optional[Any], cfg: Any, device: torch.device, logger: ExperimentLogger) -> Tuple[int, BestCheckpointTracker]:
    tracker = BestCheckpointTracker(metric_name=str(cfg.save_best_metric), maximize=True)
    start_epoch = 0
    if cfg.pretrained:
        load_info = load_model_weights(unwrap_model(model_bundle.model), cfg.pretrained, device, strict=True)
        logger.info(f'Loaded pretrained from {cfg.pretrained}; missing={len(load_info["missing_keys"])} unexpected={len(load_info["unexpected_keys"])}')
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device)
        state_dict = ckpt.get('state_dict', ckpt)
        unwrap_model(model_bundle.model).load_state_dict(state_dict, strict=True)
        start_epoch, _ = load_training_states(ckpt, optimizer, scheduler)
        if 'best_metric' in ckpt:
            tracker.best_value = float(ckpt['best_metric'])
            tracker.best_epoch = int(ckpt.get('best_epoch', -1))
            tracker.best_path = ckpt.get('best_path')
        logger.info(f'Resumed from {cfg.resume} at epoch={start_epoch}.')
    return start_epoch, tracker



def fit_model(
    model_bundle: ModelBundle,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    cfg: Any,
    run_dir: str,
    logger: ExperimentLogger,
    ocp_layer_names: Optional[Sequence[str]] = None,
    fixed_mask_applier: Optional[Any] = None,
    for_finetune: bool = False,
) -> Dict[str, Any]:
    device = select_device(cfg.device)
    model_bundle.criterion = model_bundle.criterion.to(device)
    model_bundle.model = model_bundle.model.to(device)
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        model_bundle.model = maybe_data_parallel(model_bundle.model)
    if fixed_mask_applier is not None and getattr(fixed_mask_applier, 'masks', None):
        fixed_mask_applier.rebind(model_bundle.model)
        logger.info(f'Rebound fixed pruning masks on {len(fixed_mask_applier.masks)} layers before training/finetuning.')

    optimizer, scheduler = build_optimizer_and_scheduler(model_bundle.model, cfg, for_finetune=for_finetune)
    start_epoch, tracker = _maybe_load_checkpoint(model_bundle, optimizer, scheduler, cfg, device, logger)

    ocp_engine = None
    if bool(cfg.ocp_enable) and ocp_layer_names:
        ocp_engine = OCPComputer(model_bundle.model, ocp_layer_names, cfg)
        logger.info(f'OCP enabled on layers={list(ocp_layer_names)}')

    history = []
    global_step = 0
    total_epochs = int(cfg.finetune_epochs if for_finetune else cfg.epochs)

    for epoch in range(start_epoch, total_epochs):
        model_bundle.model.train()
        det_meter = AverageMeter()
        pol_meter = AverageMeter()
        pre_meter = AverageMeter()
        slim_meter = AverageMeter()
        total_meter = AverageMeter()
        q_meter = AverageMeter()
        p_meter = AverageMeter()
        e_meter = AverageMeter()

        for batch in train_loader:
            global_step += 1
            if len(batch) == 4:
                img, mask, _, _ = batch
            else:
                img, mask = batch
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            if ocp_engine is not None:
                ocp_engine.clear()
            raw = model_bundle.model(img)
            det_loss = compute_detection_loss(model_bundle, raw, mask)
            ocp_result = ocp_engine.compute_batch(mask, epoch, global_step) if ocp_engine is not None else None
            pol_loss = ocp_result.pol_loss if ocp_result is not None else det_loss * 0.0
            pre_loss = ocp_result.pre_loss if ocp_result is not None else det_loss * 0.0
            extra_loss = ocp_result.total_extra_loss if ocp_result is not None else det_loss * 0.0
            slim_loss = (
                compute_bn_sparsity_loss(model_bundle.model)
                if bool(getattr(cfg, 'slim_regularization_enable', False))
                else det_loss * 0.0
            )
            total_loss = det_loss + extra_loss + float(getattr(cfg, 'slim_lambda', 0.0)) * slim_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            bs = int(img.shape[0])
            det_meter.update(float(det_loss.detach().cpu().item()), bs)
            pol_meter.update(float(pol_loss.detach().cpu().item()), bs)
            pre_meter.update(float(pre_loss.detach().cpu().item()), bs)
            slim_meter.update(float(slim_loss.detach().cpu().item()), bs)
            total_meter.update(float(total_loss.detach().cpu().item()), bs)
            if ocp_result is not None:
                q_meter.update(float(ocp_result.avg_q), max(ocp_result.valid_samples, 1))
                p_meter.update(float(ocp_result.avg_P), max(ocp_result.valid_samples, 1))
                e_meter.update(float(ocp_result.avg_E_ent), max(ocp_result.valid_samples, 1))

        scheduler.step()

        row: Dict[str, Any] = {
            'epoch': epoch + 1,
            'lr': float(optimizer.param_groups[0]['lr']),
            'train': {
                'det_loss': det_meter.avg,
                'pol_loss': pol_meter.avg,
                'pre_loss': pre_meter.avg,
                'slim_loss': slim_meter.avg,
                'total_loss': total_meter.avg,
                'avg_q': q_meter.avg,
                'avg_P': p_meter.avg,
                'avg_E_ent': e_meter.avg,
            },
        }

        best_refreshed = False
        val_metrics: Dict[str, float] = {}
        if (epoch + 1) % max(int(cfg.val_every), 1) == 0:
            val_metrics = evaluate_model(
                model_bundle,
                val_loader,
                device,
                threshold=float(cfg.threshold),
                save_predictions=bool(cfg.save_predictions and epoch + 1 == total_epochs),
                predictions_dir=str(Path(run_dir) / 'predictions') if bool(cfg.save_predictions) else None,
            )
            row['val'] = val_metrics
            current_metric = resolve_best_metric(val_metrics, str(cfg.save_best_metric))
            best_path = str(Path(run_dir) / 'best.pth')
            checkpoint_eligible = checkpoint_selection_eligible(
                epoch + 1,
                ocp_enabled=ocp_engine is not None,
                ocp_warmup_epochs=int(getattr(cfg, 'ocp_warmup_epochs', 0)),
            )
            row['checkpoint_eligible'] = bool(checkpoint_eligible)
            best_refreshed = checkpoint_eligible and tracker.is_better(current_metric)
            if best_refreshed:
                save_checkpoint(
                    model_bundle.model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    run_dir,
                    'best.pth',
                    extra={
                        'best_metric': current_metric,
                        'best_epoch': epoch + 1,
                        'best_path': best_path,
                        'checkpoint_selection_eligible': True,
                        'ocp_warmup_epochs': int(getattr(cfg, 'ocp_warmup_epochs', 0)),
                        'history_tail': history[-5:] if len(history) > 5 else history,
                    },
                )
                tracker.update(current_metric, epoch + 1, best_path)

        last_path = save_checkpoint(
            model_bundle.model,
            optimizer,
            scheduler,
            epoch + 1,
            run_dir,
            'last.pth',
            extra={
                'best_metric': tracker.best_value,
                'best_epoch': tracker.best_epoch,
                'best_path': tracker.best_path,
            },
        )
        if bool(cfg.save_every_epoch_ckpt):
            save_checkpoint(model_bundle.model, optimizer, scheduler, epoch + 1, run_dir, f'epoch_{epoch + 1:03d}.pth')

        row['best'] = {
            'refreshed': bool(best_refreshed),
            'best_metric': float(tracker.best_value),
            'best_epoch': int(tracker.best_epoch),
            'best_path': tracker.best_path,
            'last_path': last_path,
        }
        history.append(row)
        logger.log_record(row, prefix='epoch')
        logger.info(
            f"Epoch [{epoch + 1}/{total_epochs}] det={det_meter.avg:.6f} pol={pol_meter.avg:.6f} pre={pre_meter.avg:.6f} slim={slim_meter.avg:.6f} total={total_meter.avg:.6f} "
            + (f"mIoU={val_metrics.get('mIoU', 0.0):.6f} pixAcc={val_metrics.get('pixAcc', 0.0):.6f} PD={val_metrics.get('PD', 0.0):.6f} FA={val_metrics.get('FA', 0.0):.6f} " if val_metrics else '')
            + f"checkpoint_eligible={row.get('checkpoint_eligible', True)} "
            + f"best_refresh={best_refreshed} best_epoch={tracker.best_epoch} best_{cfg.save_best_metric}={tracker.best_value:.6f}"
        )

    if ocp_engine is not None:
        ocp_engine.close()

    save_training_history(history, run_dir)
    plot_training_history(history, run_dir, metric_name=str(cfg.save_best_metric))
    summary = {
        'run_dir': run_dir,
        'best_metric_name': str(cfg.save_best_metric),
        'best_metric': float(tracker.best_value),
        'best_epoch': int(tracker.best_epoch),
        'best_path': tracker.best_path,
        'last_path': str(Path(run_dir) / 'last.pth'),
        'history': history,
    }
    save_json(summary, str(Path(run_dir) / 'summary.json'))
    return summary


@torch.no_grad()
def evaluate_checkpoint(
    model_bundle: ModelBundle,
    checkpoint_path: str,
    data_loader: torch.utils.data.DataLoader,
    cfg: Any,
    run_dir: str,
    logger: ExperimentLogger,
) -> Dict[str, Any]:
    device = select_device(cfg.device)
    model_bundle.model = model_bundle.model.to(device)
    load_info = load_model_weights(model_bundle.model, checkpoint_path, device, strict=True)
    if load_info['missing_keys'] or load_info['unexpected_keys']:
        logger.info(f'Checkpoint load diagnostics: missing={load_info["missing_keys"][:20]} unexpected={load_info["unexpected_keys"][:20]}')
    metrics = evaluate_model(model_bundle, data_loader, device, threshold=float(cfg.threshold), save_predictions=bool(cfg.save_predictions), predictions_dir=str(Path(run_dir) / 'predictions'))
    profile = profile_model_safe(model_bundle.model, (1, 1, int(cfg.patch_size), int(cfg.patch_size)), device)
    speed = measure_inference_speed(model_bundle.model, (1, 1, int(cfg.patch_size), int(cfg.patch_size)), device, repeats=int(cfg.inference_speed_repeats))
    result = {
        'checkpoint': checkpoint_path,
        'metrics': metrics,
        'profile': profile,
        'speed': speed,
    }
    save_json(result, str(Path(run_dir) / 'evaluation.json'))
    logger.log_record(result, prefix='evaluation')
    logger.info(f'Evaluation metrics={metrics} profile={profile} speed={speed}')
    return result
