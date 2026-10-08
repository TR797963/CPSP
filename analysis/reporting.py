from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from core.utils import plot_bar, plot_curve, plot_hist, save_json, save_simple_csv


def save_training_history(history: Sequence[Dict[str, Any]], run_dir: str) -> None:
    save_simple_csv(history, str(Path(run_dir) / 'train_val_history.csv'))
    save_json(list(history), str(Path(run_dir) / 'train_val_history.json'))


def plot_training_history(history: Sequence[Dict[str, Any]], run_dir: str, metric_name: str = 'mIoU') -> None:
    epochs = [int(x['epoch']) for x in history if 'epoch' in x]
    if not epochs:
        return
    train_total_loss = [float(x.get('train.total_loss', x.get('train_total_loss', 0.0))) for x in history]
    val_metric = [float(x.get(f'val.{metric_name}', x.get(metric_name, 0.0))) for x in history]
    plot_curve({'train_total_loss': train_total_loss}, str(Path(run_dir) / 'train_loss_curve.png'), 'Train Loss', 'Epoch', 'Loss')
    plot_curve({f'val_{metric_name}': val_metric}, str(Path(run_dir) / 'val_metric_curve.png'), f'Validation {metric_name}', 'Epoch', metric_name)

    if any('train.pol_loss' in x or 'train_pol_loss' in x for x in history):
        plot_curve(
            {
                'det_loss': [float(x.get('train.det_loss', x.get('train_det_loss', 0.0))) for x in history],
                'pol_loss': [float(x.get('train.pol_loss', x.get('train_pol_loss', 0.0))) for x in history],
                'pre_loss': [float(x.get('train.pre_loss', x.get('train_pre_loss', 0.0))) for x in history],
            },
            str(Path(run_dir) / 'ocp_loss_curve.png'),
            'OCP Loss Terms',
            'Epoch',
            'Loss',
        )
        plot_curve(
            {
                'avg_q': [float(x.get('train.avg_q', x.get('train_avg_q', 0.0))) for x in history],
                'avg_P': [float(x.get('train.avg_P', x.get('train_avg_P', 0.0))) for x in history],
                'avg_E_ent': [float(x.get('train.avg_E_ent', x.get('train_avg_E_ent', 0.0))) for x in history],
            },
            str(Path(run_dir) / 'ocp_stats_curve.png'),
            'OCP Statistics',
            'Epoch',
            'Value',
        )


def plot_polarization_stats(stats: Dict[str, Any], out_dir: str, prefix: str = '') -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    q_values: List[float] = []
    p_values: List[float] = []
    ent_values: List[float] = []
    layer_names: List[str] = []
    mean_q: List[float] = []

    for layer_name, layer_stat in stats.get('layers', {}).items():
        layer_names.append(layer_name)
        mean_q.append(float(layer_stat.get('mean_q', 0.0)))
        p_values.append(float(layer_stat.get('mean_P', 0.0)))
        ent_values.append(float(layer_stat.get('mean_E_ent', 0.0)))
        q_values.extend([float(v) for v in layer_stat.get('q_values', [])])

    if q_values:
        plot_hist(q_values, str(out / f'{prefix}q_hist.png'), 'q distribution', 'q', bins=20)
    if layer_names:
        plot_bar(layer_names, mean_q, str(out / f'{prefix}per_layer_q_bar.png'), 'Per-layer mean q', 'Layer', 'mean q')
        plot_bar(layer_names, p_values, str(out / f'{prefix}polarization_comparison.png'), 'Per-layer mean P', 'Layer', 'P')
        plot_bar(layer_names, ent_values, str(out / f'{prefix}entanglement_comparison.png'), 'Per-layer mean E_ent', 'Layer', 'E_ent')
    save_simple_csv(stats.get('layer_rows', []), str(out / f'{prefix}polarization_stats.csv'))
    save_json(stats, str(out / f'{prefix}polarization_stats.json'))


def plot_pruning_summary(summary_rows: Sequence[Dict[str, Any]], out_dir: str, method_name: str, x_key: str) -> None:
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_simple_csv(summary_rows, str(out / f'{method_name}_summary.csv'))
    save_json(list(summary_rows), str(out / f'{method_name}_summary.json'))

    x = [float(row[x_key]) for row in summary_rows]
    y = [float(row['best_metric_after_finetune']) for row in summary_rows]
    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker='o')
    plt.title(f'Pruning curve ({method_name})')
    plt.xlabel(x_key)
    plt.ylabel('Best metric after finetune')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(str(out / f'pruning_curve_{method_name}.png'), dpi=200)
    plt.close()

    if summary_rows and 'high_q_retention' in summary_rows[0]:
        y2 = [float(row.get('high_q_retention', 0.0)) if row.get('high_q_retention') is not None else 0.0 for row in summary_rows]
        plt.figure(figsize=(7, 5))
        plt.plot(x, y2, marker='o')
        plt.title('High-q retention')
        plt.xlabel(x_key)
        plt.ylabel('Retention')
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(str(out / 'high_q_retention.png'), dpi=200)
        plt.close()


def plot_pruning_compare(method_to_rows: Dict[str, Sequence[Dict[str, Any]]], out_path: str, x_key_map: Optional[Dict[str, str]] = None) -> None:
    series: Dict[str, List[float]] = {}
    for method, rows in method_to_rows.items():
        series[method] = [float(r['best_metric_after_finetune']) for r in rows]
    plot_curve(series, out_path, 'Pruning comparison', 'Point index', 'Best metric after finetune')
