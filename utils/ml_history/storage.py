from pathlib import Path
import os
import json
import csv
from typing import TYPE_CHECKING, Optional, List, Dict, Any, Union
from statistics import mean, stdev
import pandas as pd
import matplotlib.pyplot as plt

from configs.globals import CATEGORIES_MAP

if TYPE_CHECKING:
    from utils.ml_history.metrics import MLHistory


def _map_category(category_key: Union[int, float, str]) -> str:
    if isinstance(category_key, (int, float)):
        return CATEGORIES_MAP.get(int(category_key), f"Unknown-{category_key}")
    if isinstance(category_key, str):
        try:
            num = float(category_key)
            return CATEGORIES_MAP.get(int(num), f"Unknown-{category_key}")
        except ValueError:
            return category_key
    return str(category_key)


class HistoryStorage:
    """
    A class to store and retrieve training history.
    """

    def __init__(self, history: 'MLHistory'):
        self.h = history

    def _ensure_dir(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _dump_json(self, obj: Any, filepath: Union[str, Path]) -> None:
        with open(filepath, 'w') as f:
            json.dump(obj, f, indent=2)

    def _save_csv(self, df: pd.DataFrame, filepath: Union[str, Path], **kwargs) -> None:
        df.to_csv(filepath, index=kwargs.pop('index', False), **kwargs)

    def _build_folder_name(self, extras: Optional[List[str]] = None) -> str:
        parts = [
            self.h.model_name,
            f"lr{self.h.model_parms.learning_rate:.1e}".replace(".0", ""),
            f"bs{self.h.model_parms.batch_size}",
            f"ep{self.h.model_parms.num_epochs}",
            self.h.start_date,
        ]
        if extras:
            parts.extend(extras)
        return "_".join(parts).lower()

    def _save_params(self, params_dir: Path) -> None:
        # Collect parameters
        params = {
            'model_name': self.h.model_name,
            'model_type': self.h.model_type,
            'num_folds': self.h.num_folds,
            'tasks': ','.join(self.h.tasks),
            'start_date': self.h.start_date,
            'end_date': self.h.end_date,
            'total_time': self.h.timer.get_duration() if self.h.timer.duration else 0
        }
        # Add hyperparameters
        for attr in dir(self.h.model_parms):
            if attr.startswith('_'):
                continue
            val = getattr(self.h.model_parms, attr)
            if callable(val):
                continue
            try:
                if hasattr(val, 'tolist'):
                    params[attr] = json.dumps(val.tolist())
                elif isinstance(val, dict) or hasattr(val, 'items'):
                    params[attr] = json.dumps(dict(val))
                elif isinstance(val, (list, tuple)):
                    params[attr] = json.dumps(list(val))
                else:
                    params[attr] = val
            except Exception:
                params[attr] = str(val)

        df = pd.DataFrame(params.items(), columns=['Parameter', 'Value'])
        self._save_csv(df, params_dir / 'model_params.csv')

    def _save_metrics(self, metrics_dir: Path) -> None:
        for idx, fold in enumerate(self.h.folds, start=1):
            # Per-task metrics
            for task in self.h.tasks:
                th = fold.tasks_history.get(task)
                if not th:
                    continue
                tm = th.task_metrics
                # Training
                df_train = pd.DataFrame({
                    'epoch': list(range(1, len(tm.loss) + 1)),
                    'loss': tm.loss,
                    'accuracy': tm.accuracy,
                    'f1': tm.f1
                })
                self._save_csv(df_train, metrics_dir / f'fold{idx}_{task}_train.csv')
                # Validation
                df_val = pd.DataFrame({
                    'epoch': list(range(1, len(tm.val_loss) + 1)),
                    'val_loss': tm.val_loss,
                    'val_accuracy': tm.val_accuracy,
                    'val_f1': tm.val_f1
                })
                self._save_csv(df_val, metrics_dir / f'fold{idx}_{task}_val.csv')
            # Model-wide metrics
            if getattr(fold, 'model', None) and getattr(fold.model, 'task_metrics', None):
                mm = fold.model.task_metrics
                df_model = pd.DataFrame({
                    'epoch': list(range(1, len(mm.loss) + 1)),
                    'loss': mm.loss,
                    'accuracy': mm.accuracy,
                    'f1': mm.f1,
                    'val_loss': mm.val_loss,
                    'val_accuracy': mm.val_accuracy,
                    'val_f1': mm.val_f1
                })
                self._save_csv(df_model, metrics_dir / f'fold{idx}_model.csv')

    def _save_fold_reports(self, folds_dir: Path) -> None:
        for idx, fold in enumerate(self.h.folds, start=1):
            fold_info: Dict[str, Any] = {
                'fold_number': idx,
                'duration': fold.timer.get_duration() if fold.timer.duration else 0,
                'best_epochs': {}
            }
            for task in self.h.tasks:
                th = fold.tasks_history.get(task)
                if not th or not th.task_outcome.report:
                    continue
                # Map categories
                mapped = {}
                for cat, metrics in th.task_outcome.report.items():
                    key = _map_category(cat) if isinstance(metrics, dict) else cat
                    mapped[key] = metrics
                # JSON report
                self._dump_json(mapped, folds_dir / f'fold{idx}_{task}_report.json')
                # CSV report
                rows = []
                for cat, met in mapped.items():
                    if not isinstance(met, dict):
                        continue
                    rows.append({
                        'Category': cat,
                        'Precision': met.get('precision', ''),
                        'Recall': met.get('recall', ''),
                        'F1-Score': met.get('f1-score', ''),
                        'Support': met.get('support', '')
                    })
                if rows:
                    df = pd.DataFrame(rows)
                    self._save_csv(df, folds_dir / f'fold{idx}_{task}_report.csv', float_format='%.4f')
                # Best epoch
                be = th.best_epoch
                acc = th.task_metrics.val_accuracy[be] if be < len(th.task_metrics.val_accuracy) else None
                f1 = th.task_metrics.val_f1[be] if be < len(th.task_metrics.val_f1) else None
                fold_info['best_epochs'][task] = {'epoch': be, 'accuracy': acc, 'f1': f1}
            self._dump_json(fold_info, folds_dir / f'fold{idx}_info.json')

    def _create_summary(self, summary_dir: Path) -> None:
        summary_dir.mkdir(parents=True, exist_ok=True)
        # Initialize accumulators
        perf: Dict[str, Dict[str, List[float]]] = {
            task: {'accuracy': [], 'f1': []} for task in self.h.tasks
        }
        cat_metrics: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        for fold in self.h.folds:
            for task in self.h.tasks:
                th = fold.tasks_history.get(task)
                if not th:
                    continue
                be = th.best_epoch
                if be < len(th.task_metrics.val_accuracy):
                    perf[task]['accuracy'].append(th.task_metrics.val_accuracy[be])
                if be < len(th.task_metrics.val_f1):
                    perf[task]['f1'].append(th.task_metrics.val_f1[be])
                # Category-level
                report = getattr(th.task_outcome, 'report', {}) or {}
                for cat, metrics in report.items():
                    if not isinstance(metrics, dict) or cat in ['accuracy', 'macro avg', 'weighted avg']:
                        continue
                    name = _map_category(cat)
                    task_cat = cat_metrics.setdefault(task, {}).setdefault(name, {
                        'precision': [], 'recall': [], 'f1-score': [], 'support': []
                    })
                    for m in ['precision', 'recall', 'f1-score', 'support']:
                        if m in metrics:
                            task_cat[m].append(metrics[m])
        # Save performance summary
        rows = []
        for task, vals in perf.items():
            for metric in ['accuracy', 'f1']:
                arr = vals[metric]
                rows.append([task, metric.capitalize(),
                             mean(arr) if arr else 0,
                             stdev(arr) if len(arr) > 1 else 0,
                             min(arr) if arr else 0,
                             max(arr) if arr else 0])
        df_perf = pd.DataFrame(rows, columns=['Task', 'Metric', 'Mean', 'Std', 'Min', 'Max'])
        self._save_csv(df_perf, summary_dir / 'performance_summary.csv', float_format='%.4f')
        # Full JSON
        summary_stats = {
            task: {
                metric: {
                    'mean': mean(vals[metric]) if vals[metric] else 0,
                    'std': stdev(vals[metric]) if len(vals[metric]) > 1 else 0,
                    'min': min(vals[metric]) if vals[metric] else 0,
                    'max': max(vals[metric]) if vals[metric] else 0
                } for metric in ['accuracy', 'f1']
            } for task, vals in perf.items()
        }
        self._dump_json(summary_stats, summary_dir / 'full_summary.json')
        # Category summaries
        for task, cats in cat_metrics.items():
            rows = []
            macro = {'precision': [], 'recall': [], 'f1-score': [], 'support': []}
            for name, mets in cats.items():
                mean_vals = {m: mean(mets[m]) if mets[m] else 0 for m in macro}
                for m in macro:
                    macro[m].append(mean_vals[m])
                rows.append({
                    'Category': name,
                    'Precision': mean_vals['precision'],
                    'Recall': mean_vals['recall'],
                    'F1-Score': mean_vals['f1-score'],
                    'Support': mean_vals['support']
                })
            if rows:
                rows.append({
                    'Category': 'macro avg',
                    'Precision': mean(macro['precision']),
                    'Recall': mean(macro['recall']),
                    'F1-Score': mean(macro['f1-score']),
                    'Support': ''
                })
                df_cat = pd.DataFrame(rows)
                self._save_csv(df_cat, summary_dir / f'summary_{task}_metrics.csv', float_format='%.4f')

    def _generate_plots(self, plots_dir: Path) -> None:
        for task in self.h.tasks:
            for metric in ['loss', 'accuracy', 'f1']:
                plt.figure(figsize=(12, 6))
                for idx, fold in enumerate(self.h.folds, start=1):
                    th = fold.tasks_history.get(task)
                    if not th:
                        continue
                    tm = th.task_metrics
                    epochs = range(1, len(getattr(tm, metric)) + 1)
                    plt.plot(epochs, getattr(tm, metric), label=f'Fold {idx} Train {metric.capitalize()}')
                    plt.plot(epochs, getattr(tm, f'val_{metric}'),
                             label=f'Fold {idx} Val {metric.capitalize()}', linestyle='--')
                plt.title(f'Training and Validation {metric.capitalize()} for Task: {task}')
                plt.xlabel('Epoch')
                plt.ylabel(metric.replace('_', ' ').capitalize())
                plt.legend()
                plt.grid(True)
                os.makedirs(plots_dir / task, exist_ok=True)
                plt.savefig(plots_dir / task / f"{metric}.png")
                plt.close()
        for metric in ['loss', 'accuracy', 'f1']:
            plt.figure(figsize=(12, 6))
            for idx, fold in enumerate(self.h.folds, start=1):
                th = fold.model
                if not th:
                    continue
                tm = th.task_metrics
                epochs = range(1, len(getattr(tm, metric)) + 1)
                plt.plot(epochs, getattr(tm, metric), label=f'Fold {idx} Train {metric.capitalize()}')
                plt.plot(epochs, getattr(tm, f'val_{metric}'),
                         label=f'Fold {idx} Val {metric.capitalize()}', linestyle='--')
            plt.title(f'Training and Validation {metric.capitalize()} for Model')
            plt.xlabel('Epoch')
            plt.ylabel(metric.replace('_', ' ').capitalize())
            plt.legend()
            plt.grid(True)
            os.makedirs(plots_dir / 'model', exist_ok=True)
            plt.savefig(plots_dir / 'model' / f"{metric}.png")
            plt.close()

    def save(self, path: str, extra_names: Optional[List[str]] = None) -> str:
        folder_name = self._build_folder_name(extra_names)
        root = self._ensure_dir(Path(path) / folder_name)

        dirs = {
            'params': root / 'params',
            'metrics': root / 'metrics',
            'folds': root / 'folds',
            'summary': root / 'summary',
            'plots': root / 'plots'
        }
        for d in dirs.values():
            self._ensure_dir(d)

        self._save_params(dirs['params'])
        self._save_metrics(dirs['metrics'])
        self._save_fold_reports(dirs['folds'])
        self._create_summary(dirs['summary'])
        self._generate_plots(dirs['plots'])

        return str(root)
