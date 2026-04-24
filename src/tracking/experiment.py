from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from tracking.fold import FoldHistory
from tracking.parms.neural import HyperParams
from tracking.utils.dataset import DatasetHistory
from tracking.utils.time_history import TimeHistory


class FlopsMetrics:
    """Holds model profiling metrics."""

    def __init__(self, flops, params):
        self.flops = flops
        self.params = params
        self.memory: List[float] = []
        self.inference_time: List[float] = []
        self.training_time: List[float] = []

    def to_dict(self) -> dict:
        return {
            'flops': self.flops,
            'params': self.params,
            'memory': self.memory,
            'inference_time': self.inference_time,
            'training_time': self.training_time,
        }


class MLHistory:
    """Top-level experiment tracker for cross-validated ML training.

    Manages folds, timing, model metadata, and delegates to
    HistoryDisplay and HistoryStorage for output.

    Usage:
        history = MLHistory(
            'MyModel',
            tasks={'next', 'category'},
            num_folds=5,
            label_map={0: 'Food', 1: 'Shop'},
            save_path='/results',
            verbose=True,
        )

        with history:
            for fold in folds:
                # training loop...
                history.step()
        # end_training() auto-fires, results auto-saved
    """

    def __init__(
        self,
        model_name: str,
        tasks: Set[str] | str,
        model_type: str = 'MTL',
        num_folds: int = 1,
        model_parms: Optional[HyperParams] = None,
        datasets: Optional[Set[DatasetHistory]] = None,
        monitor: str = 'f1',
        mode: str = 'max',
        label_map: Optional[Dict[int, str]] = None,
        save_path: Optional[Union[str, Path]] = None,
        verbose: bool = False,
        display_report: bool = False,
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.num_folds = num_folds
        self.model_parms: Optional[HyperParams] = model_parms
        self.model_arch: Optional[str] = None
        self.datasets: Optional[Set[DatasetHistory]] = datasets
        self.monitor = monitor
        self.mode = mode

        self.tasks: Set[str] = {tasks} if isinstance(tasks, str) else tasks
        self.folds: List[FoldHistory] = [
            FoldHistory(i, self.tasks, monitor=monitor, mode=mode)
            for i in range(num_folds)
        ]
        self.flops: Optional[FlopsMetrics] = None

        self.timer = TimeHistory()
        self.start_date: Optional[str] = None
        self.end_date: Optional[str] = None
        self.curr_i_fold: int = 0

        # Config for auto-lifecycle
        self._label_map = label_map
        self._save_path = Path(save_path) if save_path else None
        self._verbose = verbose or display_report
        self._display_report = display_report
        self._ended = False

        # Lazy init to avoid circular imports
        self._display = None
        self._storage = None
        self._adapter = None  # Optional external tracking (W&B, etc.)

    @property
    def display(self):
        if self._display is None:
            from tracking.display import HistoryDisplay
            self._display = HistoryDisplay(
                self,
                label_map=self._label_map,
                show_report=self._display_report,
            )
        return self._display

    @property
    def storage(self):
        if self._storage is None:
            from tracking.storage import HistoryStorage
            self._storage = HistoryStorage(self)
        return self._storage

    # --- Context manager ---

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()
        if exc_type is None and self._save_path is not None:
            run_path = self.storage.save(path=self._save_path, label_map=self._label_map)
            self._compare_and_save_records(run_path)

    # --- Iterator ---

    def __iter__(self):
        self.start()
        for fold in self.folds:
            yield fold
        self.end()

    # --- Fold management ---

    @property
    def fold(self) -> FoldHistory:
        """Shortcut for get_curr_fold()."""
        return self.folds[self.curr_i_fold]

    def get_curr_fold(self) -> FoldHistory:
        """Get the current fold."""
        return self.folds[self.curr_i_fold]

    def step(self):
        """End current fold and advance to next."""
        self.folds[self.curr_i_fold].end()
        if self._verbose:
            self.display.end_fold()
        if self._adapter is not None:
            fold = self.folds[self.curr_i_fold]
            fold_metrics = {}
            for task_name in self.tasks:
                th = fold.task(task_name)
                best_ep, best_f1 = th.val.best("f1") if "f1" in th.val else (-1, 0)
                fold_metrics[f"{task_name}_best_f1"] = best_f1
                fold_metrics[f"{task_name}_best_epoch"] = best_ep
            self._adapter.on_fold_end(self.curr_i_fold, fold_metrics)
        # Persist this fold's artefacts now so a later-fold crash (OOM SIGKILL,
        # SSD SIGBUS on long MPS runs) doesn't wipe the work we already did.
        # Best-effort: the storage method itself swallows exceptions so the
        # per-fold partial save never aborts training.
        if self._save_path is not None:
            try:
                self.storage.save_fold_partial(
                    fold_idx=self.curr_i_fold,
                    path=self._save_path,
                    label_map=self._label_map,
                )
            except Exception as exc:  # defensive; should not propagate
                import logging
                logging.getLogger(__name__).warning(
                    "per-fold partial save failed for fold %d: %s",
                    self.curr_i_fold, exc,
                )
        if self.curr_i_fold >= self.num_folds - 1:
            self.end()
            return
        self.curr_i_fold += 1
        self.folds[self.curr_i_fold].timer.start()
        if self._verbose:
            self.display.start_fold()

    # --- Metadata setters ---

    def set_model_parms(self, model_parms: HyperParams):
        self.model_parms = model_parms

    def set_model_arch(self, model_arch: str):
        self.model_arch = model_arch

    def set_flops(self, flops: FlopsMetrics):
        self.flops = flops
        if self._verbose:
            self.display.flops()

    def set_adapter(self, adapter) -> None:
        """Attach an optional external tracking adapter (W&B, MLflow, etc.).

        The adapter receives callbacks at run start, fold end, and run end.
        Pass None to disable external tracking (the default).
        """
        self._adapter = adapter

    # --- Lifecycle ---

    def start(self):
        self._ended = False
        self.timer.start()
        self.start_date = time.strftime("%Y%m%d_%H%M")
        self.get_curr_fold().start()
        if self._verbose:
            self.display.start_fold()
        if self._adapter is not None:
            self._adapter.on_run_start({
                "model_name": self.model_name,
                "model_type": self.model_type,
                "num_folds": self.num_folds,
                "tasks": list(self.tasks),
            })

    def _compare_and_save_records(self, run_path: Path) -> None:
        """Compare the current run against historical bests and display/save records."""
        summary_path = run_path / "summary" / "full_summary.json"
        if not summary_path.exists():
            return
        try:
            from tracking.records import compare_records, save_best_record

            current_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            comparison = compare_records(self._save_path, current_summary, run_path.name)
            if self._verbose:
                self.display.display_records(comparison)
            save_best_record(
                self._save_path, comparison, current_summary, run_path.name,
            )
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Record comparison failed: %s", exc,
            )

    def end(self):
        if self._ended:
            return
        self._ended = True
        self.timer.stop()
        self.end_date = time.strftime("%Y%m%d_%H%M%S")
        if self._verbose:
            self.display.end_training()
        if self._adapter is not None:
            self._adapter.on_run_end({
                "model_name": self.model_name,
                "num_folds": self.num_folds,
                "duration": self.timer.get_duration() if hasattr(self.timer, 'get_duration') else 0,
            })
            self._adapter.close()
