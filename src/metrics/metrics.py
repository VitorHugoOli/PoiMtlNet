import dataclasses
from typing import List, Optional
import time
from datetime import timedelta

import pandas as pd
from sklearn.metrics import classification_report

from configs.globals import CATEGORIES_MAP


@dataclasses.dataclass
class RawMetrics:
    """Class to hold type metrics."""

    loss: List[float] = dataclasses.field(default_factory=list)
    accuracy: List[float] = dataclasses.field(default_factory=list)
    val_loss: List[float] = dataclasses.field(default_factory=list)
    val_accuracy: List[float] = dataclasses.field(default_factory=list)

    def add_loss(self, loss: float):
        """Add loss value."""
        self.loss.append(loss)

    def add_accuracy(self, accuracy: float):
        """Add accuracy value."""
        self.accuracy.append(accuracy)

    def add_val_loss(self, val_loss: float):
        """Add validation loss value."""
        self.val_loss.append(val_loss)

    def add_val_accuracy(self, val_accuracy: float):
        """Add validation accuracy value."""
        self.val_accuracy.append(val_accuracy)

    def get_last_loss(self) -> Optional[float]:
        """Get last loss value."""
        return self.loss[-1] if self.loss else None

    def get_last_accuracy(self) -> Optional[float]:
        """Get last accuracy value."""
        return self.accuracy[-1] if self.accuracy else None

    def get_last_val_loss(self) -> Optional[float]:
        """Get last validation loss value."""
        return self.val_loss[-1] if self.val_loss else None

    def get_last_val_accuracy(self) -> Optional[float]:
        """Get last validation accuracy value."""
        return self.val_accuracy[-1] if self.val_accuracy else None


@dataclasses.dataclass
class FlopsMetrics:
    """Class to hold flops metrics."""

    flops: str = 0.0
    params: str = 0.0
    macs: str = 0.0

    def display(self):
        """Display flops metrics."""
        print(str(self))

    def __str__(self):
        """String representation of FlopsMetrics."""
        return (f'FLOPS: {self.flops} | '
                f'PARAMS: {self.params} | '
                f'MACS: {self.macs}')


@dataclasses.dataclass
class FoldResults:
    """Class to hold task results."""

    next: RawMetrics = dataclasses.field(default_factory=RawMetrics)
    category: RawMetrics = dataclasses.field(default_factory=RawMetrics)
    mtl: RawMetrics = dataclasses.field(default_factory=RawMetrics)
    flops: FlopsMetrics = dataclasses.field(default_factory=FlopsMetrics)
    
    # Time tracking attributes
    start_time: float = dataclasses.field(default_factory=lambda: time.time())
    epoch_start_times: List[float] = dataclasses.field(default_factory=list)
    epoch_durations: List[float] = dataclasses.field(default_factory=list)

    def start_epoch(self):
        """Record the start time of an epoch."""
        self.epoch_start_times.append(time.time())
        
    def end_epoch(self):
        """Record the end time of an epoch and calculate duration."""
        if self.epoch_start_times:
            duration = time.time() - self.epoch_start_times[-1]
            self.epoch_durations.append(duration)
            return duration
        return None
        
    def get_time_stats(self, current_epoch, total_epochs):
        """Get time statistics for the current training state."""
        if not self.epoch_durations:
            return None
            
        # Current epoch time
        current_epoch_time = self.epoch_durations[-1] if self.epoch_durations else 0
        
        # Total elapsed time since training started
        elapsed_time = time.time() - self.start_time
        
        # Calculate average epoch time from all completed epochs
        avg_epoch_time = sum(self.epoch_durations) / len(self.epoch_durations)
        
        # Estimate remaining time
        remaining_epochs = total_epochs - (current_epoch + 1)
        estimated_time_remaining = avg_epoch_time * remaining_epochs
        
        return {
            'epoch_time': current_epoch_time,
            'elapsed_time': elapsed_time,
            'estimated_remaining': estimated_time_remaining,
            'avg_epoch_time': avg_epoch_time
        }

    def display_training_status(self, epoch: Optional[int] = None, num_epochs: Optional[int] = None):
        """Display current training status including losses and accuracies."""
        # Training metrics
        mtl_loss = self.mtl.get_last_loss()
        mtl_acc = self.mtl.get_last_accuracy()
        mtl_val_loss = self.mtl.get_last_val_loss()
        mtl_val_acc = self.mtl.get_last_val_accuracy()

        next_loss = self.next.get_last_loss()
        next_acc = self.next.get_last_accuracy()
        next_val_loss = self.next.get_last_val_loss()
        next_val_acc = self.next.get_last_val_accuracy()

        cat_loss = self.category.get_last_loss()
        cat_acc = self.category.get_last_accuracy()
        cat_val_loss = self.category.get_last_val_loss()
        cat_val_acc = self.category.get_last_val_accuracy()

        # Format values with proper handling of None and potential errors
        def format_metric(value, as_percentage=False):
            try:
                if value is None:
                    return "N/A"
                else:
                    if as_percentage:
                        return f"{value * 100:.2f}%"
                    else:
                        return f"{value:.6f}"
            except (ValueError, TypeError):
                return "ERR"
        
        # Create a beautiful table-like output
        header_width = 80
        
        def print_header(title):
            print(f"\n{'='*header_width}")
            print(f"{title}".center(header_width))
            print(f"{'-'*header_width}")
            
        def print_metrics_row(name, loss, acc, val_loss, val_acc):
            print(f"| {name:<14} | {'Loss:':>7} {format_metric(loss):<10} | {'Acc:':>6} {format_metric(acc, True):<8} | "
                  f"{'Val Loss:':>10} {format_metric(val_loss):<10} | {'Val Acc:':>8} {format_metric(val_acc, True):<8} |")
            
        # Display epoch header with time information if applicable
        if epoch is not None and num_epochs is not None:
            print_header(f"EPOCH {epoch + 1}/{num_epochs} METRICS")
            
            # Display time information if available
            time_stats = self.get_time_stats(epoch, num_epochs)
            if time_stats:
                print(f"Time: {timedelta(seconds=int(time_stats['epoch_time']))}, "
                      f"Elapsed: {timedelta(seconds=int(time_stats['elapsed_time']))}, "
                      f"Remaining: {timedelta(seconds=int(time_stats['estimated_remaining']))}")
        else:
            print_header("CURRENT TRAINING METRICS")
        
        # Table header
        print(f"| {'Task':<14} | {'Training':<19} | {'':>15} | {'Validation':<20} | {'':>17} |")
        print(f"| {'':<14} | {'':<19} | {'':>15} | {'':>20} | {'':>17} |")
        
        # Print metrics rows
        print_metrics_row("MTL", mtl_loss, mtl_acc, mtl_val_loss, mtl_val_acc)
        print_metrics_row("Next POI", next_loss, next_acc, next_val_loss, next_val_acc)
        print_metrics_row("Category", cat_loss, cat_acc, cat_val_loss, cat_val_acc)
        
        # Table footer
        print(f"{'='*header_width}")
        
    def display_final_summary(self):
        """Display a summary of training results for this fold."""
        header_width = 80
        
        total_time = time.time() - self.start_time
        
        print(f"\n{'='*header_width}")
        print("FOLD TRAINING COMPLETE".center(header_width))
        print(f"{'-'*header_width}")
        
        # Show total training time
        print(f"Total training time: {timedelta(seconds=int(total_time))}")
        if self.epoch_durations:
            avg_epoch_time = sum(self.epoch_durations) / len(self.epoch_durations)
            print(f"Average epoch time: {timedelta(seconds=int(avg_epoch_time))}")
            
        # Show best metrics
        if self.mtl.val_accuracy:
            best_epoch = self.mtl.val_accuracy.index(max(self.mtl.val_accuracy))
            print(f"\nBest validation accuracy at epoch {best_epoch + 1}:")
            print(f"MTL: {self.mtl.val_accuracy[best_epoch] * 100:.2f}%")
            print(f"Next POI: {self.next.val_accuracy[best_epoch] * 100:.2f}%")
            print(f"Category: {self.category.val_accuracy[best_epoch] * 100:.2f}%")
            
        print(f"{'='*header_width}")


class TrainingMetrics:
    """Class to hold training metrics."""

    def __init__(self):
        """Initialize training metrics."""
        self.folds: List[FoldResults] = []
        self.report: dict = {}
        self.start_time = time.time()

    def add_fold_results(self, fold_results: FoldResults):
        """Add fold results."""
        self.folds.append(fold_results)

    def add_report(self, report: dict):
        """Add report."""
        self.report = report

    def get_average_metrics(self) -> dict:
        """Calculate average metrics across all folds."""
        if not self.folds:
            return {}

        avg_metrics = {
            'mtl': {
                'val_accuracy': 0.0,
                'val_loss': 0.0
            },
            'next': {
                'val_accuracy': 0.0,
                'val_loss': 0.0
            },
            'category': {
                'val_accuracy': 0.0,
                'val_loss': 0.0
            }
        }

        fold_count = len(self.folds)
        for fold in self.folds:
            for metric_type in ['mtl', 'next', 'category']:
                fold_metrics = getattr(fold, metric_type)
                val_acc = fold_metrics.get_last_val_accuracy()
                val_loss = fold_metrics.get_last_val_loss()

                if val_acc is not None:
                    avg_metrics[metric_type]['val_accuracy'] += val_acc / fold_count
                if val_loss is not None:
                    avg_metrics[metric_type]['val_loss'] += val_loss / fold_count

        return avg_metrics
        
    def display_summary(self):
        """Display a summary of training metrics across all folds."""
        if not self.folds:
            print("No training data available.")
            return
            
        avg_metrics = self.get_average_metrics()
        
        print("\n" + "="*80)
        print("TRAINING SUMMARY".center(80))
        print("="*80)
        
        print(f"\nTotal folds: {len(self.folds)}")
        print(f"Total training time: {timedelta(seconds=int(time.time() - self.start_time))}")
        
        print("\n" + "-"*80)
        print("Average Validation Metrics Across All Folds".center(80))
        print("-"*80)
        
        for task in ['mtl', 'next', 'category']:
            task_name = {'mtl': 'Multi-Task Learning', 'next': 'Next POI', 'category': 'Category'}[task]
            acc = avg_metrics[task]['val_accuracy']
            loss = avg_metrics[task]['val_loss']
            print(f"{task_name:<20} | Accuracy: {acc*100:.2f}% | Loss: {loss:.6f}")
            
        print("="*80)


class UtilsMetrics:
    """Class to hold utility functions for metrics."""

    @staticmethod
    def display_report(y_true, y_pred, task_name):
        """Display classification report."""
        print(f"\nReport for {task_name}:\n")
        try:
            report = classification_report(y_true, y_pred, zero_division=1, output_dict=True)

            # Extract class metrics (exclude 'accuracy', 'macro avg', etc.)
            class_metrics = {k: v for k, v in report.items()
                             if isinstance(v, dict) and k not in ['macro avg', 'weighted avg']}

            # Map numerical class labels to category names if they exist in CATEGORIES_MAP
            mapped_metrics = {}
            for k, v in class_metrics.items():
                try:
                    # Try to convert to int for mapping, if not possible, use as is
                    class_key = int(k) if k.isdigit() else k
                    class_name = CATEGORIES_MAP.get(class_key, k)
                    mapped_metrics[class_name] = v
                except (ValueError, TypeError):
                    # If conversion fails, use original key
                    mapped_metrics[k] = v

            # Format header
            metrics = ['precision', 'recall', 'f1-score']
            header = " " * 15 + "".join(f"{m:>12}" for m in metrics)
            print(header)
            print("-" * len(header))

            # Print each class's metrics
            for class_name, metrics_dict in mapped_metrics.items():
                # Truncate class name if too long
                truncated_name = str(class_name)[:14] + "…" if len(str(class_name)) > 15 else f"{class_name:<15}"
                metrics_display = "".join(f"{metrics_dict.get(m, 0) * 100:>11.1f}%" for m in metrics)
                print(f"{truncated_name}{metrics_display}")

            # Print average metrics
            print("-" * len(header))
            for avg_type in ['macro avg', 'weighted avg']:
                if avg_type in report:
                    avg_metrics = report[avg_type]
                    metrics_display = "".join(f"{avg_metrics.get(m, 0) * 100:>11.1f}%" for m in metrics)
                    print(f"{avg_type:<15}{metrics_display}")

            # Print accuracy
            if 'accuracy' in report:
                print(f"Accuracy: {report['accuracy'] * 100:.2f}%")

        except Exception as e:
            print(f"Error generating classification report: {str(e)}")
