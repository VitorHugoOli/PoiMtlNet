import dataclasses
from typing import List, Optional
import time
from datetime import timedelta
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report

from configs.globals import CATEGORIES_MAP


@dataclasses.dataclass
class RawMetrics:
    """Class to hold type metrics."""

    loss: List[float] = dataclasses.field(default_factory=list)
    accuracy: List[float] = dataclasses.field(default_factory=list)
    val_loss: List[float] = dataclasses.field(default_factory=list)
    val_accuracy: List[float] = dataclasses.field(default_factory=list)

    best_model: Optional[dict] = None

    def add_loss(self, loss: float):
        """Add loss value."""
        self.loss.append(loss)

    def add_accuracy(self, accuracy: float):
        """Add accuracy value."""
        self.accuracy.append(accuracy)

    def add_val_loss(self, val_loss: float):
        """Add validation loss value."""
        self.val_loss.append(val_loss)

    def add_val_accuracy(self, val_accuracy: float, model_state: Optional[dict] = None):
        """Add validation accuracy value and save the model if it's the best."""
        self.val_accuracy.append(val_accuracy)
        if val_accuracy == max(self.val_accuracy):  # Check if it's the best accuracy
            self.save_best_model(model_state)

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

    def save_best_model(self, model_state: dict):
        """Save the best model state."""
        self.best_model = model_state


@dataclasses.dataclass
class FlopsMetrics:
    """Class to hold flops metrics."""

    flops: str = 0.0
    params: str = 0.0

    def display(self):
        """Display flops metrics."""
        print(str(self))

    def __str__(self):
        """String representation of FlopsMetrics."""
        return (f'FLOPS: {self.flops} | '
                f'PARAMS: {self.params}')


@dataclasses.dataclass
class FoldResults:
    """Class to hold task results."""

    next: RawMetrics = dataclasses.field(default_factory=RawMetrics)
    category: RawMetrics = dataclasses.field(default_factory=RawMetrics)
    mtl: RawMetrics = dataclasses.field(default_factory=RawMetrics)
    flops: FlopsMetrics = dataclasses.field(default_factory=FlopsMetrics)
    next_report: dict = dataclasses.field(default_factory=dict)
    category_report: dict = dataclasses.field(default_factory=dict)

    # Time tracking attributes
    start_time: float = dataclasses.field(default_factory=lambda: time.time())

    def add_next_report(self, report: dict):
        """Add report."""
        self.next_report = report

    def add_category_report(self, report: dict):
        """Add report."""
        self.category_report = report

    def display_final_summary(self):
        """Display a summary of training results for this fold."""
        header_width = 80

        total_time = time.time() - self.start_time

        print(f"\n{'=' * header_width}")
        print("FOLD TRAINING COMPLETE".center(header_width))
        print(f"{'-' * header_width}")

        # Show total training time
        print(f"Total training time: {timedelta(seconds=int(total_time))}")

        # Show best metrics
        if self.mtl.val_accuracy:
            best_epoch = self.mtl.val_accuracy.index(max(self.mtl.val_accuracy))
            print(f"\nBest validation accuracy at epoch {best_epoch + 1}:")
            print(f"MTL: {self.mtl.val_accuracy[best_epoch] * 100:.2f}%")
            print(f"Next POI: {self.next.val_accuracy[best_epoch] * 100:.2f}%")
            print(f"Category: {self.category.val_accuracy[best_epoch] * 100:.2f}%")

        if self.next_report:
            print("\nNext POI Classification Report:")
            UtilsMetrics.display_report_json(self.next_report, "Next POI")

        if self.category_report:
            print("\nCategory Classification Report:")
            UtilsMetrics.display_report_json(self.category_report, "Category")

        print(f"{'=' * header_width}")


class TrainingMetrics:
    """Class to hold training metrics."""

    def __init__(self):
        """Initialize training metrics."""
        self.folds: List[FoldResults] = []
        self.start_time = time.time()

    def add_fold_results(self, fold_results: FoldResults):
        """Add fold results."""
        self.folds.append(fold_results)

    def get_average_metrics(self) -> dict:
        """Calculate average metrics across all folds with detailed category metrics."""
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
            },
            'next_categories': {},
            'category_categories': {}
        }

        fold_count = len(self.folds)

        # Calculate average validation metrics
        for fold in self.folds:
            for metric_type in ['mtl', 'next', 'category']:
                fold_metrics = getattr(fold, metric_type)
                val_acc = fold_metrics.get_last_val_accuracy()
                val_loss = fold_metrics.get_last_val_loss()

                if val_acc is not None:
                    avg_metrics[metric_type]['val_accuracy'] += val_acc / fold_count
                if val_loss is not None:
                    avg_metrics[metric_type]['val_loss'] += val_loss / fold_count

        # Calculate average classification report metrics
        for fold in self.folds:
            # Process next POI category metrics
            if fold.next_report:
                for class_key in fold.next_report:
                    if class_key not in ['accuracy', 'macro avg', 'weighted avg']:
                        if class_key not in avg_metrics['next_categories']:
                            avg_metrics['next_categories'][class_key] = {
                                'precision': 0.0,
                                'recall': 0.0,
                                'f1-score': 0.0,
                                'support': 0
                            }

                        for metric in ['precision', 'recall', 'f1-score']:
                            avg_metrics['next_categories'][class_key][metric] += fold.next_report[class_key][
                                                                                     metric] / fold_count

                        # For support, we sum rather than average
                        avg_metrics['next_categories'][class_key]['support'] += fold.next_report[class_key][
                                                                                    'support'] / fold_count

            # Process category metrics
            if fold.category_report:
                for class_key in fold.category_report:
                    if class_key not in ['accuracy', 'macro avg', 'weighted avg']:
                        if class_key not in avg_metrics['category_categories']:
                            avg_metrics['category_categories'][class_key] = {
                                'precision': 0.0,
                                'recall': 0.0,
                                'f1-score': 0.0,
                                'support': 0
                            }

                        for metric in ['precision', 'recall', 'f1-score']:
                            avg_metrics['category_categories'][class_key][metric] += fold.category_report[class_key][
                                                                                         metric] / fold_count

                        # For support, we sum rather than average
                        avg_metrics['category_categories'][class_key]['support'] += fold.category_report[class_key][
                                                                                        'support'] / fold_count

        # Add average of averages
        for report_type, report_key in [('next', 'next_categories'), ('category', 'category_categories')]:
            if avg_metrics[report_key]:
                avg_metrics[f'{report_type}_macro_avg'] = {
                    'precision': sum(cat['precision'] for cat in avg_metrics[report_key].values()) / len(
                        avg_metrics[report_key]),
                    'recall': sum(cat['recall'] for cat in avg_metrics[report_key].values()) / len(
                        avg_metrics[report_key]),
                    'f1-score': sum(cat['f1-score'] for cat in avg_metrics[report_key].values()) / len(
                        avg_metrics[report_key])
                }

        return avg_metrics

    def display_summary(self):
        """Display a summary of training metrics across all folds with detailed category metrics."""
        if not self.folds:
            print("No training data available.")
            return

        avg_metrics = self.get_average_metrics()

        print("\n" + "=" * 80)
        print("TRAINING SUMMARY".center(80))
        print("=" * 80)

        print(f"\nTotal folds: {len(self.folds)}")
        print(f"Total training time: {timedelta(seconds=int(time.time() - self.start_time))}")

        print("\n" + "-" * 80)
        print("Average Validation Metrics Across All Folds".center(80))
        print("-" * 80)

        for task in ['mtl', 'next', 'category']:
            task_name = {'mtl': 'Multi-Task Learning', 'next': 'Next POI', 'category': 'Category'}[task]
            acc = avg_metrics[task]['val_accuracy']
            loss = avg_metrics[task]['val_loss']
            print(f"{task_name:<20} | Accuracy: {acc * 100:.2f}% | Loss: {loss:.6f}")

        # Display average metrics per category
        for report_type, report_title in [('next_categories', 'Next POI Categories'),
                                          ('category_categories', 'Category Types')]:
            if report_type in avg_metrics and avg_metrics[report_type]:
                print("\n" + "-" * 80)
                print(f"Average Metrics for {report_title}".center(80))
                print("-" * 80)

                # Create a DataFrame for better display
                metrics_df = pd.DataFrame()

                # Add category metrics
                for class_key, metrics in avg_metrics[report_type].items():
                    try:
                        # Map numerical class to category name if applicable
                        category_name = CATEGORIES_MAP.get(int(class_key), class_key) if isinstance(class_key,
                                                                                                    str) and class_key.isdigit() else class_key

                        # Create row with metrics
                        row_data = {
                            'Category': category_name,
                            'Precision': f"{metrics['precision'] * 100:.2f}%",
                            'Recall': f"{metrics['recall'] * 100:.2f}%",
                            'F1-Score': f"{metrics['f1-score'] * 100:.2f}%",
                            'Support': int(metrics['support'])
                        }
                        metrics_df = pd.concat([metrics_df, pd.DataFrame([row_data])], ignore_index=True)

                    except (ValueError, KeyError) as e:
                        print(f"Error processing class {class_key}: {str(e)}")

                # Add macro average
                task_type = report_type.split('_')[0]  # 'next' or 'category'
                if f'{task_type}_macro_avg' in avg_metrics:
                    macro_avg = avg_metrics[f'{task_type}_macro_avg']
                    row_data = {
                        'Category': 'macro avg',
                        'Precision': f"{macro_avg['precision'] * 100:.2f}%",
                        'Recall': f"{macro_avg['recall'] * 100:.2f}%",
                        'F1-Score': f"{macro_avg['f1-score'] * 100:.2f}%",
                        'Support': '-'
                    }
                    metrics_df = pd.concat([metrics_df, pd.DataFrame([row_data])], ignore_index=True)

                # Display the DataFrame
                pd.set_option('display.max_rows', None)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', 120)
                print(metrics_df.to_string(index=False))
                pd.reset_option('display.max_rows')
                pd.reset_option('display.max_columns')
                pd.reset_option('display.width')

        print("=" * 80)

    def _create_loss_plots(self, output_dir):
        """
        Create and save plots showing the evolution of loss and validation loss by epoch.

        Args:
            output_dir: Directory to save the plot files
        """
        if not self.folds:
            print("No training data available to plot.")
            return

        # Ensure plots directory exists
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Create plots for each task type
        task_types = [
            ("mtl", "Multi-Task Learning"),
            ("next", "Next POI Prediction"),
            ("category", "Category Classification")
        ]

        for task_attr, task_name in task_types:
            plt.figure(figsize=(12, 8))

            # Get the maximum length of epochs across all folds for this task
            max_epochs = max([len(getattr(fold, task_attr).loss) for fold in self.folds])
            epochs = np.arange(1, max_epochs + 1)

            # Plot training loss for each fold
            for fold_idx, fold in enumerate(self.folds):
                fold_metrics = getattr(fold, task_attr)
                fold_num = fold_idx + 1

                # Training loss
                loss_data = fold_metrics.loss
                plt.plot(
                    np.arange(1, len(loss_data) + 1),
                    loss_data,
                    linestyle='-',
                    marker='o',
                    markersize=4,
                    label=f'Fold {fold_num} - Training Loss'
                )

                # Validation loss
                val_loss_data = fold_metrics.val_loss
                plt.plot(
                    np.arange(1, len(val_loss_data) + 1),
                    val_loss_data,
                    linestyle='--',
                    marker='x',
                    markersize=4,
                    label=f'Fold {fold_num} - Validation Loss'
                )

            # Add plot styling
            plt.title(f'{task_name} - Loss Evolution by Epoch', fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('Loss', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='upper right', fontsize=10)

            # Ensure y-axis starts from 0 or the minimum value if it's negative
            min_y = min([min(min(getattr(fold, task_attr).loss, default=0),
                             min(getattr(fold, task_attr).val_loss, default=0))
                         for fold in self.folds], default=0)
            plt.ylim(bottom=max(0, min_y - 0.1))

            # Save the plot
            plot_file = os.path.join(plots_dir, f'{task_attr}_loss_evolution.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Loss evolution plot for {task_name} saved to {plot_file}")

            # Create a second plot for accuracy evolution
            plt.figure(figsize=(12, 8))

            # Plot training accuracy for each fold
            for fold_idx, fold in enumerate(self.folds):
                fold_metrics = getattr(fold, task_attr)
                fold_num = fold_idx + 1

                # Training accuracy
                acc_data = fold_metrics.accuracy
                plt.plot(
                    np.arange(1, len(acc_data) + 1),
                    acc_data,
                    linestyle='-',
                    marker='o',
                    markersize=4,
                    label=f'Fold {fold_num} - Training Accuracy'
                )

                # Validation accuracy
                val_acc_data = fold_metrics.val_accuracy
                plt.plot(
                    np.arange(1, len(val_acc_data) + 1),
                    val_acc_data,
                    linestyle='--',
                    marker='x',
                    markersize=4,
                    label=f'Fold {fold_num} - Validation Accuracy'
                )

            # Add plot styling
            plt.title(f'{task_name} - Accuracy Evolution by Epoch', fontsize=16)
            plt.xlabel('Epoch', fontsize=14)
            plt.ylabel('Accuracy', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(loc='lower right', fontsize=10)

            # Set y-axis limits for accuracy (0-1)
            plt.ylim(0, 1.05)

            # Save the plot
            plot_file = os.path.join(plots_dir, f'{task_attr}_accuracy_evolution.png')
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Accuracy evolution plot for {task_name} saved to {plot_file}")

    def export_to_csv(self, output_dir="./metrics_results"):
        """
    Export metrics results to CSV files and create visualization plots.

        Args:
        output_dir: Directory to save the CSV files and plots (default: ./metrics_results)

        Creates:
            - Individual fold CSV files with per-category metrics
            - Summary CSV with average metrics across all folds
        - Plots showing loss and accuracy evolution for each task type
        """
        import os

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        if not self.folds:
            print("No training data available to export.")
            return

        # Export each fold's results
        for i, fold in enumerate(self.folds):
            fold_idx = i + 1

            # Export Next POI metrics
            if fold.next_report:
                next_df = self._create_metrics_dataframe(fold.next_report, "next", CATEGORIES_MAP)
                next_df.to_csv(f"{output_dir}/fold_{fold_idx}_next_poi_metrics.csv", index=False)

            # Export Category metrics
            if fold.category_report:
                category_df = self._create_metrics_dataframe(fold.category_report, "category", CATEGORIES_MAP)
                category_df.to_csv(f"{output_dir}/fold_{fold_idx}_category_metrics.csv", index=False)

        # Export average metrics across all folds
        avg_metrics = self.get_average_metrics()

        # Create summary dataframe for Next POI
        if 'next_categories' in avg_metrics and avg_metrics['next_categories']:
            next_summary_df = pd.DataFrame()

            for class_key, metrics in avg_metrics['next_categories'].items():
                category_name = CATEGORIES_MAP.get(int(class_key), class_key) if isinstance(class_key,
                                                                                            str) and class_key.isdigit() else class_key

                row_data = {
                    'Category': category_name,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1-score'],
                    'Support': metrics['support']
                }
                next_summary_df = pd.concat([next_summary_df, pd.DataFrame([row_data])], ignore_index=True)

            # Add macro average
            if 'next_macro_avg' in avg_metrics:
                macro_avg = avg_metrics['next_macro_avg']
                row_data = {
                    'Category': 'macro avg',
                    'Precision': macro_avg['precision'],
                    'Recall': macro_avg['recall'],
                    'F1-Score': macro_avg['f1-score'],
                    'Support': None
                }
                next_summary_df = pd.concat([next_summary_df, pd.DataFrame([row_data])], ignore_index=True)

            next_summary_df.to_csv(f"{output_dir}/summary_next_poi_metrics.csv", index=False)

        # Create summary dataframe for Categories
        if 'category_categories' in avg_metrics and avg_metrics['category_categories']:
            category_summary_df = pd.DataFrame()

            for class_key, metrics in avg_metrics['category_categories'].items():
                category_name = CATEGORIES_MAP.get(int(class_key), class_key) if isinstance(class_key,
                                                                                            str) and class_key.isdigit() else class_key

                row_data = {
                    'Category': category_name,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1-score'],
                    'Support': metrics['support']
                }
                category_summary_df = pd.concat([category_summary_df, pd.DataFrame([row_data])], ignore_index=True)

            # Add macro average
            if 'category_macro_avg' in avg_metrics:
                macro_avg = avg_metrics['category_macro_avg']
                row_data = {
                    'Category': 'macro avg',
                    'Precision': macro_avg['precision'],
                    'Recall': macro_avg['recall'],
                    'F1-Score': macro_avg['f1-score'],
                    'Support': None
                }
                category_summary_df = pd.concat([category_summary_df, pd.DataFrame([row_data])], ignore_index=True)

            category_summary_df.to_csv(f"{output_dir}/summary_category_metrics.csv", index=False)

        # Export overall summary with validation metrics
        overall_df = pd.DataFrame([{
            'Task': 'Multi-Task Learning',
            'Validation Accuracy': avg_metrics['mtl']['val_accuracy'],
            'Validation Loss': avg_metrics['mtl']['val_loss']
        }, {
            'Task': 'Next POI',
            'Validation Accuracy': avg_metrics['next']['val_accuracy'],
            'Validation Loss': avg_metrics['next']['val_loss']
        }, {
            'Task': 'Category',
            'Validation Accuracy': avg_metrics['category']['val_accuracy'],
            'Validation Loss': avg_metrics['category']['val_loss']
        }])

        overall_df.to_csv(f"{output_dir}/summary_overall_metrics.csv", index=False)

        # Export flops results
        flops_df = pd.DataFrame([{
            'Task': 'Multi-Task Learning',
            'FLOPS': self.folds[0].flops.flops,
            'PARAMS': self.folds[0].flops.params
        }])
        flops_df.to_csv(f"{output_dir}/summary_flops_metrics.csv", index=False)

        # Export loss and accuracy data as CSV
        for task_attr, task_name in [("mtl", "Multi-Task Learning"),
                                     ("next", "Next POI"),
                                     ("category", "Category")]:
            # Create DataFrame for loss and validation loss
            loss_data = {
                'Epoch': list(range(1, max([len(getattr(fold, task_attr).loss) for fold in self.folds]) + 1))
            }

            # Add loss and val_loss for each fold
            for fold_idx, fold in enumerate(self.folds):
                fold_num = fold_idx + 1
                task_metrics = getattr(fold, task_attr)

                # Add loss data with padding for shorter folds
                loss_values = task_metrics.loss
                max_epochs = len(loss_data['Epoch'])
                padded_loss = loss_values + [None] * (max_epochs - len(loss_values))
                loss_data[f'Fold_{fold_num}_Loss'] = padded_loss

                # Add validation loss data
                val_loss_values = task_metrics.val_loss
                padded_val_loss = val_loss_values + [None] * (max_epochs - len(val_loss_values))
                loss_data[f'Fold_{fold_num}_Val_Loss'] = padded_val_loss

                # Add accuracy data
                acc_values = task_metrics.accuracy
                padded_acc = acc_values + [None] * (max_epochs - len(acc_values))
                loss_data[f'Fold_{fold_num}_Accuracy'] = padded_acc

                # Add validation accuracy data
                val_acc_values = task_metrics.val_accuracy
                padded_val_acc = val_acc_values + [None] * (max_epochs - len(val_acc_values))
                loss_data[f'Fold_{fold_num}_Val_Accuracy'] = padded_val_acc

            # Save to CSV
            loss_df = pd.DataFrame(loss_data)
            loss_df.to_csv(f"{output_dir}/{task_attr}_loss_accuracy_by_epoch.csv", index=False)

        # Create and save plots
        self._create_loss_plots(output_dir)

        print(f"Metrics and plots exported to {output_dir}/")

    def _create_metrics_dataframe(self, report_dict, task_type, categories_map=None):
        """
        Create a DataFrame from a classification report dictionary.

        Args:
            report_dict: Classification report dictionary
            task_type: Type of task ('next' or 'category')
            categories_map: Dictionary mapping class IDs to category names

        Returns:
            pandas.DataFrame: DataFrame with metrics
        """
        metrics_df = pd.DataFrame()

        # Process class metrics
        for class_key in [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]:
            try:
                # Map numerical class to category name if applicable
                category_name = categories_map.get(int(class_key), class_key) if categories_map and isinstance(
                    class_key, str) and class_key.isdigit() else class_key
                class_data = report_dict[class_key]

                # Create row with metrics
                row_data = {
                    'Category': category_name,
                    'Task': task_type,
                    'Precision': class_data['precision'],
                    'Recall': class_data['recall'],
                    'F1-Score': class_data['f1-score'],
                    'Support': int(class_data['support'])
                }
                metrics_df = pd.concat([metrics_df, pd.DataFrame([row_data])], ignore_index=True)

            except (ValueError, KeyError) as e:
                print(f"Error processing class {class_key}: {str(e)}")

        # Add average metrics
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in report_dict:
                avg_data = report_dict[avg_type]
                row_data = {
                    'Category': avg_type,
                    'Task': task_type,
                    'Precision': avg_data['precision'],
                    'Recall': avg_data['recall'],
                    'F1-Score': avg_data['f1-score'],
                    'Support': int(avg_data['support']) if 'support' in avg_data else None
                }
                metrics_df = pd.concat([metrics_df, pd.DataFrame([row_data])], ignore_index=True)

        # Add accuracy as a separate row if available
        if 'accuracy' in report_dict:
            accuracy_row = {
                'Category': 'Overall Accuracy',
                'Task': task_type,
                'Precision': None,
                'Recall': None,
                'F1-Score': report_dict['accuracy'],
                'Support': None
            }
            metrics_df = pd.concat([metrics_df, pd.DataFrame([accuracy_row])], ignore_index=True)

        return metrics_df


class UtilsMetrics:
    """Class to hold utility functions for metrics."""

    @staticmethod
    def display_report_json(report_dict, task_name=None):
        """
        Display a classification report dictionary with mapped category names.

        Args:
            report_dict: Classification report dictionary
            task_name: Optional name of the task for display purposes
        """
        if task_name:
            print(f"\nJSON Report for {task_name}:\n")

        # Create a DataFrame for better display
        metrics_df = pd.DataFrame()

        # Process class metrics
        for class_key in [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]:
            try:
                # Map numerical class to category name
                category_name = CATEGORIES_MAP.get(int(class_key), class_key) if class_key.isdigit() else class_key
                class_data = report_dict[class_key]

                # Create row with metrics
                row_data = {
                    'Class': category_name,
                    'F1-Score': f"{class_data['f1-score'] * 100:.2f}%",
                    'Precision': f"{class_data['precision'] * 100:.2f}%",
                    'Recall': f"{class_data['recall'] * 100:.2f}%",
                    'Support': int(class_data['support'])
                }
                metrics_df = pd.concat([metrics_df, pd.DataFrame([row_data])], ignore_index=True)

            except (ValueError, KeyError) as e:
                print(f"Error processing class {class_key}: {str(e)}")

        # Add average metrics
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in report_dict:
                avg_data = report_dict[avg_type]
                row_data = {
                    'Class': avg_type,
                    'F1-Score': f"{avg_data['f1-score'] * 100:.2f}%",
                    'Precision': f"{avg_data['precision'] * 100:.2f}%",
                    'Recall': f"{avg_data['recall'] * 100:.2f}%",
                    'Support': int(avg_data['support']) if 'support' in avg_data else '-'
                }
                metrics_df = pd.concat([metrics_df, pd.DataFrame([row_data])], ignore_index=True)

        # Add accuracy as a separate row
        if 'accuracy' in report_dict:
            accuracy_row = {
                'Class': 'Overall Accuracy',
                'Precision': '-',
                'Recall': '-',
                'F1-Score': f"{report_dict['accuracy'] * 100:.2f}%",
                'Support': '-'
            }
            metrics_df = pd.concat([metrics_df, pd.DataFrame([accuracy_row])], ignore_index=True)

        # Display the DataFrame
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        print(metrics_df.to_string(index=False))
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
