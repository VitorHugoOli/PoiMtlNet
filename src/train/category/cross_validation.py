import numpy as np
import torch
from pathlib import Path
from sklearn.utils import compute_class_weight
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.data import DataLoader
from typing import Optional

from configs.globals import DEVICE
from configs.experiment import ExperimentConfig
from model.category.category_head_enhanced import CategoryHeadGated, CategoryHeadResidual, CategoryHeadEnsemble, \
    CategoryHeadAttentionPooling, ResidualBlock
from train.category.evaluation import evaluate
from train.category.trainer import train
from model.category.CategoryHeadTransformer import CategoryHeadTransformer
from criterion.FocalLoss import FocalLoss
from common.calc_flops.calculate_model_flops import calculate_model_flops
from common.ml_history import MLHistory, FlopsMetrics, NeuralParams
from common.mps_support import clear_mps_cache


def run_cv(
        history: MLHistory,
        folds: list[tuple[DataLoader, DataLoader]],
        config: ExperimentConfig,
        results_path: Optional[Path] = None,
):
    """Run cross-validation for the model."""
    num_classes = config.model_params.get('num_classes', 7)

    for idx, (train_loader, val_loader) in enumerate(folds):
        model = CategoryHeadEnsemble(
            **config.model_params,
        ).to(DEVICE)

        y_all = train_loader.dataset.targets.numpy()
        cls = np.arange(num_classes)
        weights = compute_class_weight('balanced', classes=cls, y=y_all)
        alpha = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
        criterion = nn.CrossEntropyLoss(
            reduction='mean',
            weight=alpha if config.use_class_weights else None,
        )

        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=config.max_lr,
            epochs=config.epochs,
            steps_per_epoch=len(train_loader),
        )

        history.set_model_arch(str(model))

        history.set_model_parms(
            NeuralParams(
                batch_size=config.batch_size,
                num_epochs=config.epochs,
                learning_rate=config.learning_rate,
                optimizer=optimizer.__class__.__name__,
                optimizer_state=optimizer.state_dict(),
                scheduler=scheduler.__class__.__name__,
                scheduler_state=scheduler.state_dict(),
                criterion={
                    'category': criterion.__class__.__name__},
                criterion_state={
                    'category': criterion.state_dict()
                }
            )
        )

        # Calculate FLOPs only on first fold (architecture is identical across folds)
        if idx == 0:
            sample = next(iter(train_loader))[0].to(DEVICE)
            result = calculate_model_flops(model,
                                           sample_input=sample,
                                           print_report=True,
                                           units='K'
                                           )
            if 'total_flops' in result:
                history.set_flops(FlopsMetrics(flops=result['total_flops'], params=result['params']['total']))
            else:
                print(f"Warning: FLOPs calculation failed: {result.get('error', 'Unknown error')}")

        train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            DEVICE,
            history=history.get_curr_fold(),
            epochs=config.epochs,
            max_grad_norm=config.max_grad_norm,
            timeout=config.timeout,
            target_cutoff=config.target_cutoff,
        )

        report = evaluate(model, val_loader, DEVICE, best_state=history.fold.task('category').best.best_state)

        history.fold.task('category').report = report
        history.step()

        # Free MPS memory between folds
        if DEVICE.type == 'mps':
            clear_mps_cache()

    # Write run manifest
    if results_path is not None:
        from configs.experiment import RunManifest
        from configs.paths import IoPaths, EmbeddingEngine
        engine = EmbeddingEngine(config.embedding_engine)
        manifest = RunManifest.from_current_env(
            config=config,
            dataset_paths={
                "category_input": IoPaths.get_category(config.state, engine),
            },
        )
        manifest.write(results_path)
