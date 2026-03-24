import numpy as np
import torch
from sklearn.utils import compute_class_weight
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.data import DataLoader

from configs.globals import DEVICE
from configs.model import InputsConfig

from configs.category_config import CfgCategoryModel, CfgCategoryHyperparams, CfgCategoryTraining
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
):
    """Run cross-validation for the model."""
    for idx, (train_loader, val_loader) in enumerate(folds):
        model = CategoryHeadEnsemble(
            input_dim=CfgCategoryModel.INPUT_DIM,
            hidden_dim=64,
            # hidden_dims=CfgCategoryModel.HIDDEN_DIMS,
            num_classes=CfgCategoryModel.NUM_CLASSES,
            dropout=CfgCategoryModel.DROPOUT,
        ).to(DEVICE)

        y_all = train_loader.dataset.targets.numpy()
        cls = np.arange(CfgCategoryModel.NUM_CLASSES)
        weights = compute_class_weight('balanced', classes=cls, y=y_all)
        alpha = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
        criterion = nn.CrossEntropyLoss(
            reduction='mean',
            # weight=alpha
        )

        optimizer = AdamW(
            model.parameters(),
            lr=CfgCategoryHyperparams.LR,
            weight_decay=CfgCategoryHyperparams.WEIGHT_DECAY,
        )
        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=CfgCategoryHyperparams.MAX_LR,
            epochs=CfgCategoryTraining.EPOCHS,
            steps_per_epoch=len(train_loader),
        )

        history.set_model_arch(str(model))

        history.set_model_parms(
            NeuralParams(
                batch_size=CfgCategoryTraining.BATCH_SIZE,
                num_epochs=CfgCategoryTraining.EPOCHS,
                learning_rate=CfgCategoryHyperparams.LR,
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
            timeout=InputsConfig.TIMEOUT_TEST,
            target_cutoff=InputsConfig.CATEGORY_TARGET,
        )

        report = evaluate(model, val_loader, DEVICE, best_state=history.fold.task('category').best.best_state)

        history.fold.task('category').report = report
        history.step()

        # Free MPS memory between folds
        if DEVICE.type == 'mps':
            clear_mps_cache()
