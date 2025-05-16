from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.data import DataLoader

from configs.globals import DEVICE

from model.category.head.configs.category_config import CfgCategoryModel, CfgCategoryHyperparams, CfgCategoryTraining
from model.category.head.engine.evaluation import evaluate
from model.category.head.engine.trainer import train
from model.category.head.modeling.category_head import CategoryHeadSingle
from utils.calc_flops.calculate_model_flops import calculate_model_flops
from utils.ml_history.metrics import MLHistory, FlopsMetrics
from utils.ml_history.parms.neural import NeuralParams


def run_cv(
        history: MLHistory,
        folds: list[tuple[DataLoader, DataLoader]],
):
    """Run cross-validation for the model."""
    for idx, (train_loader, val_loader) in enumerate(folds):
        history.display.start_fold()
        model = CategoryHeadSingle(
            input_dim=CfgCategoryModel.INPUT_DIM,
            hidden_dims=tuple(CfgCategoryModel.HIDDEN_DIMS),
            num_classes=CfgCategoryModel.NUM_CLASSES,
            dropout=CfgCategoryModel.DROPOUT,
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()

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

        # Calculate FLOPs
        sample = next(iter(train_loader))[0].to(DEVICE)
        result = calculate_model_flops(model,
                                       sample_input=sample,
                                       print_report=True,
                                       units='K'
                                       )
        history.set_flops(FlopsMetrics(flops=result['total_flops'], params=result['params']['total']))

        train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            DEVICE,
            history=history,
        )

        report = evaluate(model, val_loader, DEVICE)

        history.get_curr_fold().to('category').add_report(report)
        history.step()
        history.display.end_fold()