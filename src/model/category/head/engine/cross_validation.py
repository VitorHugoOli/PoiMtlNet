from torch_geometric.data import DataLoader

from configs.globals import DEVICE

from model.category.head.configs.category_config import CfgCategoryModel
from model.category.head.engine.evaluation import evaluate
from model.category.head.engine.trainer import train
from model.mtlnet.modeling.category_head import CategoryHeadSingle
from utils.ml_history.metrics import MLHistory

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

        train(
            model,
            train_loader,
            val_loader,
            DEVICE,
            history=history,
        )

        report = evaluate(model, val_loader, DEVICE)

        history.get_curr_fold().to('category').add_report(report)
        history.step()
        history.display.end_fold()