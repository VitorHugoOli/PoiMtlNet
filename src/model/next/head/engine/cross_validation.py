from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from configs.globals import DEVICE

from model.next.head.configs.next_config import CfgNextModel, CfgNextHyperparams, CfgNextTraining
from model.next.head.engine.evaluation import evaluate
from model.next.head.engine.trainer import train
from model.next.head.modeling.next_head import NextHead
from utils.ml_history.metrics import MLHistory
from utils.ml_history.parms.neural import NeuralParams


def run_cv(
        history: MLHistory,
        folds: list[tuple[DataLoader, DataLoader]],
):
    """Run cross-validation for the model."""
    for idx, (train_loader, val_loader) in enumerate(folds):
        history.display.start_fold()
        model = NextHead(
            embed_dim=CfgNextModel.INPUT_DIM,
            num_classes=CfgNextModel.NUM_CLASSES,
            num_heads=CfgNextModel.NUM_HEADS,
            seq_length=CfgNextModel.MAX_SEQ_LENGTH,
            num_layers=CfgNextModel.NUM_LAYERS,
            dropout=CfgNextModel.DROPOUT,
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()

        optimizer = AdamW(
            model.parameters(),
            lr=CfgNextHyperparams.LR,
            weight_decay=CfgNextHyperparams.WEIGHT_DECAY,

        )

        scheduler = OneCycleLR(
            optimizer=optimizer,
            max_lr=CfgNextHyperparams.MAX_LR,
            epochs=CfgNextTraining.EPOCHS,
            steps_per_epoch=len(train_loader),
        )

        history.set_model_arch(str(model))

        history.set_model_parms(
            NeuralParams(
                batch_size=CfgNextTraining.BATCH_SIZE,
                num_epochs=CfgNextTraining.EPOCHS,
                learning_rate=CfgNextHyperparams.LR,
                optimizer=optimizer.__class__.__name__,
                optimizer_state=optimizer.state_dict(),
                scheduler=scheduler.__class__.__name__,
                scheduler_state=scheduler.state_dict(),
                criterion={
                    'next': criterion.__class__.__name__},
                criterion_state={
                    'next': criterion.state_dict()
                }

            )
        )

        train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            history.get_curr_fold(),
            DEVICE,
        )

        report = evaluate(model, val_loader, DEVICE)

        history.get_curr_fold().to('next').add_report(report)
        history.step()
        history.display.end_fold()
