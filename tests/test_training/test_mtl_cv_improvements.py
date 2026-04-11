"""Focused tests for MTL CV runner improvements."""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from configs.globals import DEVICE
from data.folds import FoldData, TaskFoldData
from losses.registry import create_loss
from tracking.fold import FoldHistory
from training.runners.mtl_cv import train_model


class _TinyMTL(nn.Module):
    def __init__(self, feature_size: int = 4, hidden_size: int = 6, num_classes: int = 3):
        super().__init__()
        self.shared = nn.Linear(feature_size, hidden_size)
        self.cat_head = nn.Linear(hidden_size, num_classes)
        self.next_head = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        category_input, next_input = inputs
        category_rep = self.shared(category_input.squeeze(1))
        next_rep = self.shared(next_input.mean(dim=1))
        return self.cat_head(category_rep), self.next_head(next_rep)

    def shared_parameters(self):
        return self.shared.parameters()

    def task_specific_parameters(self):
        return list(self.cat_head.parameters()) + list(self.next_head.parameters())


class _CountingScheduler:
    def __init__(self):
        self.step_count = 0

    def step(self):
        self.step_count += 1


def _task_data(features: torch.Tensor, targets: torch.Tensor, batch_size: int) -> TaskFoldData:
    train_loader = DataLoader(TensorDataset(features, targets), batch_size=batch_size)
    val_loader = DataLoader(TensorDataset(features[:6], targets[:6]), batch_size=batch_size)
    return TaskFoldData(
        train=FoldData(train_loader, features, targets),
        val=FoldData(val_loader, features[:6], targets[:6]),
    )


def test_mtl_train_model_accumulates_gradients_and_tracks_joint_best():
    torch.manual_seed(42)
    batch_size = 2
    num_samples = 10
    num_classes = 3

    category_x = torch.randn(num_samples, 1, 4)
    next_x = torch.randn(num_samples, 3, 4)
    targets = torch.arange(num_samples) % num_classes

    category_data = _task_data(category_x, targets, batch_size)
    next_data = _task_data(next_x, targets, batch_size)

    model = _TinyMTL(num_classes=num_classes).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = _CountingScheduler()
    step_calls = {"count": 0}
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        step_calls["count"] += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step

    fold_history = FoldHistory.standalone({"next", "category"})
    criterion = nn.CrossEntropyLoss()
    mtl_criterion = create_loss("equal_weight", n_tasks=2, device=DEVICE)

    train_model(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader_next=next_data,
        dataloader_category=category_data,
        next_criterion=criterion,
        category_criterion=criterion,
        mtl_criterion=mtl_criterion,
        num_epochs=1,
        num_classes=num_classes,
        fold_history=fold_history,
        gradient_accumulation_steps=2,
    )

    assert step_calls["count"] == 3
    assert scheduler.step_count == 3
    assert fold_history.model_task.best.best_state
    assert fold_history.model_task.best.best_value > float("-inf")
    assert "pareto_front" in fold_history.artifacts
    assert "grad_cosine_shared" in fold_history.diagnostics
