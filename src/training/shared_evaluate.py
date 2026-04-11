from typing import Union

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             best_state: dict = None
             ) -> Union[str, dict]:
    model.eval()
    if best_state is not None:
        model.load_state_dict(best_state)
    preds_list, truths_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            logits = model(X_batch)
            preds_list.append(logits.argmax(dim=1))
            truths_list.append(y_batch)

    # Single GPU→CPU transfer for all predictions
    preds = torch.cat(preds_list).cpu().numpy()
    truths = torch.cat(truths_list).cpu().numpy()

    report = classification_report(
        truths,
        preds,
        output_dict=True,
        zero_division=0,
    )
    return report
