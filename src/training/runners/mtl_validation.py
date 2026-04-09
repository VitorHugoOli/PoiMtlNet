import torch
from sklearn.metrics import classification_report

from utils.progress import zip_longest_cycle
from configs.globals import DEVICE

def validation_best_model(data_next,
                          data_category,
                          best_next,
                          best_category,
                          model):
    """Validate best models for both tasks using zip_longest_cycle (matches training coverage)."""

    all_pred_next = []
    all_truth_next = []
    all_pred_category = []
    all_truth_category = []

    with torch.no_grad():
        model.load_state_dict(best_next)
        model.eval()
        for batch_next, batch_category in zip_longest_cycle(data_next, data_category):
            x_next, y_next = batch_next
            x_next, y_next = x_next.to(DEVICE, non_blocking=True), y_next.to(DEVICE, non_blocking=True)
            x_category, _ = batch_category
            x_category = x_category.to(DEVICE, non_blocking=True)
            out_category, out_next = model((x_category, x_next))
            pred_next, truth_next = out_next, y_next
            pred_next_class = torch.argmax(pred_next, dim=1)
            all_pred_next.append(pred_next_class.cpu())
            all_truth_next.append(truth_next.cpu())

        model.load_state_dict(best_category)
        model.eval()
        for batch_next, batch_category in zip_longest_cycle(data_next, data_category):
            x_next, _ = batch_next
            x_next = x_next.to(DEVICE, non_blocking=True)
            x_category, y_category = batch_category
            x_category, y_category = x_category.to(DEVICE, non_blocking=True), y_category.to(DEVICE, non_blocking=True)
            out_category, out_next = model((x_category, x_next))
            pred_category, truth_category = out_category, y_category
            pred_category_class = torch.argmax(pred_category, dim=1)
            all_pred_category.append(pred_category_class.cpu())
            all_truth_category.append(truth_category.cpu())


    pred_next = torch.cat(all_pred_next)
    truth_next = torch.cat(all_truth_next)
    pred_category = torch.cat(all_pred_category)
    truth_category = torch.cat(all_truth_category)

    # Generate classification reports
    report_next = classification_report(truth_next, pred_next, output_dict=True,zero_division=0)
    report_category = classification_report(truth_category, pred_category, output_dict=True,zero_division=0)
    return report_next, report_category