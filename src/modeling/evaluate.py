import torch


def evaluate_model(model, dataloaders, loss_function, device, num_classes=None):
    """
    Unified evaluation function for both validation and testing.

    Args:
        model: The MTLPOI model
        dataloader: DataLoader with evaluation data
        loss_functions: Dictionary of loss functions
        reshape_method: Method to reshape output and target
        foward_method: Method to forward data through the model
        device: Device to run evaluation on
        num_classes: Number of POI classes

    Returns:
        Tuple of (accuracy, loss)
    """
    model.eval()
    total_loss_next = 0.0
    total_loss_category = 0.0
    total_correct_next = 0
    total_correct_category = 0
    total_samples = 0
    batchs = 0

    with torch.no_grad():
        for data_next, data_cat in zip(*dataloaders):
            x_next, y_next = data_next['x'].to(device), data_next['y'].to(device)
            x_category, y_category = data_cat['x'].to(device), data_cat['y'].to(device)

            # Forward pass
            cat_out, next_out = model((x_category, x_next))
            pred_next, truth_next = next_out, y_next
            pred_category, truth_category = cat_out, y_category

            # Calculate loss
            total_loss_next += loss_function(pred_next, truth_next).item()
            total_loss_category += loss_function(pred_category, truth_category).item()

            # Calculate accuracy
            pred_next_class = torch.argmax(pred_next, dim=1)
            pred_category_class = torch.argmax(pred_category, dim=1)
            total_correct_next += (pred_next_class == truth_next).sum().item()
            total_correct_category += (pred_category_class == truth_category).sum().item()
            total_samples += truth_next.size(0)
            batchs += 1

    acc_next = total_correct_next / total_samples
    acc_category = total_correct_category / total_samples
    loss_next = total_loss_next / batchs
    loss_category = total_loss_category / batchs

    return acc_next, loss_next, acc_category, loss_category


def evaluate_model_by_head(model, dataloader, loss_function, foward_method, device, num_classes=None):
    """
    Unified evaluation function for both validation and testing.

    Args:
        model: The MTLPOI model
        dataloader: DataLoader with evaluation data
        loss_functions: Dictionary of loss functions
        reshape_method: Method to reshape output and target
        foward_method: Method to forward data through the model
        device: Device to run evaluation on
        num_classes: Number of POI classes

    Returns:
        Tuple of (accuracy, loss)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data in dataloader:
            x, y = data['x'].to(device), data['y'].to(device)

            # Forward pass
            out = foward_method(x)
            pred, truth = out, y

            # Calculate loss
            loss = loss_function(pred, truth).item()
            total_loss += loss

            # Calculate accuracy
            pred_class = torch.argmax(pred, dim=1)
            total_correct += (pred_class == truth).sum().item()
            total_samples += truth.size(0)

    # Return average accuracy and loss
    return total_correct / total_samples, total_loss / total_samples