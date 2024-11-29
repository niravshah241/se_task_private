import torch


def train_nn(dataloader, model, loss_fn, optimizer,
             report=True):
    """Training of ANN

    Args:
        dataloader (torch.utils.data.Dataloader): Dataloader
        model (torch.nn.Module): Neural net
        loss_fn (torch.nn Loss): Loss function
        optimizer (torch.optim Optimiser): Optimiser
        report (bool, optional): Loss printing. Defaults to True.

    Returns:
        Float: Loss value
    """
    dataset_size = len(dataloader.dataset)
    current_size = 0
    model.train()  # NOTE
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        current_size += X.shape[0]
        if report is True and batch % 1 == 0:
            print(f"Loss: {loss.item()} {current_size}/{dataset_size}")
    return loss.item()


def valid_nn(dataloader, model, loss_fn,
             report=True):
    """Validation of ANN

    Args:
        dataloader (torch.utils.data.Dataloader): Dataloader
        model (torch.nn.Module): Neural net
        loss_fn (torch.nn Loss): Loss function
        report (bool, optional): Loss printing. Defaults to True.

    Returns:
        Float: Loss value
    """
    model.eval()  # NOTE
    valid_loss = torch.tensor([0.])
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            valid_loss += loss_fn(pred, y).item()
    if report is True:
        print(f"\n Validation loss: {valid_loss.item(): >7f}")
    return valid_loss
