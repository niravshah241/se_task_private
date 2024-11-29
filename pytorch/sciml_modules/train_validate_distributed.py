import torch
import torch.distributed as dist
import torch.multiprocessing as mp  # noqa: F401


def train_nn(dataloader, model, loss_fn,
             optimizer, report=True, verbose=False):
    """Training of ANN

    Args:
        dataloader (torch.utils.data.Dataloader): Dataloader
        model (torch.nn.Module): Neural net
        loss_fn (torch.nn Loss): Loss function
        optimizer (torch.optim Optimiser): Optimiser
        report (bool, optional): Loss printing. Defaults to True.
        verbose (bool, optional): Net param printing. Defaults to False.

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
        for param in model.parameters():
            dist.barrier()
            if verbose is True:
                print(f"param before all_reduce: {param.grad.data}")

            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

            if verbose is True:
                print(f"param after all_reduce: {param.grad.data}")

        optimizer.step()
        # dist.barrier()
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        optimizer.zero_grad()

        current_size += X.shape[0]
        if report is True and batch % 1 == 0:
            print(f"Loss: {loss.item()} {current_size}/{dataset_size}")
    return loss.item()


def valid_nn(dataloader, model, loss_fn,
             report=True, verbose=False):
    """Validation of ANN

    Args:
        dataloader (torch.utils.data.Dataloader): Dataloader
        model (torch.nn.Module): Neural net
        loss_fn (torch.nn Loss): Loss function
        report (bool, optional): Loss printing. Defaults to True.
        verbose (bool, optional): Net param printing. Defaults to False.

    Returns:
        Float: Loss value
    """
    model.eval()  # NOTE
    valid_loss = torch.tensor([0.])
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            valid_loss += loss_fn(pred, y)
    # dist.barrier()

    if verbose is True:
        print(f"Validation loss before all_reduce: {valid_loss.item(): >7f}")

    dist.all_reduce(valid_loss, op=dist.ReduceOp.SUM)

    if verbose is True:
        print(f"Validation loss after all_reduce: {valid_loss.item(): >7f}")

    if report is True:
        print(f"Validation loss: {valid_loss.item(): >7f}")
    return valid_loss.item()
