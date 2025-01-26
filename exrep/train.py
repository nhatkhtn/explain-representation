import logging
from operator import itemgetter
from typing import Sequence

import torch

from exrep.loss import KDLoss
from exrep.model import LocalRepresentationApproximator
from exrep.utils import pythonize

logger = logging.getLogger(__name__)

def train_local_representation(
    model_config: dict,
    loss_config: dict,
    optimizer_config: dict,
    query_inputs: torch.Tensor,
    query_targets: torch.Tensor,
    keys: torch.Tensor,
    groups: Sequence[Sequence[int]],
    alpha: float,
    num_epochs: int = 10,
    batch_size: int = 256,
    log_every_n_steps: int = 10,
    device: str = "cuda",
):
    """Train local representation with student-teacher similarity matching.

    Args:
        model_config (dict): model configuration.
        loss_config (dict): loss configuration.
        optimizer_config (dict): optimizer configuration.
        query_inputs (torch.Tensor): tensor of shape (n_queries, n_features) containing inputs in local feature space.
        query_references (torch.Tensor): tensor of shape (n_queries, n_repr_dim) containing inputs in the original (teacher) space.
        keys (torch.Tensor): tensor of shape (n_keys, n_repr_dim) containing keys in the teacher space.
        groups (Sequence[Sequence[int]]): groups for group lasso regularization.
        alpha (float): regularization parameter.
        num_epochs (int, optional): number of epochs. Defaults to 10.
        batch_size (int, optional): batch size. Defaults to 256.
        device (str, optional): device. Defaults to "cuda".
    """
    # infer configuration based on input
    model_config["local_dim"] = query_inputs.shape[1]
    model_config["repr_dim"] = query_targets.shape[1]
    loss_config["data_size"] = query_inputs.shape[0]

    # create the model, optimizer, and loss
    model = LocalRepresentationApproximator(device=device, **model_config)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
    loss_fn = KDLoss(**loss_config)

    # verify data
    n_queries = query_inputs.shape[0]
    assert n_queries == query_targets.shape[0], "Query inputs and targets must have the same number of samples"

    # move data to device
    keys = keys.to(device)
    query_inputs = query_inputs.to(device)
    query_targets = query_targets.to(device)

    # create dataloaders
    indices = torch.arange(n_queries)
    query_dataset = torch.utils.data.StackDataset(inputs=torch.Tensor(query_inputs), indices=indices)
    query_dataloader = torch.utils.data.DataLoader(query_dataset, batch_size=batch_size, shuffle=True)


    model.train()
    logs = []
    for epoch in range(num_epochs):
        for i, batch in enumerate(query_dataloader):
            images, indices = itemgetter("inputs", "indices")(batch)
            images = images.to(device)

            queries_student = model.encode(query=images)
            keys_student = model.encode(key=keys)

            loss_dict = loss_fn(queries_student, keys_student, query_targets[indices], keys, indices)
            
            loss_regularize = alpha * model.get_regularization_term(groups)
            total_loss = loss_dict["grad_estimator"] + loss_regularize

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if i % log_every_n_steps == 0:
                logger.info("Epoch %2d, Step %4d, Loss: %.5f", epoch, i, total_loss.item())
            logs.append({"epoch": epoch, "step": i, "loss_reg": loss_regularize.item()} | pythonize(loss_dict))

    return model, logs