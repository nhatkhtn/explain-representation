import logging
from operator import itemgetter
from functools import partial
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
    query_inputs_train: torch.Tensor,
    query_targets_train: torch.Tensor,
    query_inputs_val: torch.Tensor,
    query_targets_val: torch.Tensor,
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
        query_inputs_train (torch.Tensor): tensor of shape (n_queries, n_features) containing inputs in local feature space.
        query_targets_train (torch.Tensor): tensor of shape (n_queries, n_repr_dim) containing inputs in the original (teacher) space.
        query_inputs_val (torch.Tensor): tensor of shape (n_queries, n_features) containing inputs in local feature space.
        query_targets_val (torch.Tensor): tensor of shape (n_queries, n_repr_dim) containing inputs in the original (teacher) space.
        keys (torch.Tensor): tensor of shape (n_keys, n_repr_dim) containing keys in the teacher space.
        groups (Sequence[Sequence[int]]): groups for group lasso regularization.
        alpha (float): regularization parameter.
        num_epochs (int, optional): number of epochs. Defaults to 10.
        batch_size (int, optional): batch size. Defaults to 256.
        device (str, optional): device. Defaults to "cuda".
    """
    # infer configuration based on input
    model_config["local_dim"] = query_inputs_train.shape[1]
    model_config["repr_dim"] = query_targets_train.shape[1]
    train_loss_config = loss_config.copy()
    validation_loss_config = loss_config.copy()
    train_loss_config["data_size"] = query_inputs_train.shape[0]
    validation_loss_config["data_size"] = query_inputs_val.shape[0]

    # create the model, optimizer, and loss
    model = LocalRepresentationApproximator(device=device, **model_config)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
    train_loss_fn = KDLoss(**train_loss_config)

    # verify data
    n_queries = query_inputs_train.shape[0]
    assert n_queries == query_targets_train.shape[0], "Query inputs and targets must have the same number of samples"

    # move data to device
    keys = keys.to(device)
    query_inputs_train = query_inputs_train.to(device)
    query_targets_train = query_targets_train.to(device)
    query_inputs_val = query_inputs_val.to(device)
    query_targets_val = query_targets_val.to(device)

    # create dataloaders
    indices = torch.arange(n_queries)
    query_dataset = torch.utils.data.StackDataset(inputs=torch.Tensor(query_inputs_train), indices=indices)
    query_dataloader = torch.utils.data.DataLoader(query_dataset, batch_size=batch_size, shuffle=True)

    def compute_validation_loss():
        model.eval()
        val_loss_fn = KDLoss(**validation_loss_config)
        with torch.inference_mode():
            queries_student = model.encode(query=query_inputs_val)
            keys_student = model.encode(key=keys)
            indices = torch.arange(query_inputs_val.shape[0])
            loss_dict = val_loss_fn(queries_student, keys_student, query_targets_val, keys, indices)
        return {"val_" + k: v for k, v in loss_dict.items()}

    # main training loop
    logs = {"train": [], "val": []}
    steps = 0
    for epoch in range(num_epochs):
        model.train()
        for _, batch in enumerate(query_dataloader):
            images, indices = itemgetter("inputs", "indices")(batch)
            images = images.to(device)

            queries_student = model.encode(query=images)
            keys_student = model.encode(key=keys)

            loss_dict = train_loss_fn(queries_student, keys_student, query_targets_train[indices], keys, indices)
            
            loss_regularize = alpha * model.get_regularization_term(groups)
            total_loss = loss_dict["grad_estimator"] + loss_regularize

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            steps += 1
            if log_every_n_steps > 0 and steps % log_every_n_steps == 0:
                logger.info("Epoch %2d, Step %4d, Loss: %.5f", epoch, steps, total_loss.item())
            logs["train"].append({"epoch": epoch, "step": steps, "loss_reg": loss_regularize.item()} | pythonize(loss_dict))
        logs["val"].append({"epoch": epoch, "step": steps} | pythonize(compute_validation_loss()))

    return model, logs

def regularization_path(
    alphas: Sequence[float],
    repeat_per_alpha: int,
    **kwargs,
):
    """Train multiple local representations with different regularization strengths.
    
    Args:
        alphas (Sequence[float]): regularization strengths.
        repeat_per_alpha (int): number of repetitions per alpha.
        **kwargs: other arguments to `train_local_representation`.

    For other parameter details, see `train_local_representation`.
    """
    train_fn = partial(train_local_representation, log_every_n_steps=0, **kwargs)
    raw_data, alpha_data = [], []
    for alpha in alphas:
        logger.info("Training with alpha = %s", alpha)
        
        all_norms, all_val_losses = [], []
        for _ in range(repeat_per_alpha):
            model, logs = train_fn(alpha=alpha)
            norms = model.query_encoder.weight.detach().norm(p=2, dim=0).cpu()
            all_norms.append(norms)
            all_val_losses.append(logs["val"][-1]["val_loss"])
            raw_data.append({"alpha": alpha, "model": model, "logs": logs})

        all_norms = torch.stack(all_norms)
        mean_norms = all_norms.mean(dim=0)
        std_norms = all_norms.std(dim=0)
        mean_val_loss = torch.tensor(all_val_losses).mean()
        std_val_loss = torch.tensor(all_val_losses).std()

        alpha_data.append({
            "alpha": alpha, 
            "mean_weights": mean_norms, "std_weights": std_norms, 
            "mean_val_loss": mean_val_loss, "std_val_loss": std_val_loss
        })
    return alpha_data, raw_data