import logging
from operator import itemgetter
from functools import partial
from typing import Sequence, Optional

import numpy as np
import torch
from tqdm.notebook import tqdm
import datasets
from sklearn.svm import SVC

from exrep.loss import init_loss
from exrep.model import LocalRepresentationApproximator
from exrep.utils import pythonize

logger = logging.getLogger(__name__)

def train_local_representation(
    model_config: dict,
    loss_config: dict,
    optimizer_config: dict,
    train_dataset: datasets.Dataset,
    val_dataset: datasets.Dataset,
    keys_train: torch.Tensor,
    keys_val: torch.Tensor,
    groups: Optional[Sequence[Sequence[int]]],
    alpha: float,
    eval_downstream=True,
    wandb_run=None,
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
    sample_input = train_dataset[0]
    model_config["local_dim"] = sample_input["inputs"].shape[0]
    model_config["repr_dim"] = sample_input["targets"].shape[0]
    model_config["temperature"] = loss_config["temp_student"]
    train_loss_config = loss_config.copy()
    validation_loss_config = loss_config.copy()
    train_loss_config["data_size"] = len(train_dataset)
    validation_loss_config["data_size"] = len(val_dataset)

    # create the model, optimizer, and loss
    model = LocalRepresentationApproximator(device=device, **model_config)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
    train_loss_fn = init_loss(**train_loss_config)
    val_loss_fn = init_loss(**validation_loss_config)

    # prepare data
    keys_train = keys_train.to(device)
    keys_val = keys_val.to(device)
    train_dataset = train_dataset.add_column("indices", np.arange(len(train_dataset)))
    val_dataset = val_dataset.add_column("indices", np.arange(len(val_dataset)))

    # main training loop
    logs = {"train": [], "val": []}
    steps = 0
    for epoch in range(num_epochs):
        model.train()
        train_features, train_targets, train_labels = [], [], []
        for batch in train_dataset.iter(batch_size=batch_size):
            features, targets, labels, indices = itemgetter("inputs", "targets", "label", "indices")(batch)
            features = features.to(device)
            targets = targets.to(device)
            labels = labels.to(device)

            queries_student = model.encode(query=features)
            keys_student = model.encode(key=keys_train)

            train_features.append(model.encode(query=features, normalize=True))
            train_targets.append(targets)
            train_labels.append(labels)

            loss_dict = train_loss_fn(queries_student, keys_student, targets, keys_train, indices)
            loss_regularize = alpha * model.get_regularization_term(groups)
            total_loss = loss_dict["grad_estimator"] + loss_regularize.to(device)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            steps += 1
            if log_every_n_steps > 0 and steps % log_every_n_steps == 0:
                logger.info("Epoch %2d, Step %4d, Loss: %.5f", epoch, steps, total_loss.item())
            logs["train"].append({"epoch": epoch, "step": steps, "loss_reg": loss_regularize.item()} | pythonize(loss_dict))
            wandb_run.log(logs["train"][-1], step=steps)

        # validation
        model.eval()
        val_losses = []
        
        val_features, val_targets, val_labels = [], [], []
        with torch.inference_mode():
            keys_student = model.encode(key=keys_val)
            for batch in val_dataset.iter(batch_size=batch_size):
                features, targets, labels, indices = itemgetter("inputs", "targets", "label", "indices")(batch)
                features = features.to(device)
                targets = targets.to(device)
                labels = labels.to(device)
                queries_student = model.encode(query=features)
                val_features.append(model.encode(query=features, normalize=True))
                val_targets.append(targets)
                val_labels.append(labels)
                loss_dict = val_loss_fn(queries_student, keys_student, targets, keys_val, indices)
                val_losses.append(loss_dict)

        val_loss_dict = {
            f"val_{k}": torch.stack([d[k] for d in val_losses]).mean().item() 
            for k in val_losses[0].keys()
        }
        logs["val"].append({"epoch": epoch, "step": steps} | val_loss_dict)

        # downstream evaluation
        if eval_downstream:
            train_features = torch.cat(train_features, dim=0).detach().cpu().numpy()
            train_targets = torch.cat(train_targets, dim=0).detach().cpu().numpy()
            train_labels = torch.cat(train_labels, dim=0).detach().cpu().numpy()
            val_features = torch.cat(val_features, dim=0).detach().cpu().numpy()
            val_targets = torch.cat(val_targets, dim=0).detach().cpu().numpy()
            val_labels = torch.cat(val_labels, dim=0).detach().cpu().numpy()
            classifier = SVC(kernel="linear").fit(train_features, train_labels)
            train_acc = classifier.score(train_features, train_labels)
            val_acc = classifier.score(val_features, val_labels)
            logs["val"][-1]["classify_train_acc"] = train_acc
            logs["val"][-1]["classify_val_acc"] = val_acc

        wandb_run.log(logs["val"][-1], step=steps)

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