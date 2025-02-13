import logging
from operator import itemgetter
from functools import partial
from typing import Sequence, Optional

import numpy as np
import torch
from tqdm.notebook import tqdm
import datasets

from exrep.loss import KDLoss
from exrep.model import LocalRepresentationApproximator
from exrep.utils import pythonize

logger = logging.getLogger(__name__)

def train_local_representation(
    model_config: dict,
    loss_config: dict,
    optimizer_config: dict,
    train_dataset: datasets.Dataset,
    val_dataset: datasets.Dataset,
    keys: torch.Tensor,
    groups: Optional[Sequence[Sequence[int]]],
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
    sample_input = train_dataset[0]
    model_config["local_dim"] = sample_input["inputs"].shape[0]
    model_config["repr_dim"] = sample_input["targets"].shape[0]
    train_loss_config = loss_config.copy()
    validation_loss_config = loss_config.copy()
    train_loss_config["data_size"] = len(train_dataset)
    validation_loss_config["data_size"] = len(val_dataset)

    # create the model, optimizer, and loss
    model = LocalRepresentationApproximator(device=device, **model_config)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
    train_loss_fn = KDLoss(**train_loss_config)

    # prepare data
    keys = keys.to(device)
    train_dataset = train_dataset.add_column("indices", np.arange(len(train_dataset)))
    val_dataset = val_dataset.add_column("indices", np.arange(len(val_dataset)))

    def compute_validation_loss():
        model.eval()
        val_loss_fn = KDLoss(**validation_loss_config)
        losses = []
        with torch.inference_mode():
            keys_student = model.encode(key=keys)
            for batch in val_dataset.iter(batch_size=batch_size):
                features, targets, indices = itemgetter("inputs", "targets", "indices")(batch)
                features = features.to(device)
                targets = targets.to(device)
                queries_student = model.encode(query=features)
                loss_dict = val_loss_fn(queries_student, keys_student, targets, keys, indices)
                losses.append(loss_dict)
        agg_loss = {k: torch.stack([d[k] for d in losses]).mean() for k in losses[0].keys()}
        return {"val_" + k: v for k, v in agg_loss.items()}

    # main training loop
    logs = {"train": [], "val": []}
    steps = 0
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataset.iter(batch_size=batch_size):
            features, targets, indices = itemgetter("inputs", "targets", "indices")(batch)
            features = features.to(device)
            targets = targets.to(device)

            queries_student = model.encode(query=features)
            keys_student = model.encode(key=keys)

            loss_dict = train_loss_fn(queries_student, keys_student, targets, keys, indices)
            
            loss_regularize = alpha * model.get_regularization_term(groups)
            total_loss = loss_dict["grad_estimator"] + loss_regularize.to(device)

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

def concept_shap(
    query_inputs_train: torch.Tensor,
    query_inputs_val: torch.Tensor,
    num_samples=1000,
    **kwargs,
):
    """Compute the SHAP values indicating the contribution of each concept to the local representation.
    
    Args:
        query_inputs_train (torch.Tensor): tensor of shape (n_queries, n_features) containing inputs in local feature space.
        query_inputs_val (torch.Tensor): tensor of shape (n_queries, n_features) containing inputs in local feature space.
        num_samples (int, optional): number of samples. Defaults to 1000.
    """
    num_features = query_inputs_train.shape[1]
    train_fn = partial(train_local_representation,
        groups=None, alpha=0,
        log_every_n_steps=0, 
        **kwargs
    )

    rng = np.random.default_rng(seed=42)
    all_runs = []

    shap_values = np.zeros(num_features)
    
    _, logs_base = train_fn(
        query_inputs_train=torch.zeros_like(query_inputs_train),
        query_inputs_val=torch.zeros_like(query_inputs_val),
    )
    val_loss_base = logs_base["val"][-1]["val_loss"]

    for f in tqdm(range(num_features)):
        concepts_masks = rng.choice(2, size=(num_samples, num_features))
        concepts_masks[:, f] = 1
        game_values = np.zeros(num_samples)

        for i, concept_mask in tqdm(enumerate(concepts_masks)):
            concept_indices = np.where(concept_mask == 1)[0]
            
            model_with, logs_with = train_fn(
                query_inputs_train=query_inputs_train[:, concept_indices],
                query_inputs_val=query_inputs_val[:, concept_indices],
            )
            # value = 0 when val_loss equals the base loss
            # value = 1 when val_loss is 0
            value_with = (val_loss_base - logs_with["val"][-1]["val_loss"]) / val_loss_base

            concept_mask[f] = 0
            model_without, logs_without = train_fn(
                query_inputs_train=query_inputs_train[:, concept_indices],
                query_inputs_val=query_inputs_val[:, concept_indices],
            )
            value_without = (val_loss_base - logs_without["val"][-1]["val_loss"]) / val_loss_base

            game_values[i] = value_with - value_without
            all_runs.append({
                "concept_mask": concept_mask,
                "concept_index": f,
                "game_value": game_values[i],
                "logs_with": logs_with,
                "logs_without": logs_without,
                "model_with": model_with,
                "model_without": model_without,
            })
        shap_values[f] = game_values.mean()

    return shap_values, all_runs