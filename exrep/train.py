import logging
from operator import itemgetter
from functools import partial
from typing import Sequence

import torch

from exrep.loss import init_loss
from exrep.model import init_target, LocalRepresentationApproximator
from exrep.utils import pythonize

logger = logging.getLogger(__name__)

def train_local_representation(
    model_config: dict,
    loss_config: dict,
    optimizer_config: dict,
    target_config: dict,
    data_sizes: dict,
    train_dataset: torch.utils.data.DataLoader,
    val_dataset: torch.utils.data.DataLoader,
    wandb_run=None,
    num_epochs: int = 10,
    log_every_n_steps: int = 10,
    device: str = "cuda",
):
    """Train local representation with student-teacher similarity matching.

    Args:
        model_config (dict): model configuration.
        loss_config (dict): loss configuration.
        optimizer_config (dict): optimizer configuration.
        target_config (dict): target model configuration.
        query_inputs_train (torch.Tensor): tensor of shape (n_queries, n_features) containing inputs in local feature space.
        query_targets_train (torch.Tensor): tensor of shape (n_queries, n_repr_dim) containing inputs in the original (teacher) space.
        query_inputs_val (torch.Tensor): tensor of shape (n_queries, n_features) containing inputs in local feature space.
        query_targets_val (torch.Tensor): tensor of shape (n_queries, n_repr_dim) containing inputs in the original (teacher) space.
        keys (torch.Tensor): tensor of shape (n_keys, n_repr_dim) containing keys in the teacher space.
        num_epochs (int, optional): number of epochs. Defaults to 10.
        device (str, optional): device. Defaults to "cuda".
    """
    # init the model to be explained
    target_model = init_target(**target_config, device=device)

    # infer configuration based on input
    sample_batch = next(iter(val_dataset))
    model_config["local_dim"] = sample_batch["inputs"].shape[1]
    with torch.inference_mode():
        _, _, sample_embedding = target_model.self_sim(sample_batch["image"].to(device))
        model_config["repr_dim"] = sample_embedding.shape[1]
    model_config["temperature"] = loss_config["temp_student"]
    train_loss_config = loss_config.copy()
    validation_loss_config = loss_config.copy()
    train_loss_config["data_size"] = data_sizes['train']
    validation_loss_config["data_size"] = data_sizes['validation']

    # create the model, optimizer, and loss
    model = LocalRepresentationApproximator(device=device, **model_config)
    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_config)
    train_loss_fn = init_loss(**train_loss_config)
    val_loss_fn = init_loss(**validation_loss_config)
    
    # main training loop
    logs = {"train": [], "val": []}
    steps = 0
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataset:
            # obtain inputs
            features, images, labels, indices = itemgetter("inputs", "image", "label", "indices")(batch)
            features = features.to(device)
            images = images.to(device)

            # compute queries and keys
            with torch.no_grad():
                queries_teacher, keys_teacher, key_embeddings = target_model.self_sim(images)
            # queries_student, keys_student = model(features, key_embeddings)
            queries_student, keys_student = model(features, keys_teacher)

            # compute losses
            loss_dict = train_loss_fn(queries_student, keys_student, queries_teacher, keys_teacher, indices)
            total_loss = loss_dict["grad_estimator"]

            # update model
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # logging
            steps += 1
            if log_every_n_steps > 0 and steps % log_every_n_steps == 0:
                logger.info("Epoch %2d, Step %4d, Loss: %.5f, Acc: %.2f", epoch, steps, total_loss.item(), loss_dict["accuracy"])
            logs["train"].append({"epoch": epoch, "step": steps} | pythonize(loss_dict))
            wandb_run.log(logs["train"][-1], step=steps)

        # validation loop
        model.eval()
        val_losses = []
        with torch.inference_mode():
            for batch in val_dataset:
                features, images, labels, indices = itemgetter("inputs", "image", "label", "indices")(batch)
                features = features.to(device)
                images = images.to(device)
                
                # here everything is inference mode
                queries_teacher, keys_teacher, key_embeddings = target_model.self_sim(images)
                queries_student, keys_student = model(features, keys_teacher)
                # queries_student, keys_student = model(features, key_embeddings)
                
                loss_dict = val_loss_fn(queries_student, keys_student, queries_teacher, keys_teacher, indices)
                val_losses.append(loss_dict)

        val_loss_dict = {
            f"val_{k}": torch.stack([d[k] for d in val_losses]).mean().item()
            for k in val_losses[0].keys()
        }
        logs["val"].append({"epoch": epoch, "step": steps} | val_loss_dict)

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