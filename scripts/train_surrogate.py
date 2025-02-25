import logging
from pathlib import Path

import numpy as np
import datasets
import torch
from sklearn.model_selection import train_test_split

from exrep.registry import load_hf_dataset, load_tensor, get_artifact, save_tensor
from exrep.train import train_local_representation

logger = logging.getLogger(__name__)

def load_imagenet(run, device):
    base_name = 'imagenet'
    encoding_alias = 'latest'

    encoding_artifact = get_artifact(
        base_name=base_name,
        phase="local-encoding",
        identifier="agglomerative",
        alias=encoding_alias,
        wandb_run=run,
    )
    train_encoding = load_tensor(
        f"local-encoding-{run.config.num_clusters}_train.pt",
        map_location=device,
        artifact=encoding_artifact,
    )
    val_encoding = load_tensor(
        f"local-encoding-{run.config.num_clusters}_validation.pt",
        map_location=device,
        artifact=encoding_artifact,
    )
    
    embedding_artifact = get_artifact(
        base_name=base_name,
        phase="target-embeddings",
        identifier='mocov3-resnet50',
        wandb_run=run,
    )
    train_embeddings = load_tensor(
        "embeddings-train.pt",
        map_location=device,
        artifact=embedding_artifact,
    )
    val_embeddings = load_tensor(
        "embeddings-validation.pt",
        map_location=device,
        artifact=embedding_artifact,
    )

    images_dataset = load_hf_dataset(
        base_name=base_name,
        phase='images',
        wandb_run=run,
    )

    train_dataset = datasets.concatenate_datasets([
            datasets.Dataset.from_dict({"inputs": train_encoding}),
            datasets.Dataset.from_dict({"targets": train_embeddings}),
            images_dataset['train'].remove_columns(['image']),
        ],
        axis=1,
    ).with_format("torch")
    val_dataset = datasets.concatenate_datasets([
            datasets.Dataset.from_dict({"inputs": val_encoding}),
            datasets.Dataset.from_dict({"targets": val_embeddings}),
            images_dataset['validation'].remove_columns(['image']),
        ],
        axis=1,
    ).with_format("torch")

    combined_dataset = datasets.DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
    })
    logger.info("Combined dataset: %s", combined_dataset)
    
    return {
        "combined_dataset": combined_dataset,
        "keys_train": train_embeddings,
        "keys_val": val_embeddings,
    }

def train_surrogate_experiment(
    run,
    device=None,
):
    assert device is not None, "Please provide a device to run the experiment on."

    dataset_dict = load_imagenet(run, device)
    xy_dataset = dataset_dict["combined_dataset"]
    keys_train = dataset_dict["keys_train"]
    keys_val = dataset_dict["keys_val"]
    output_phase_name = "surrogate"

    model, logs = train_local_representation(
        alpha=0,
        model_config=run.config.surrogate,
        loss_config=run.config.loss,
        optimizer_config=run.config.optimizer,
        train_dataset=xy_dataset["train"],
        val_dataset=xy_dataset["validation"],
        keys_train=keys_train,
        keys_val=keys_val,
        groups=None,
        wandb_run=run,
        num_epochs=40,
        batch_size=512,
        log_every_n_steps=10,
        device=device,
    )

    save_tensor(
        model.state_dict(),
        f"explainer-{run.config.num_clusters}.pt",
        base_name="imagenet",
        phase=output_phase_name,
        type="model",
        identifier="mocov3-resnet50",
        mode="incremental",
        wandb_run=run,
    )

    return model, logs

if __name__ == "__main__":
    pass