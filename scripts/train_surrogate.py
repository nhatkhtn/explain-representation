import logging
from pathlib import Path

import numpy as np
import datasets
import torch
from sklearn.model_selection import train_test_split

from exrep.registry import load_tensor, get_artifact, save_tensor
from exrep.train import train_local_representation

logger = logging.getLogger(__name__)

def train_surrogate_experiment(
    run,
    random_state=42,
    device=None,
):
    assert device is not None, "Please provide a device to run the experiment on."

    embedding_artifact_name = "imagenet-1k-first-20-take-2000_target-embeddings_mocov3-resnet50"
    image_artifact_name = "imagenet-1k-first-20-take-2000_images"
    output_phase_name = "surrogate"

    encoding = load_tensor(
        base_name="imagenet",
        phase="local-encoding",
        identifier="agglomerative",
        file_name=f"local-encoding_{run.config.num_clusters}.pt",
        map_location=device,
        wandb_run=run,
    )
    embeddings = load_tensor(
        "embeddings.pt",
        artifact_name=embedding_artifact_name,
        map_location=device,
        wandb_run=run,
    )
    images_path = get_artifact(
        image_artifact_name,
        wandb_run=run,
    ).download()
    images_dataset = datasets.load_from_disk(images_path)
    labels_dataset = images_dataset.remove_columns(["image"])

    if isinstance(embeddings, list):
        embeddings = torch.cat(embeddings, dim=0)
    embeddings_dataset = datasets.Dataset.from_dict({"targets": embeddings})
    encoding_dataset = datasets.Dataset.from_dict({"inputs": encoding})

    xy_dataset = datasets.concatenate_datasets(
        [encoding_dataset, embeddings_dataset, labels_dataset],
        axis=1
    ).with_format("torch").train_test_split(0.1, shuffle=False, seed=random_state)

    logger.info("Encoding shape: %s", encoding.shape)
    logger.info("Embeddings shape: %s", embeddings.shape)
    logger.info("Image dataset: %s", images_dataset)
    logger.info("XY dataset: %s", xy_dataset)

    model, logs = train_local_representation(
        alpha=0,
        model_config=run.config.surrogate,
        loss_config=run.config.loss,
        optimizer_config=run.config.optimizer,
        train_dataset=xy_dataset["train"],
        val_dataset=xy_dataset["test"],
        keys=embeddings,
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