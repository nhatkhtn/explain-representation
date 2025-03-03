

def train_surrogate_sweep():
    import os
    import logging
    from pathlib import Path

    from dotenv import dotenv_values
    import wandb
    import numpy as np
    import datasets
    import torch
    from sklearn.model_selection import train_test_split
    from matplotlib import pyplot as plt

    from exrep.registry import load_data, save_data, load_model, load_tensor, get_artifact
    from exrep.train import train_local_representation

    local_config = dotenv_values(".env")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    random_state = 42

    device = "cuda:0"

    # WARNING: this variable also appears in the main function
    sweep_project_name = "sweep-exrep-downstream1"

    embedding_artifact_name = "imagenet-1k-first-20-take-2000_target-embeddings_mocov3-resnet50"
    image_artifact_name = "imagenet-1k-first-20-take-2000_images"
    output_phase_name = "surrogate"

    run = wandb.init(
        project=sweep_project_name,
        config={
            "job_type": "train_representation",
            "num_clusters": 80,
        },
        # reinit=True,
        # save_code=True,
    )

    encoding = load_tensor(
        base_name="imagenet",
        phase="local-encoding",
        identifier="agglomerative",
        file_name=f"local-encoding_{run.config.num_clusters}.pt",
        wandb_run=run,
    )
    embeddings = load_tensor(
        "embeddings.pt",
        artifact_name=embedding_artifact_name,
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

if __name__ == "__main__":
    import wandb
    sweep_config = {
        "method": "random",
        # note that random search disregards the metric
        # but we still specify it for bookkeeping
        "metric": {
            "goal": "minimize",
            "name": "val_loss",
        },
        "parameters": {
            "surrogate": {
                "parameters": {
                    "output_dim": {
                        "min": 16,
                        "max": 320,
                    },
                },
            },
            "loss": {
                "parameters": {
                    "gamma1": { "values": [1.0] },
                    "gamma2": { "values": [1.0] },
                    "temp_student": {
                        "distribution": "log_uniform_values",
                        "min": 0.01,
                        "max": 10,
                    },
                    "temp_teacher": {
                        "distribution": "log_uniform_values",
                        "min": 0.01,
                        "max": 10,
                    }
                },
            },
            "optimizer": {
                "parameters": {
                    "lr": { "values": [1e-3] },
                    "weight_decay": {
                        "distribution": "log_uniform_values",
                        "min": 1e-5,
                        "max": 1e-3,
                    }
                },
            },
        },
    }

    sweep_id = wandb.sweep(sweep_config, project="sweep-exrep-downstream1")
    wandb.agent(sweep_id, function=train_surrogate_sweep, count=5)