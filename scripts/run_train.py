import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    import os
    
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

    torch._logging.set_logs(dynamo=logging.DEBUG)
    
    random_state = 42

    device = "cuda:7"

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

if __name__ == "__main__":
    import signal
    import wandb

    # def handler(signum, frame):
    #     print("Exception handler called!")
    #     wandb.finish(exit_code=1)
    #     raise RuntimeError("Run timeout")
        
    # signal.signal(signal.SIGALRM, handler)

    # signal.alarm(90)

    # try:
    #     main()
    # except RuntimeError as e:
    #     print(e)

    main()