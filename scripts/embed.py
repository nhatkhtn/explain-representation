import argparse
import logging
from pathlib import Path

import datasets
from dotenv import dotenv_values
import wandb
import torch

from exrep.registry import get_artifact, load_model, save_tensor, load_processor
from exrep.utils import generic_map

local_config = dotenv_values(".env")

random_state = 42
run = wandb.init(
    project=local_config["WANDB_PROJECT"],
    config={
        "job_type": "embed_target",
        "target_model": "mocov3-resnet50",
        "processor": "imagenet",
    },
    save_code=True,
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()
    base_dataset_name = args.dataset_name
    device = args.device

    logger.info("Embedding model: %s", run.config.target_model)
    
    dataset_path = get_artifact(
        base_name=base_dataset_name,
        phase="images",
        wandb_run=run,
    ).download()
    image_dataset = datasets.load_from_disk(dataset_path)
    
    model = load_model(run.config.target_model, device)
    processor = load_processor(run.config.processor)

    dataloader = torch.utils.data.DataLoader(
        image_dataset.with_transform(
            lambda x: {'image': processor(x['image'])},
        ),
        batch_size=128,
        num_workers=2,
        prefetch_factor=2,
    )

    # embeddings should always fit in memory with no problem
    embeddings = generic_map(
        model,
        dataloader,
        input_format="positional",
        input_columns=['image'],
        device=device
    )
    
    logger.info("Embeddings shape: %s", embeddings.shape)

    save_tensor(embeddings, 
        base_name=base_dataset_name,
        phase="target-embeddings",
        type="embeddings",
        file_name="embeddings.pt",
        identifier=run.config.target_model,
        wandb_run=run
    )

main()