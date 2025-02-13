import os
from pathlib import Path

from dotenv import dotenv_values
import wandb
import numpy as np
import torch
import datasets

from exrep.registry import load_data, save_data, load_model

random_state = 42
local_save_dir = Path("data")
if 'notebooks' in os.getcwd():
    os.chdir("../")

local_config = dotenv_values(".env")

run = wandb.init(
    project=local_config["WANDB_PROJECT"],
    config={
        "job_type": "data_preprocessing",
        "dataset": "imagenet-1k-val",
        "subset": "first-20"
    },
    save_code=True,
)
ds_name = f"{run.config.dataset}-{run.config.subset}"

def main():
    dataset = datasets.load_dataset(
        "imagenet-1k",
        split="validation",
    ).filter(
        lambda x: x['label'] < 20,
        keep_in_memory=True,
        num_proc=4
    )

    save_data(
        dataset=dataset,
        base_name=ds_name,
        phase="images",
        wandb_run=run,
    )

main()