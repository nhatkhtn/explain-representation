from pathlib import Path

from dotenv import dotenv_values
import wandb
import numpy as np
import torch
import datasets
from sklearn.model_selection import train_test_split

from exrep.registry import load_data, save_data, load_model
from exrep.train import train_local_representation

random_state = 42
local_save_dir = Path("data")

local_config = dotenv_values(".env")
base_dataset_name = "imagenet"
encoding_phase_name = "encoding"
embeddings_phase_name = "target-embeddings"
output_phase_name = "surrogate"

run = wandb.init(
    project=local_config["WANDB_PROJECT"],
    config={
        "job_type": "train_representation",
    },
    save_code=True,
)

def main():
    encoding_dataset = load_data(
        base_name=base_dataset_name,
        phase=encoding_phase_name,
        load_local=True,
    )
    embeddings_dataset = load_data(
        base_name=base_dataset_name,
        phase=embeddings_phase_name,
        load_local=True,
    )

    query_inputs_train, query_inputs_val, query_targets_train, query_targets_val = train_test_split(
        replace_traits, replace_sample_embeddings, test_size=0.1, random_state=random_state, shuffle=False
    )

    model_categorical, logs = train_local_representation(
        alpha=0,
        model_config=model_config,
        loss_config=loss_config,
        optimizer_config=optimizer_config,
        query_inputs_train=query_inputs_train,
        query_targets_train=query_targets_train,
        query_inputs_val=query_inputs_val,
        query_targets_val=query_targets_val,
        keys=keys,
        groups=None,
        num_epochs=20,
        batch_size=512,
        log_every_n_steps=10,
        device=device,   
    )
    

main()