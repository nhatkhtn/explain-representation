import logging
from pathlib import Path

import numpy as np
import datasets
import torch
import torchvision.transforms.v2 as v2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, StackDataset, TensorDataset

from exrep.dataset import HFDatasetWrapper, StackDictDataset
from exrep.registry import load_hf_dataset, load_processor, load_tensor, get_artifact, save_tensor, load_model, imagenet_norm_transform
from exrep.train import train_local_representation

logger = logging.getLogger(__name__)

def load_data_from_runs(run, batch_size=1024, device=None):
    assert device is not None, "Device needs to be specified"
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
        map_location='cpu',
        artifact=encoding_artifact,
    ).float()
    val_encoding = load_tensor(
        f"local-encoding-{run.config.num_clusters}_validation.pt",
        map_location='cpu',
        artifact=encoding_artifact,
    ).float()

    # this is based on ImageNet and 224 input size
    # see https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/main_moco.py#L262
    # for typical contrastive learning augmentation
    # here, we use a more conservative one
    train_transform = v2.Compose([
        # might want to use BICUBIC here
        v2.RandomResizedCrop(224, scale=(0.8, 1.)),
        v2.RandomHorizontalFlip(),
        # v2.Resize(256),
        # v2.CenterCrop(224),
        v2.RGB(),
        v2.ToTensor(),
        imagenet_norm_transform
    ])
    # see https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/main_lincls.py#L288
    # for ImageNet validation transform
    val_transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.RGB(),
        v2.ToTensor(),
        imagenet_norm_transform
    ])

    images_dataset = load_hf_dataset(
        base_name=base_name,
        phase='images',
        wandb_run=run,
    )

    train_dataloader = DataLoader(
        StackDictDataset(
            HFDatasetWrapper(images_dataset['train'].with_transform(train_transform)), 
            StackDataset(
                inputs=train_encoding,
                indices=torch.arange(len(train_encoding)),
            )
        ),
        batch_size=batch_size,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        pin_memory_device=device,
    )
    val_dataloader = DataLoader(
        StackDictDataset(
            HFDatasetWrapper(images_dataset['validation'].with_transform(val_transform)), 
            StackDataset(
                inputs=val_encoding,
                indices=torch.arange(len(val_encoding)),
            )
        ),
        batch_size=512,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        pin_memory_device=device,
    )
    data_sizes = {
        'train': len(images_dataset['train']),
        'validation': len(images_dataset['validation']),
    }
    return train_dataloader, val_dataloader, data_sizes

def train_surrogate_experiment(
    run,
    save=True,
    device=None,
):
    assert device is not None, "Please provide a device to run the experiment on."

    train_dataloader, val_dataloader, data_sizes = load_data_from_runs(run,
        batch_size=run.config.training['batch_size'], 
        device=device
    )
    output_phase_name = "surrogate"

    model, logs = train_local_representation(
        model_config=run.config.surrogate,
        loss_config=run.config.loss,
        optimizer_config=run.config.optimizer,
        target_config=run.config.target,
        data_sizes=data_sizes,
        train_dataset=train_dataloader,
        val_dataset=val_dataloader,
        wandb_run=run,
        num_epochs=run.config.training['epochs'],
        log_every_n_steps=1,
        device=device,
    )

    if save:
        save_tensor(
            model.state_dict(),
            f"explainer-{run.config.num_clusters}.pt",
            base_name="imagenet",
            phase=output_phase_name,
            type="model",
            identifier=run.config.target_model,
            mode="write-new",
            wandb_run=run,
        )

    return model, logs

if __name__ == "__main__":
    import wandb
    from dotenv import dotenv_values

    local_config = dotenv_values(".env")

    random_state = 42

    output_phase_name = "surrogate"

    run = wandb.init(
        project=local_config["WANDB_PROJECT"],
        config={
            "job_type": "train_representation",
            "num_clusters": 40,
        },
        # reinit=True,
        save_code=True,
    )

    device = "cuda:7"

    train_configs = {
        "target": dict(
            name='mocov3',
            variant='resnet50',
        ),
        "surrogate": dict(
            output_dim=256,
            use_key_encoder=False,
        ),
        "loss": dict(
            name="KDLossNaive",
            gamma1=1.0,
            gamma2=1.0,
            temp_student=0.01,
            temp_teacher=0.01,
        ),
        "optimizer": dict(
            lr=1e-3,
            weight_decay=1e-4,
        ),
        "training": dict(
            query_batch_size=32,
            key_batch_size=512,
            epochs=200,
        ),
    }
    run.config.update(train_configs)

    from scripts.train_surrogate import train_surrogate_experiment

    model, logs = train_surrogate_experiment(run, device=device, save=True)