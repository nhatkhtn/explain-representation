import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values, find_dotenv
import torch
import datasets
from transformers import pipeline, AutoImageProcessor
import wandb
import torchvision.models as torchvision_models
import torchvision.transforms.v2 as v2

local_config = dotenv_values(find_dotenv(".env"))

logger = logging.getLogger(__name__)

def load_model(model_id: str, device: str, batch_size=256, **kwargs):
    if model_id == "mocov3-resnet50":
        return load_mocov3("resnet50", "r-50-100ep.pth.tar").to(device)

    return pipeline("image-feature-extraction", model=model_id, framework="pt",
        device=device, batch_size=batch_size, return_tensors=True, 
        pool=True, **kwargs,
    )

def load_processor(processor_id: str, **kwargs):
    if processor_id == "imagenet":
        transform = v2.Compose([
            v2.ToImage(),
            v2.RGB(),
            v2.Resize(224),
            v2.CenterCrop(224),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform
    
    return AutoImageProcessor.from_pretrained(processor_id, **kwargs)

def alias_lookup(base_name: str):
    if base_name == "imagenet":
        return "imagenet-1k-first-20-take-2000"
    return base_name

def load_data(artifact_name: Optional[str]=None, base_name: Optional[str]=None, phase: Optional[str]=None, alias="latest", wandb_run=None, load_local=False):
    if artifact_name is None:
        base_name = alias_lookup(base_name)
        artifact_name = f"{base_name}_{phase}"
        
    if load_local:
        dataset_dir = Path(local_config["DATA_DIR"]) / artifact_name
        logger.info("Loading data from local data dir %s", dataset_dir)
        dataset = datasets.load_from_disk(dataset_dir)
        return dataset
    
    else:
        logger.info("Loading data from artifact %s:%s on wandb", artifact_name, alias)
        artifact = wandb_run.use_artifact(f"{artifact_name}:{alias}")
        dataset_dir = artifact.download()
        logger.info("Artifact downloaded to %s", dataset_dir)
        dataset = datasets.load_from_disk(dataset_dir)
        return dataset
    
def save_data(dataset: datasets.Dataset, base_name: str, phase: str, alias="latest", artifact_name=None, wandb_run=None):
    base_name = alias_lookup(base_name)
    artifact_name = f"{base_name}_{phase}"

    local_dir = Path(local_config["DATA_DIR"]) / artifact_name
    dataset.save_to_disk(local_dir)

    artifact = wandb.Artifact(
        artifact_name,
        type="huggingface_dataset",
        metadata={
            "base_name": base_name,
            "phase": phase,
        },
    )
    artifact.add_dir(local_dir)
    wandb_run.log_artifact(artifact, aliases=[alias])

def save_tensor(tensor: torch.Tensor, base_name: str, phase: str, model_name: str, alias="latest", wandb_run=None):
    base_name = alias_lookup(base_name)
    model_name = model_name.replace("/", "-")
    artifact_name = f"{base_name}_{phase}_{model_name}"
    local_path = (Path(local_config["DATA_DIR"]) / artifact_name).with_suffix(".pt")

    logger.info("Saving tensor to local data dir and wandb %s:%s", local_path, alias)
    torch.save(tensor, local_path)

    artifact = wandb.Artifact(
        artifact_name,
        type="torch_tensor",
        metadata={
            "base_name": base_name,
            "phase": phase,
            "model_name": model_name,
        },
    )
    artifact.add_file(local_path.as_posix())
    wandb_run.log_artifact(artifact, aliases=[alias])

def load_tensor(artifact_name: str, alias="latest", wandb_run=None):
    logger.info("Loading tensor from artifact %s:%s on wandb", artifact_name, alias)
    artifact = wandb_run.use_artifact(f"{artifact_name}:{alias}")
    file_name = f"{artifact_name}.pt"
    file_path = artifact.get_entry(file_name).download()
    tensor = torch.load(file_path)
    return tensor

def load_mocov3(arch, pretrained_path):
    # create model
    print("=> creating model '{}'".format(arch))
    if arch.startswith('vit'):
        model = vits.__dict__[arch]()
        linear_keyword = 'head'
    else:
        model = torchvision_models.__dict__[arch]()
        linear_keyword = 'fc'

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
            param.requires_grad = False
    
    # load from pre-trained, before DistributedDataParallel constructor
    if os.path.isfile(pretrained_path):
        print("=> loading checkpoint '{}'".format(pretrained_path))
        checkpoint = torch.load(pretrained_path, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

        print("=> loaded pre-trained model '{}'".format(pretrained_path))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained_path))

    model.fc = torch.nn.Identity()
    return model