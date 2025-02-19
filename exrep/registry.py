import logging
import os
import shutil
import pickle
from pathlib import Path
from typing import Literal, Optional

from dotenv import dotenv_values, find_dotenv
import torch
import datasets
from transformers import pipeline, AutoImageProcessor
import wandb
import torchvision.models as torchvision_models
import torchvision.transforms.v2 as v2

local_config = dotenv_values(find_dotenv(".env"))
project_name = local_config["WANDB_PROJECT"]
tmp_dir = Path("tmp/")

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
    
def get_artifact(
    artifact_name: Optional[str]=None,
    base_name: Optional[str]=None,
    phase: Optional[str]=None,
    identifier: Optional[str]=None,
    type: Optional[str]=None,
    metadata: Optional[dict]=None,
    mode: Optional[Literal["write-new", "incremental"]]="write-new",
    alias="latest",
    wandb_run=None,
):
    assert wandb_run is not None, "wandb_run must be provided"
    assert mode in ["write-new", "incremental"], "mode must be one of 'write-new' or 'incremental'"

    if artifact_name is None:
        assert base_name is not None and phase is not None, "Either artifact_name or base_name and phase must be provided"
        base_name = alias_lookup(base_name)
        if identifier is None:
            artifact_name = f"{base_name}_{phase}"
        else:
            artifact_name = f"{base_name}_{phase}_{identifier}"

    api = wandb.Api()
    if api.artifact_exists(f"{project_name}/{artifact_name}:{alias}"):
        if type is not None or metadata is not None:
            logger.warning("Artifact %s:%s already exists, but type and metadata are provided", artifact_name, alias)

        artifact = wandb_run.use_artifact(f"{project_name}/{artifact_name}:{alias}")
        if mode == "incremental":
            artifact = artifact.new_draft()
        return artifact
    
    if metadata is None:
        metadata = {}

    logger.info("Creating artifact %s:%s on wandb", artifact_name, alias)
    if type is None:
        raise ValueError("type must be provided when creating a new artifact")
    artifact = wandb.Artifact(
        artifact_name,
        type=type,
        metadata={
            "base_name": base_name,
            "phase": phase,
        } | metadata,
    )
    return artifact

def save_file(
    file_path: str | Path,
    file_name: str,
    alias="latest",
    overwrite=False,
    wandb_run=None,
    artifact=None,
    finalize=True,
    **kwargs,
):
    if artifact is None:
        artifact = get_artifact(
            alias=alias,
            wandb_run=wandb_run,
            **kwargs,
        )
    if isinstance(file_path, Path):
        file_path = file_path.as_posix()
    artifact.add_file(
        local_path=file_path,
        name=file_name,
        overwrite=overwrite
    )
    if finalize:
        return wandb_run.log_artifact(artifact, aliases=[alias])
    return artifact

def save_dir(
    local_path: str | Path,
    alias="latest",
    wandb_run=None,
    artifact=None,
    finalize=True,
    **kwargs
):
    if artifact is None:
        artifact = get_artifact(alias=alias, wandb_run=wandb_run, **kwargs)
    if isinstance(local_path, Path):
        local_path = local_path.as_posix()
    artifact.add_dir(local_path)
    if finalize:
        return wandb_run.log_artifact(artifact, aliases=[alias])
    return artifact

def load_entry(
    file_name: str,
    **kwargs,
):
    artifact = get_artifact(**kwargs)
    entry_path = artifact.get_entry(file_name).download()
    return entry_path

# below are convenience functions
# based on the above base functions

def save_tensor(
    tensor: torch.Tensor,
    file_name: str,
    **kwargs,
):
    local_path = tmp_dir / file_name
    torch.save(tensor, local_path)
    return save_file(
        file_path=local_path,
        file_name=file_name,
        **kwargs,
    )

def load_tensor(
    file_name: str,
    map_location=None,
    **kwargs,
):
    if map_location is None:
        raise ValueError("Must specify map_location when loading tensors")
    file_path = load_entry(
        file_name=file_name,
        **kwargs,
    )
    tensor = torch.load(file_path, map_location=map_location)
    return tensor

def save_pickle(
    obj,
    file_name: str,
    **kwargs,
):
    local_path = tmp_dir / file_name
    with open(local_path, "wb") as f:
        pickle.dump(obj, f)
    return save_file(
        file_path=local_path,
        file_name=file_name,
        **kwargs,
    )

def save_data(
    dataset: datasets.Dataset,
    dir_name: str,
    **kwargs
):
    local_dir = tmp_dir / dir_name
    if local_dir.exists():
        raise FileExistsError(f"Directory {local_dir} already exists")
    dataset.save_to_disk(local_dir)
    artifact = save_dir(local_path=local_dir, **kwargs)
    logger.info("Removing local temp directory %s", local_dir)
    shutil.rmtree(local_dir)
    return artifact

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