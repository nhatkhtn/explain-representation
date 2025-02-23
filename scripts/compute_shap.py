import os
import logging
from pathlib import Path

from dotenv import dotenv_values
import wandb
import numpy as np
import torch
import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from exrep.registry import load_data, save_data, load_model, load_tensor, get_artifact, save_tensor

if 'notebooks' in os.getcwd():
    os.chdir("../")

local_config = dotenv_values(".env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random_state = 42

embedding_artifact_name = "imagenet-1k-first-20-take-2000_target-embeddings_mocov3-resnet50"
image_artifact_name = "imagenet-1k-first-20-take-2000_images"
output_phase_name = "surrogate"

run = wandb.init(
    project=local_config["WANDB_PROJECT"],
    config={
        "job_type": "concept_attribution",
        "num_clusters": 80,
    },
    # reinit=True,
    save_code=True,
)

device = "cuda:3"

train_configs = {
    "surrogate": dict(
        output_dim=32,
    ),
    "loss": dict(
        name="KDLoss",
        gamma1=1.0,
        gamma2=1.0,
        temp_student=0.2,
        temp_teacher=1,
    ),
    "optimizer": dict(
        lr=1e-3,
        weight_decay=1e-4,
    )
}
run.config.update(train_configs)

from scripts.train_surrogate import train_surrogate_experiment, train_local_representation

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

from operator import itemgetter
from typing import Optional

def compute_baseline_loss(
    loss_config: dict,
    val_dataset: datasets.Dataset,
    keys: torch.Tensor,
    batch_size: int,
    device: Optional[str] = None,
):
    assert device is not None, "Please provide a device to run the experiment on."
    temp_teacher = loss_config["temp_teacher"]
    losses = []
    with torch.inference_mode():
        for batch in val_dataset.iter(batch_size=batch_size):
            features, targets, labels = itemgetter("inputs", "targets", "label")(batch)
            
            sim_teacher = targets.to(device) @ keys.T      # shape (B x B)
            prob_student = torch.ones_like(sim_teacher, device=device) / sim_teacher.shape[1]

            loss_batch = torch.nn.functional.kl_div(
                input=torch.log(prob_student), 
                target=torch.softmax(sim_teacher / temp_teacher, dim=-1), 
                reduction="batchmean",
            )
            losses.append(loss_batch)
    # technically mean of means is not the same as mean of all losses
    # but in this case it should be fine
    return torch.stack(losses).mean().item()

compute_baseline_loss(run.config.loss, xy_dataset["test"], embeddings, batch_size=512, device=device)

class Nop:
    def nop(*args, **kw): pass
    def __getattr__(self, _): return self.nop

from math import ceil

from functools import partial
import shap
from tqdm import tqdm    

def test_fn(X):
    results = []
    for row in tqdm(X):
        indices = np.where(row == 0)[0]
        # print(np.where(row == 1)[0])
        
        masked_encoding = encoding.clone()
        masked_encoding[:, indices] = 0
        perturbed_encoding_dataset = datasets.Dataset.from_dict({"inputs": masked_encoding})

        perturbed_dataset = datasets.concatenate_datasets(
            [perturbed_encoding_dataset, embeddings_dataset, labels_dataset],
            axis=1
        ).with_format("torch").train_test_split(0.1, shuffle=False, seed=random_state)

        model, logs = train_local_representation(
            alpha=0,
            model_config=run.config.surrogate,
            loss_config=run.config.loss,
            optimizer_config=run.config.optimizer,
            train_dataset=perturbed_dataset["train"],
            val_dataset=perturbed_dataset["test"],
            keys=embeddings,
            groups=None,
            eval_downstream=False,
            wandb_run=Nop(),
            num_epochs=40,
            batch_size=512,
            log_every_n_steps=0,
            device=device,
        )
        best_val_loss = min(log["val_loss"] for log in logs["val"])
        # logger.info("Best validation loss: %s", best_val_loss)
        results.append(best_val_loss)
    return np.array(results)

shap_explainer = shap.KernelExplainer(test_fn, np.zeros((1, encoding.shape[1])))
shap_values = shap_explainer.shap_values(np.ones((encoding.shape[1], )), nsamples=200)

np.save(f"outputs/shap-values-{run.config.num_clusters}.npy", shap_values)

print(np.round(shap_values / shap_values.sum(), 3))