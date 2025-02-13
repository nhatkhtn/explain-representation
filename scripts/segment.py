import argparse
from pathlib import Path

from dotenv import dotenv_values
import datasets
from tqdm import tqdm
import wandb
import numpy as np
import torch
import torchvision
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from exrep.utils import MaskPostProcessor
from exrep.registry import load_data, save_data

local_config = dotenv_values(".env")

# name of input dataset
base_dataset_name = "imagenet"
input_phase_name = "images"
output_phase_name = "crops"

run = wandb.init(
    project=local_config["WANDB_PROJECT"],
    config={
        "segmentor": "SAM2",
        "model": "facebook/sam2.1-hiera-base-plus",
        "job_type": "data_segment",
        "min_crop_area_ratio": 0.005,
    },
    save_code=True,
)

# sadly the postprocessing is still not working
# the installation is fine but running the postprocessing code gives incorrect results
# large gives very bad results
def create_sam2_generator(model_id, device):
    pred_im_size = 1024
    generator = SAM2AutomaticMaskGenerator.from_pretrained(
        model_id=model_id,
        points_per_side=32,
        points_per_batch=1024,
        pred_iou_thresh=0.85,
        # min_mask_region_area=pred_im_size * pred_im_size * 0.01,
        hydra_overrides_extra=[
            "++model.compile_image_encoder=True",
        ],
        device=device,
    )
    assert generator.predictor._transforms.resolution == pred_im_size
    return generator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str)
    args = parser.parse_args()
    device = args.device
    assert device.startswith("cuda") or device == "cpu", f"Must use cuda or cpu, got {device}"

    generator = create_sam2_generator(model_id=run.config.model, device=device)
    # need to resize here since SAM2 does not work with too large images
    resize_transform = torchvision.transforms.Resize((1024, 1024))
    dataset = load_data(
        base_name=base_dataset_name,
        phase=input_phase_name,
        load_local=True,
    ).map(
        lambda x: { 'image': resize_transform(x['image']) },
        num_proc=4,
    ).cast_column(
        "image", datasets.Image('RGB')
    )

    def map_fn(inputs, indices):
        image = inputs['image'][0]
        post_processor = MaskPostProcessor(image, min_area_ratio=run.config.min_crop_area_ratio)
        outputs = post_processor(generator.generate(np.array(image)))
        return {
            'crops': outputs,
            'label': [inputs['label']] * len(outputs),
            'index': [f"{indices[0]}_{j}" for j in range(len(outputs))],
        }

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        mapped_dataset = dataset.map(
            map_fn,
            with_indices=True,
            batched=True,
            batch_size=1,
            remove_columns=['image'],
        )

    save_data(
        dataset=mapped_dataset,
        base_name=base_dataset_name,
        phase=output_phase_name,
        wandb_run=run,
    )

main()
