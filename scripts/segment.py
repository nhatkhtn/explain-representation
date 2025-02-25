import argparse
from pathlib import Path

from dotenv import dotenv_values
from tqdm import tqdm
import wandb
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from exrep.utils.image import MaskPostProcessor
from exrep.registry import load_hf_dataset, save_hf_dataset

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
        "mask_iou_thresh": 0.8,
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
        # the below does not seem to increase speed
        # hydra_overrides_extra=[
        #     "++model.compile_image_encoder=True",
        # ],
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

    # optimize for speed
    torch.set_float32_matmul_precision('high')

    generator = create_sam2_generator(model_id=run.config.model, device=device)
    transforms = v2.Compose([
        # need to resize here since SAM2 does not work with too large images
        v2.Resize(1170),
        # 1170/1024 is approximately 256/224, which commonly used for ImageNet
        v2.CenterCrop(1024),
        v2.RGB(),
    ])
    dataset = load_hf_dataset(
        base_name=base_dataset_name,
        phase=input_phase_name,
        wandb_run=run,
    ).with_transform(transforms)
    wandb.log({"example": wandb.Image(dataset['train'][0]['image'])})

    def map_fn(inputs, indices):
        image = inputs['image'][0]
        post_processor = MaskPostProcessor(
            image, min_area_ratio=run.config.min_crop_area_ratio, 
            mask_iou_thresh=run.config.mask_iou_thresh,
        )
        outputs = post_processor(generator.generate(np.array(image)))
        num_patches = len(outputs['patches'])
        return outputs | {
            'label': [inputs['label']] * num_patches,
            'image_id': [indices[0]] * num_patches,
            'patch_index': list(range(num_patches)),
        }

    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        mapped_dataset = dataset.map(
            map_fn,
            with_indices=True,
            batched=True,
            batch_size=1,
            remove_columns=['image'],
        )

    mapped_dataset.reset_format()

    save_hf_dataset(
        dataset=mapped_dataset,
        base_name=base_dataset_name,
        phase=output_phase_name,
        mode='write-new',
        wandb_run=run,
    )

if __name__ == "__main__":
    main()
