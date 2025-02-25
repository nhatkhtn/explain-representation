import logging
import collections
from operator import itemgetter
from typing import Any, Callable, Iterable, Literal, Sequence

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from transformers import pipeline
from transformers.tokenization_utils_base import BatchEncoding
from transformers.image_processing_base import BatchFeature

TensorDict = dict[str, torch.Tensor] | BatchEncoding

logger = logging.getLogger(__name__)

def torch_pairwise_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the pairwise cosine similarity between two tensors.

    Args:
        x (torch.Tensor): Tensor of shape (n_samples, n_features).
        y (torch.Tensor): Tensor of shape (n_samples, n_features).

    Returns:
        torch.Tensor: Tensor of shape (n_samples, n_samples) containing the pairwise cosine similarity.
    """
    return torch.nn.functional.cosine_similarity(x[:,:,None], y.t()[None,:,:])  

def euclidean_to_cosine_similarity_kernel(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Kernel recovering the cosine similarity from the euclidean distance

    Args:
        X (torch.Tensor): tensor of shape (N, D)
        Y (torch.Tensor): tensor of shape (M, D)

    Returns:
        torch.Tensor: tensor of shape (N, M)
    """
    return 1 - torch.cdist(X, Y, p=2) ** 2 / 2

def pythonize(d: dict[Any, torch.Tensor]) -> dict[str, Any]:
    """Convert a dictionary of torch single-element tensors to a dictionary of python objects.
    
    Mostly used for logging purposes."""
    return {key: value.item() for key, value in d.items()}

def default_comb_fn(accu, new):
    if accu is None:
        return new
    else:
        assert type(accu) == type(new), "Got {} and {}".format(type(accu), type(new))
        if isinstance(accu, torch.Tensor):
            return torch.cat([accu, new], dim=0)
        elif isinstance(accu, (BatchFeature)):
            for k in accu.keys():
                accu[k] = torch.cat([accu[k], new[k]], dim=0)
            return accu
        else:
            raise TypeError(f"Unsupported type {type(accu)}")

def generic_map(
    func, dataset, comb_fn=default_comb_fn, 
    pre_proc_fn=lambda x: x,
    post_proc_fn=lambda x: x,
    input_columns: list[str]=None,
    input_format: Literal["positional", "keyword"]="positional", device=None
):
    results = None
    for inputs in tqdm(dataset):
        inputs = pre_proc_fn(inputs)
        if isinstance(inputs, dict):
            inputs = BatchFeature(inputs)
        if device is not None:
            inputs = inputs.to(device)
        with torch.inference_mode():
            if input_format == "keyword":
                outputs = func(**inputs)
            else:
                if input_columns is not None:
                    inputs = itemgetter(*input_columns)(inputs)
                outputs = func(inputs)
        outputs = post_proc_fn(outputs)
        results = comb_fn(results, outputs)
    return results
        
def validate_mask_area(image: Image.Image, mask: torch.Tensor | np.ndarray, min_area_ratio=0.01, max_area_ratio=0.6):
    im_area = image.size[0] * image.size[1]
    mask_area = mask.sum()
    return max_area_ratio * im_area > mask_area > min_area_ratio * im_area
        
# mask generator that returns a list of masks
def get_sam_mask_generator(
    model="facebook/sam-vit-huge", points_per_batch=64, device=None,
    mask_validator=validate_mask_area,
    **pipeline_kwargs
):
    if device is None:
        raise ValueError("device must be specified")
    # see available args at https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/mask_generation.py#L94
    # see args value at https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/automatic_mask_generator.py#L36
    generator = pipeline("mask-generation",
        model=model,
        points_per_batch=points_per_batch,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        stability_score_offset=1,
        crops_n_layers=0,
        crops_nms_thresh=0.7,
        crop_overlap_ratio=512 / 1500,
        crop_n_points_downscale_factor=1,
        device=device,
        **pipeline_kwargs
    )
    def gen_fn(images: list[Image.Image], **kwargs):
        batch = generator(images, **kwargs)
        masks = [[
                mask for mask in outputs['masks'] if mask_validator(image, mask)
            ]
            for image, outputs in zip(images, batch)
        ]
        return masks
    return gen_fn

class Nop:
    def nop(*args, **kw): pass
    def __getattr__(self, _): return self.nop