import logging
import collections
from typing import Any, Callable, Iterable, Sequence

from PIL import Image
import numpy as np
import torch
from transformers.tokenization_utils_base import BatchEncoding
from transformers import pipeline

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

class Embedder:
    def __init__(self, 
                 inference_fn: Callable[[torch.Tensor | TensorDict], torch.Tensor], 
                 preprocess_fn: Callable[[Iterable[Image.Image | np.ndarray]], torch.Tensor | TensorDict], 
                 batch_size: int = 256
        ):
        self.inference_fn = inference_fn
        self.preprocess_fn = preprocess_fn
        self.batch_size = batch_size

    def __call__(self, x: Sequence[Image.Image | np.ndarray] | torch.Tensor | TensorDict, normalize=True) -> torch.Tensor:
        with torch.inference_mode():
            # if passed in a sequence of images, preprocess them
            if isinstance(x, collections.abc.Sequence) and isinstance(x[0], (Image.Image, np.ndarray)):
                x = self.preprocess_fn(x)

            # if passed in a numpy array, preprocess it
            elif isinstance(x, np.ndarray):
                x = self.preprocess_fn(x)

            # if passed in a dictionary or BatchEncoding, assume the input is already preprocessed
            if isinstance(x, (dict, BatchEncoding)):
                dataset = torch.utils.data.StackDataset(**x)
            else:
                dataset = x
            
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
            embeddings = []
            for batch in dataloader:
                with torch.inference_mode():
                    outputs = self.inference_fn(batch)
                embeddings.append(outputs)
            embeddings = torch.cat(embeddings)
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings
        
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