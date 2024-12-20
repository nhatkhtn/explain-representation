import collections
from typing import Callable, Iterable, Sequence

from PIL import Image
import numpy as np
import torch
from transformers.tokenization_utils_base import BatchEncoding

TensorDict = dict[str, torch.Tensor] | BatchEncoding

class Embedder:
    def __init__(self, 
                 inference_fn: Callable[[torch.Tensor | TensorDict], torch.Tensor], 
                 preprocess_fn: Callable[[Iterable[Image.Image]], torch.Tensor], 
                 batch_size: int = 256
        ):
        self.inference_fn = inference_fn
        self.preprocess_fn = preprocess_fn
        self.batch_size = batch_size

    def __call__(self, x: Sequence[Image.Image] | torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            if isinstance(x, collections.abc.Sequence) and isinstance(x[0], Image.Image):
                x = self.preprocess_fn(x)

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
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings
