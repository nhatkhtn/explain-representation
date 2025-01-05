import logging
import collections
from typing import Callable, Iterable, Sequence

from PIL import Image
import numpy as np
import torch
from transformers.tokenization_utils_base import BatchEncoding

from exrep.kd import KDLoss, distill_one_epoch

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

class Embedder:
    def __init__(self, 
                 inference_fn: Callable[[torch.Tensor | TensorDict], torch.Tensor], 
                 preprocess_fn: Callable[[Iterable[Image.Image]], torch.Tensor], 
                 batch_size: int = 256
        ):
        self.inference_fn = inference_fn
        self.preprocess_fn = preprocess_fn
        self.batch_size = batch_size

    def __call__(self, x: Sequence[Image.Image] | torch.Tensor, normalize=True) -> torch.Tensor:
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
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings

class KDRegressor:
    def __init__(self, 
        n_output_dim: int,
        temp_student: float = 1.0,
        temp_teacher: float = 1.0,
        num_epochs: int = 10,
        batch_size: int = 256,
        device: str = "cuda",
    ):
        self.n_output_dim = n_output_dim
        self.temp_student = temp_student
        self.temp_teacher = temp_teacher
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.logs = []
        self.query_encoder = None
        self.key_encoder = None

    def fit(self, data: np.ndarray, embeddings: np.ndarray, sample_weight: np.ndarray):
        """Fit the regressor to the data.

        Args:
            data (np.ndarray): NumPy array of shape (n_samples, n_patches) containing the data with local features.
            embeddings (np.ndarray): NumPy array of shape (n_samples, n_repr_dim) containing the embedded data (representations).
            sample_weights (np.ndarray): NumPy array of shape (n_samples,) containing the sample weights.
        """
        # get the constants
        n_samples, n_local_features = data.shape
        n_repr_dim = embeddings.shape[1]
        n_output_dim = self.n_output_dim
        assert embeddings.shape[0] == n_samples, "Number of samples in data and embeddings must match."
        
        # create models
        # student model (e')
        self.query_encoder = torch.nn.Linear(n_local_features, n_output_dim).to(self.device)
        # projector (r)
        self.key_encoder = torch.nn.Linear(n_repr_dim, n_output_dim).to(self.device)

        # create optimizer
        params = list(self.query_encoder.parameters()) + list(self.key_encoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-1)

        # create loss
        sample_weight = torch.tensor(sample_weight)
        # use fixed sample weight for now
        sample_weight = torch.ones_like(sample_weight)
        logger.warning("Using fixed sample weight for now.")
        loss = KDLoss(
            data_size=n_samples, gamma1=0.9, gamma2=0.9,
            weights=sample_weight,
            temp_student=self.temp_student, temp_teacher=self.temp_teacher,
        ).to(self.device)

        # load data
        indices = torch.arange(n_samples)
        dataset = torch.utils.data.StackDataset(images=torch.Tensor(data), indices=indices)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # distill
        for epoch in range(self.num_epochs):
            logs = distill_one_epoch(
                self.query_encoder, self.key_encoder, embeddings, 
                dataloader, loss, optimizer, epoch, self.device, 
                log_every_n_steps=5,
            )
            self.logs.extend(logs)

        # intercept_ is (n_features,)
        # self.intercept_ = self.query_encoder.bias.detach().cpu().numpy()
        self.intercept_ = None
        # coef_ is (n_features, n_patches)
        # self.coef_ = self.query_encoder.weight.detach().cpu().numpy()
        self.coef_ = []
        
    def score(self, X, y, sample_weight=None):
        logging.warning("Score is not implemented.")
        return None
        # y_pred = self.predict(X)
        # return np.linalg.norm(y - y_pred, axis=1)

    def predict(self, x: np.ndarray) -> np.ndarray:
        logger.warning("Predict is not implemented.")
        return None
        # assert x.ndim == 2
        # # x is (n_samples, n_patches)
        # return x @ self.coef_.T + self.intercept_
    
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds the input data with the query encoder."""
        return self.query_encoder(x)
    
    def project(self, x: np.ndarray) -> np.ndarray:
        """Projects an embedding to the output space with the key encoder."""
        with torch.inference_mode():
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            return self.key_encoder(x).detach().cpu().numpy()