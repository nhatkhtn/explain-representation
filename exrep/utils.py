import logging
import collections
from typing import Any, Callable, Iterable, Sequence

from PIL import Image
import numpy as np
import torch
from transformers.tokenization_utils_base import BatchEncoding

from exrep.kd import KDLossNaive, distill_one_epoch

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
        self.keys = None

    def set_keys(self, keys: np.ndarray):
        self.keys = torch.tensor(keys, device=self.device, dtype=torch.float32)

    def fit(self, data: np.ndarray, embeddings: np.ndarray, sample_weight: np.ndarray):
        """Fit the regressor to the (query) data.

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
        assert self.keys is not None, "Keys must be set before fitting the model."

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
        loss = KDLossNaive(
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
                self.query_encoder, self.key_encoder, 
                embeddings, self.keys,
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

class LocalRepresentationFitter:
    def __init__(self,
        n_output_dim: int,
        temp_student: float = 1.0,
        temp_teacher: float = 1.0,
        alpha: float = 0.0,
        num_epochs: int = 10,
        batch_size: int = 256,
        device: str = "cuda",
    ):
        self.n_output_dim = n_output_dim
        self.temp_student = temp_student
        self.temp_teacher = temp_teacher
        self.alpha = alpha
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.logs = []
        self.query_encoder = None
        self.key_encoder = None

    def fit(self, inputs: np.ndarray, targets: np.ndarray, keys: np.ndarray, sample_weight: np.ndarray, feature_groups: Sequence[Sequence[int]]):
        """Fit the regressor to the (query) data.

        Args:
            inputs (np.ndarray): NumPy array of shape (n_samples, n_features) containing the data with local features.
            targets (np.ndarray): NumPy array of shape (n_samples, n_repr_dim) containing the embedded data (representations).
            keys (np.ndarray): NumPy array of shape (n_samples, n_output_dim) containing the keys.
            sample_weights (np.ndarray): NumPy array of shape (n_samples,) containing the sample weights.
            feature_groups (Sequence[Sequence[int]]): Sequence of sequences containing the feature groups, used for regularization.
        """
        # get the constants
        n_samples, n_local_features = inputs.shape
        n_repr_dim = targets.shape[1]
        n_output_dim = self.n_output_dim
        assert targets.shape[0] == n_samples, "Number of samples in inputs and targets must match."

        # create models
        # student model (e')
        self.query_encoder = torch.nn.Linear(n_local_features, n_output_dim).to(self.device)
        # projector (r)
        self.key_encoder = torch.nn.Linear(n_repr_dim, n_output_dim).to(self.device)

        # create optimizer
        params = list(self.query_encoder.parameters()) + list(self.key_encoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-1)

        sample_weight = torch.tensor(sample_weight)
        # use fixed sample weight for now
        sample_weight = torch.ones_like(sample_weight)

        # create loss
        logger.warning("Using fixed sample weight for now.")
        loss = KDLossNaive(
            data_size=n_samples, gamma1=0.9, gamma2=0.9,
            weights=sample_weight,
            temp_student=self.temp_student, temp_teacher=self.temp_teacher,
        ).to(self.device)
        def regularizer(layer):
            return self.alpha * torch.sum(torch.stack([torch.linalg.vector_norm(layer.weight[:, group]) for group in feature_groups]))

        # load data
        indices = torch.arange(n_samples)
        dataset = torch.utils.data.StackDataset(images=torch.Tensor(inputs), indices=indices)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        keys = torch.tensor(keys, device=self.device, dtype=torch.float32)

        # distill
        for epoch in range(self.num_epochs):
            logs = distill_one_epoch(
                self.query_encoder, self.key_encoder, 
                targets, keys,
                dataloader, loss, regularizer, optimizer, epoch, self.device, 
                log_every_n_steps=5,
            )
            self.logs.extend(logs)

        return self
    
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds the input data with the query encoder."""
        return self.query_encoder(x)
    
    def project(self, x: np.ndarray) -> np.ndarray:
        """Projects an embedding to the output space with the key encoder."""
        with torch.inference_mode():
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            return self.key_encoder(x).detach().cpu().numpy()

    def get_informative_groups(self, groups: Sequence[Sequence[int]], norm_threshold=0.1) -> Sequence[Sequence[int]]:
        """Get the most informative feature groups.

        Args:
            groups (Sequence[Sequence[int]]): Sequence of sequences containing the feature groups.

        Returns:
            Sequence[Sequence[int]]: Sequence of sequences containing the most informative feature groups.
        """
        return [
            group for group in groups if torch.linalg.vector_norm(self.query_encoder.weight[:, group]) > norm_threshold
        ]    
        
class LocalRepresentationExplainer:
    def __init__(self):
        pass

    def fit_instance(self):
        pass

    def generate_local_data(self):
        pass

    def explain_downstream(self):
        pass