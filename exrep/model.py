from typing import Optional, Sequence

import torch

class LocalRepresentationApproximator(torch.nn.Module):
    def __init__(self, local_dim: int, repr_dim: int, output_dim: int, temperature: float, use_key_encoder=True, device: str = "cuda"):
        super().__init__()
        self.temperature = temperature
        self.query_encoder = torch.nn.Linear(local_dim, output_dim, device=device)
        if use_key_encoder:
            self.key_encoder = torch.nn.Linear(repr_dim, output_dim, device=device)
        else:
            self.key_encoder = torch.nn.Identity()

    def forward(self, queries: torch.Tensor, keys: torch.Tensor):
        """Compute the similarity logits between queries and keys.
        Args:
            queries: A tensor of shape (query_size, local_dim).
            keys: A tensor of shape (key_size, repr_dim).
        Returns:
            A tensor of shape (query_size, key_size) representing the similarity logits.
        """
        query_repr = self.query_encoder(queries)
        key_repr = self.key_encoder(keys)
        logits = torch.nn.functional.softmax(query_repr @ key_repr.T / self.temperature, dim=1)
        return logits
    
    def encode(self, *, query: Optional[torch.Tensor] = None, key: Optional[torch.Tensor] = None):
        if query is not None and key is not None:
            return (self.query_encoder(query), self.key_encoder(key))
        elif query is not None:
            return self.query_encoder(query)
        elif key is not None:
            return self.key_encoder(key)
        else:
            raise ValueError("At least one of query or key must be provided.")
        
    def get_regularization_term(self, groups: Sequence[Sequence[int]]):
        """Returns the group lasso regularization term.
        Args:
            groups: A list of groups, where each group is a list of indices.
        
        Returns:
            A scalar representing the group lasso regularization term.
        """
        # weights have shape (output_dim, input_dim), as in https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        return torch.sum(torch.stack(
            [torch.norm(self.query_encoder.weight[:, indices], p=2) 
            for indices in groups]
        ))