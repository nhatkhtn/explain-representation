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

    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        return self.query_encoder(x)
    
    def encode_key(self, x: torch.Tensor) -> torch.Tensor:
        return self.key_encoder(x)

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor):
        return self.encode_query(x_q), self.encode_key(x_k)