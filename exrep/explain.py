import torch

def train_local_model(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    model_args: dict,
    device,
):
        