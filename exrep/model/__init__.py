from .surrogate import LocalRepresentationApproximator
from .mocov3 import MoCoV3

def init_target(name: str, **kwargs):
    """Initialize the model to be explained."""
    if name == 'mocov3':
        model = MoCoV3.from_pretrained('r-50-100ep.pth.tar', **kwargs)
        # this is not necessary if we use the right context manager
        # but it is for good measure
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        return model
    else:
        raise ValueError(f"Unknown target model {name}")