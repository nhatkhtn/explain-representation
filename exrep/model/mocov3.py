from functools import partial
from typing import Literal

import torch
import torch.nn as nn
import torchvision.models as torchvision_models

class MoCoV3(nn.Module):
    def __init__(self, model: "MoCo"):
        super().__init__()

        # separate the projector from the base encoder
        self.projector = model.base_encoder.fc
        self.predictor = model.predictor

        # remove the fc (i.e., projector) from the base encoder
        model.base_encoder.fc = nn.Identity()
        # the base encoder is what used in the downstream tasks
        self.base_encoder = model.base_encoder

        # TODO: remove this
        self.model = model

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_encoder(x)

    def encode_query(self, h: torch.Tensor) -> torch.Tensor:
        # return self.predictor(self.projector(self.embed(x)))
        # we did not find significant differences between using the predictor and not using it
        return self.projector(h)
    
    def encode_key(self, h: torch.Tensor) -> torch.Tensor:
        return self.projector(h)
    
    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor):
        """Compute the query and key for the given inputs. Also returns the embeddings of the keys.
        
        Args:
            x_q (torch.Tensor): query input
            x_k (torch.Tensor): key input
            
        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): query, key, and embedding
        """
        q = self.encode_query(self.embed(x_q))
        key_embedding = self.embed(x_k)
        k = self.encode_key(key_embedding)
        return q, k, key_embedding

    def self_sim(self, x: torch.Tensor):
        """Optimized version of forward when x is both the query and the key.
        
        Args:
            x (torch.Tensor): input tensor
        
        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): query, key, and embedding
        """
        embedding = self.embed(x)
        q = self.encode_query(embedding)
        k = self.encode_key(embedding)
        return q, k, embedding

    @classmethod
    def from_pretrained(cls, path: str, variant: Literal['resnet50', 'vit'], device: str):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = checkpoint['state_dict']

        if variant == 'resnet50':
            # see https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/main_moco.py#L185C30-L185C41
            model = MoCo_ResNet(partial(torchvision_models.__dict__["resnet50"], zero_init_residual=True))

            # remove prefix
            for k in list(state_dict.keys()):
                state_dict[k.removeprefix("module.")] = state_dict.pop(k)

            # here we use strict loading to reduce the need for validation
            model.load_state_dict(state_dict, strict=True)
            return cls(model).to(device)
        elif variant == 'vit':
            raise NotImplementedError("ViT variant not implemented yet")
        else:
            raise ValueError(f"Unknown variant {variant}")


# MoCo model definitions untouched from https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/moco/builder.py#L11

class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)


class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output