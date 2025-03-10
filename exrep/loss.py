import logging
from typing import Literal, Sequence

import torch

class SimilarityLoss(torch.nn.Module):
    """Base class for teacher-student similarity losses."""
    def __init__(self, temp_student: float = 1.0, temp_teacher: float = 1.0):
        super().__init__()
        self.temp_student = temp_student
        self.temp_teacher = temp_teacher
                 
    @staticmethod
    def get_scaled_sim(queries: torch.Tensor, keys: torch.Tensor, temp: float):
        """Compute cosine similarity with temperature scaling."""
        q = torch.nn.functional.normalize(queries, dim=-1)
        k = torch.nn.functional.normalize(keys, dim=-1)
        return q @ k.T / temp

    def get_teacher_sim(self, queries: torch.Tensor, keys: torch.Tensor):
        """Convinience function to compute similarity with teacher temperature."""
        return self.get_scaled_sim(queries, keys, self.temp_teacher)

    def get_student_sim(self, queries: torch.Tensor, keys: torch.Tensor):
        """Convinience function to compute similarity with student temperature."""
        return self.get_scaled_sim(queries, keys, self.temp_student)

class KDNaiveLoss(SimilarityLoss):
    """This loss use the same interface as KDLoss, but does not use the moving average mechanism."""
    def __init__(self,
                 data_size: int = None,
                 gamma1: float = 1.0,
                 gamma2: float = 1.0,
                 temp_student: float = 1.0,
                 temp_teacher: float = 1.0,
                 weights: torch.Tensor | None = None,
                 variant: Literal['kd', 'ce'] = 'kd',
                 ):
        """Knowledge distillation loss

        Args:
            data_size (int): size of the dataset, used to determine the length of u vector
            gamma1 (float): moving average coefficient for u update
            gamma2 (float): moving average coefficient for v update
            temp_student (float, optional): temperature for student. Defaults to 1.0.
            temp_teacher (float, optional): temperature for teacher. Defaults to 1.0.
            weights (torch.Tensor, optional): weights for each sample. Defaults to None.
        """
        super().__init__(temp_student=temp_student, temp_teacher=temp_teacher)

        logging.info("Initializing KDNaiveLoss with student temp %.2f and teacher temp %.2f", temp_student, temp_teacher)

        self.weights = weights
        self.variant = variant

        assert variant in ('kd', 'ce'), f"Unknown variant {variant}"
        

    def forward(self,
                queries_student: torch.Tensor,
                keys_student: torch.Tensor,
                queries_teacher: torch.Tensor,
                keys_teacher: torch.Tensor,
                index: torch.Tensor,
                ):
        sim_student = self.get_student_sim(queries_student, keys_student)
        sim_teacher = self.get_teacher_sim(queries_teacher, keys_teacher)
        
        if self.variant == 'kd':
            logprob_student = torch.log_softmax(sim_student, dim=-1)
            logprob_teacher = torch.log_softmax(sim_teacher, dim=-1)
            
            loss_batch = torch.nn.functional.kl_div(                
                input=logprob_student,
                target=logprob_teacher,
                reduction="batchmean",
                log_target=True,
            )
        else:
            loss_batch = torch.nn.functional.cross_entropy(
                input=sim_student,
                target=torch.softmax(sim_teacher, dim=-1),
            )
        return {"loss": loss_batch}
        
class KDLoss(SimilarityLoss):
    def __init__(self,
                 data_size: int,
                 gamma1: float = 1.0,
                 gamma2: float = 1.0,
                 temp_student: float = 1.0,
                 temp_teacher: float = 1.0,
                 weights: torch.Tensor | None = None,
                 ):
        """Knowledge distillation loss

        Args:
            data_size (int): size of the dataset, used to determine the length of u vector
            gamma1 (float): moving average coefficient for u update
            gamma2 (float): moving average coefficient for v update
            temp_student (float, optional): temperature for student. Defaults to 1.0.
            temp_teacher (float, optional): temperature for teacher. Defaults to 1.0.
            weights (torch.Tensor, optional): weights for each sample. Defaults to None.
        """
        super().__init__()
        self.data_size = data_size
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.temp_student = temp_student
        self.temp_teacher = temp_teacher
        self.weights = weights

        self.u = torch.zeros(data_size, device="cpu").reshape(-1, 1)
        self.v = torch.zeros(data_size, device="cpu").reshape(-1, 1)    

    def forward(self,
                queries_student: torch.Tensor,
                keys_student: torch.Tensor,
                queries_teacher: torch.Tensor,
                keys_teacher: torch.Tensor,
                index: torch.Tensor,
                ):
        # update u
        # logits are exp(cosine_similarity / temperature)
        sim_teacher = self.get_teacher_sim(queries_teacher, keys_teacher)
        logits_teacher = torch.exp(sim_teacher)                                        # shape: (B^x, B^k)
        with torch.no_grad():
            u = self.u[index].to(queries_teacher.device)                               # shape: (B^x, 1)
            if u.sum() == 0:
                gamma1 = 1.0
            else:
                gamma1 = self.gamma1
            u = (1 - gamma1) * u + gamma1 * torch.mean(logits_teacher, dim=-1, keepdim=True)
            self.u[index] = u.cpu()

        # update v
        sim_student = self.get_student_sim(queries_student, keys_student)
        logits_student = torch.exp(sim_student)                                        # shape: (B^x, B^k)
        with torch.no_grad():
            v = self.v[index].to(queries_student.device)                               # shape: (B^x, 1)
            if v.sum() == 0:
                gamma2 = 1.0
            else:
                gamma2 = self.gamma2
            v = (1 - gamma2) * v + gamma2 * torch.mean(logits_student, dim=-1, keepdim=True)
            self.v[index] = v.cpu()
        g_student_batch = torch.mean(logits_student, dim=-1, keepdim=True) / logits_student # shape: (B^x, B^k)

        # compute gradient estimator
        if self.weights is not None:
            weights = self.weights[index].to(queries_student.device)
        else:
            weights = torch.ones_like(index, dtype=torch.float32, device=queries_student.device)
    
        grad_estimator = torch.mean(logits_teacher.detach() / u * logits_student.detach() / v * g_student_batch * weights[:, None])
        # with torch.no_grad():
            # loss_per_element = torch.sum(torch.softmax(logits_teacher, dim=-1) * torch.log_softmax(logits_teacher, dim=-1) -\
            #                              torch.softmax(logits_teacher, dim=-1) * torch.log_softmax(logits_student, dim=-1), dim=-1)
            # loss_batch_mean = torch.mean(loss_per_element)
        loss_batch = torch.nn.functional.kl_div(
            input=torch.log_softmax(sim_student / self.temp_student, dim=-1),
            target=torch.softmax(sim_teacher / self.temp_teacher, dim=-1),
            reduction="batchmean",
        )
        return {"grad_estimator": loss_batch, "loss": loss_batch}       # TODO: check this
        # return {"grad_estimator": grad_estimator, "loss": loss_batch}

class CELoss(SimilarityLoss):
    """This is the 'hard' cross-entropy loss, where the targets are one-hot labels."""
    def __init__(self,
                 data_size: int = None,
                 gamma1: float = 1.0,
                 gamma2: float = 1.0,
                 temp_student: float = 1.0,
                 temp_teacher: float = 1.0,
                 weights: torch.Tensor | None = None,
                 ):
        super().__init__(temp_student=temp_student, temp_teacher=temp_teacher)
        self.weights = weights

    def forward(self,
                queries_student: torch.Tensor,
                keys_student: torch.Tensor,
                queries_teacher: torch.Tensor,
                keys_teacher: torch.Tensor,
                index: torch.Tensor,
                ):
        
        sim_student = self.get_student_sim(queries_student, keys_student)
        sim_teacher = self.get_teacher_sim(queries_teacher, keys_teacher)

        # prediction = argmax over the key dimension
        teacher_labels = torch.argmax(sim_teacher, dim=-1)

        loss_batch = torch.nn.functional.cross_entropy(
            input=sim_student,
            target=teacher_labels,
        )

        accuracy = torch.mean((sim_student.argmax(dim=-1) == teacher_labels).float())
        return {"loss": loss_batch, "accuracy": accuracy}

class CombinedLoss(torch.nn.Module):
    """Weighted sum of KL and CE losses"""
    def __init__(self, labda: float, **kwargs):
        super().__init__()
        self.labda = labda
        self.kl_loss = KDNaiveLoss(**kwargs)
        kwargs.pop('variant')
        self.ce_loss = CELoss(**kwargs)

    def forward(self, *args, **kwargs):
        kl_loss_dict = self.kl_loss(*args, **kwargs)
        ce_loss_dict = self.ce_loss(*args, **kwargs)
        kl_loss_val = kl_loss_dict.pop("loss")
        ce_loss_val = ce_loss_dict.pop("loss")
        return kl_loss_dict | ce_loss_dict | {
            "loss": self.labda * kl_loss_val + (1 - self.labda) * ce_loss_val,
            "kl_loss": kl_loss_val,
            "ce_loss": ce_loss_val,
        }

def init_loss(name: str, **kwargs):
    if name == "KDLoss":
        return KDLoss(**kwargs)
    elif name == "CELoss":
        return CELoss(**kwargs)
    elif name == "KDNaiveLoss":
        return KDNaiveLoss(**kwargs)
    elif name == 'KDNaiveLoss+CELoss':
        labda = kwargs.pop('labda')
        return CombinedLoss(labda, **kwargs)
    else:
        raise ValueError(f"Unknown loss name: {name}")