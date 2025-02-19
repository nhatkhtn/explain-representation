import torch

class KDLoss(torch.nn.Module):
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
        # normalize the queries and keys
        queries_student = torch.nn.functional.normalize(queries_student, dim=-1)
        keys_student = torch.nn.functional.normalize(keys_student, dim=-1)

        # update u
        # logits are exp(cosine_similarity / temperature)
        sim_teacher = queries_teacher @ keys_teacher.t()
        logits_teacher = torch.exp(sim_teacher / self.temp_teacher)                      # shape: (B^x, B^k)
        with torch.no_grad():
            u = self.u[index].to(queries_teacher.device)                               # shape: (B^x, 1)
            if u.sum() == 0:
                gamma1 = 1.0
            else:
                gamma1 = self.gamma1
            u = (1 - gamma1) * u + gamma1 * torch.mean(logits_teacher, dim=-1, keepdim=True)
            self.u[index] = u.cpu()

        # update v
        sim_student = queries_student @ keys_student.t()
        logits_student = torch.exp(sim_student / self.temp_student)                      # shape: (B^x, B^k)
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

class CELoss(torch.nn.Module):
    """Optimizing this loss with a fixed teacher is equivalent to optimizing the KL divergence loss."""
    def __init__(self,
                 data_size: int,
                 gamma1: float = 1.0,
                 gamma2: float = 1.0,
                 temp_student: float = 1.0,
                 temp_teacher: float = 1.0,
                 weights: torch.Tensor | None = None,
                 ):
        """Has the same signature as KDLoss for easy switching, but ignores the gamma parameters."""
        super().__init__()
        self.temp_student = temp_student
        self.temp_teacher = temp_teacher
        self.weights = weights

    def forward(self,
                queries_student: torch.Tensor,
                keys_student: torch.Tensor,
                queries_teacher: torch.Tensor,
                keys_teacher: torch.Tensor,
                index: torch.Tensor,
                ):
        # normalize the queries and keys
        queries_student = torch.nn.functional.normalize(queries_student, dim=-1)
        keys_student = torch.nn.functional.normalize(keys_student, dim=-1)

        # logits are exp(cosine_similarity / temperature)
        sim_teacher = queries_teacher @ keys_teacher.t() / self.temp_teacher
        sim_student = queries_student @ keys_student.t() / self.temp_student

        loss_batch = torch.nn.functional.cross_entropy(
            input=sim_student,
            target=torch.softmax(sim_teacher, dim=-1),
        )
        return {"grad_estimator": loss_batch, "loss": loss_batch}
        
def init_loss(name: str, **kwargs):
    if name == "KDLoss":
        return KDLoss(**kwargs)
    elif name == "CELoss":
        return CELoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss name: {name}")