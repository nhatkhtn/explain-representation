import argparse
import logging
import sys
from operator import itemgetter
from typing import Callable

import torch
import torch.utils.data

logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def euclidean_to_cosine_similarity_kernel(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Kernel recovering the cosine similarity from the euclidean distance

    Args:
        X (torch.Tensor): tensor of shape (N, D)
        Y (torch.Tensor): tensor of shape (M, D)

    Returns:
        torch.Tensor: tensor of shape (N, M)
    """
    return 1 - torch.cdist(X, Y, p=2) ** 2 / 2

class KDLoss(torch.nn.Module):
    def __init__(self,
                 data_size: int,
                 gamma1: float,
                 gamma2: float,
                 temperature: float = 1.0,
                 kernel: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = euclidean_to_cosine_similarity_kernel
                 ):
        """Knowledge distillation loss

        Args:
            data_size (int): size of the dataset, used to determine the length of u vector
            gamma1 (float): moving average coefficient for u update
            gamma2 (float): moving average coefficient for v update
            temperature (float): temperature for softmax
        """
        super().__init__()
        self.data_size = data_size
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.temperature = temperature

        self.u = torch.zeros(data_size, device="cpu").reshape(-1, 1)
        self.v = torch.zeros(data_size, device="cpu").reshape(-1, 1)
        self.kernel = kernel

    def forward(self, features_student, features_teacher, index):
        # normalize the features
        # features_student = torch.nn.functional.normalize(features_student, dim=-1)
        # features_teacher = torch.nn.functional.normalize(features_teacher, dim=-1)

        # update u
        # logits are exp(cosine_similarity / temperature)
        # sim_teacher = features_teacher @ features_teacher.t()
        sim_teacher = self.kernel(features_teacher, features_teacher)
        logits_teacher = torch.exp(sim_teacher / self.temperature)                      # shape: (B, B)
        with torch.no_grad():
            u = self.u[index].to(features_teacher.device)                               # shape: (B, 1)
            if u.sum() == 0:
                gamma1 = 1.0
            else:
                gamma1 = self.gamma1
            u = (1 - gamma1) * u + gamma1 * torch.mean(logits_teacher, dim=-1, keepdim=True)
            self.u[index] = u.cpu()

        # update v
        # sim_student = features_student @ features_student.t()
        sim_student = self.kernel(features_student, features_student)
        logits_student = torch.exp(sim_student / self.temperature)                      # shape: (B, B)
        with torch.no_grad():
            v = self.v[index].to(features_student.device)                               # shape: (B, 1)
            if v.sum() == 0:
                gamma2 = 1.0
            else:
                gamma2 = self.gamma2
            v = (1 - gamma2) * v + gamma2 * torch.mean(logits_student, dim=-1, keepdim=True)
            self.v[index] = v.cpu()
        g_student_batch = torch.mean(logits_student, dim=-1, keepdim=True) / logits_student # shape: (B, B)

        # compute gradient estimator
        grad_estimator = torch.mean(logits_teacher.detach() / u * logits_student.detach() / v * g_student_batch)
        with torch.no_grad():
            loss_batch = torch.mean(logits_teacher / torch.mean(logits_teacher, dim=-1, keepdim=True) * torch.log(torch.mean(logits_student, dim=-1, keepdim=True) / logits_student))
        return {"grad_estimator": grad_estimator, "loss": loss_batch}


def ditill_one_epoch(model_student, model_teacher, dataloader, loss, optimizer, epoch, args):
    """Distill one epoch

    Args:
        model_student (torch.nn.Module): student model
        model_teacher (torch.nn.Module): teacher model
        dataloader (torch.utils.data.DataLoader): data loader
        loss (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
        args (argparse.Namespace): arguments
    """
    model_student.train()
    model_teacher.eval()

    device = torch.device(args.device)

    for i, batch in enumerate(dataloader):
        images, index = batch
        images = images.to(device)

        features_student = model_student(images)
        with torch.no_grad():
            features_teacher = model_teacher(images)

        losses = loss(features_student, features_teacher, index)
        optimizer.zero_grad()
        losses["grad_estimator"].backward()
        optimizer.step()

        if i % args.log_every_n_steps == 0:
            logging.info((f"Epoch {epoch:>2d}, Step {i:>4d}, Loss: {losses['loss'].item():.5f}"))


def distill_one_epoch(model_student, teacher_embeddings, dataloader, loss, optimizer, epoch, device, log_every_n_steps):
    """Distill one epoch

    Args:
        model_student (torch.nn.Module): student model
        teacher_embeddings (torch.nn.Tensor): teacher embeddings
        dataloader (torch.utils.data.DataLoader): data loader
        loss (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
        epoch (int): epoch
        device (str): device
        log_every_n_steps (int): log every n steps
    """
    model_student.train()

    device = torch.device(device)

    if not isinstance(teacher_embeddings, torch.Tensor):
        teacher_embeddings = torch.tensor(teacher_embeddings, device=device)

    logs = []

    for i, batch in enumerate(dataloader):
        images, indices = itemgetter("images", "indices")(batch)
        images = images.to(device)

        features_student = model_student(images)
        features_teacher = teacher_embeddings[indices]

        losses = loss(features_student, features_teacher, indices)
        
        optimizer.zero_grad()
        losses["grad_estimator"].backward()
        optimizer.step()

        if i % log_every_n_steps == 0:
            logging.info((f"Epoch {epoch:>2d}, Step {i:>4d}, Loss: {losses['loss'].item():.5f}"))
            logs.append({"epoch": epoch, "step": i, "loss": losses["loss"].item()})

    return logs

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--data_size", type=int, default=1000000, help="Size of the dataset, used to determine the length of u vector.")
    parser.add_argument("--gamma", type=float, default=0.9, help="Moving average coefficient for u and v update.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--log_every_n_steps", type=int, default=50, help="Log every n steps")
    args = parser.parse_args(args)

    logging.info("Params:")
    for name in sorted(vars(args)):
        val = getattr(args, name)
        logging.info(f"  {name}: {val}")

    # create models
    model_student = torch.nn.Linear(10, 10).to(args.device)
    model_teacher = torch.nn.Linear(10, 10).to(args.device)

    # create loss
    loss = KDLoss(data_size=args.data_size, gamma1=args.gamma, gamma2=args.gamma).to(args.device)

    # create optimizer
    optimizer = torch.optim.AdamW(model_student.parameters(), lr=args.lr, weight_decay=args.wd)

    # load data
    data = torch.randn(args.data_size, 10)
    index = torch.arange(data.shape[0])
    dataset = torch.utils.data.TensorDataset(data, index)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # distill
    for epoch in range(args.epochs):
        ditill_one_epoch(model_student, model_teacher, dataloader, loss, optimizer, epoch, args)


if __name__ == "__main__":
    main(sys.argv[1:])
