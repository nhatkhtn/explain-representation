import argparse
import logging
import sys
from operator import itemgetter

import torch
import torch.utils.data

logging.basicConfig(
    level=logging.INFO,
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class KDLossNaive(torch.nn.Module):
    def __init__(self,
                 data_size: int,
                 gamma1: float,
                 gamma2: float,
                 temp_student: float = 1.0,
                 temp_teacher: float = 1.0,
                 weights: torch.Tensor | None = None,
                 ):
        """Knowledge distillation loss

        Args:
            data_size (int): size of the dataset, used to determine the length of u vector
            gamma1 (float): moving average coefficient for u update
            gamma2 (float): moving average coefficient for v update
            temp_s (float, optional): temperature for student. Defaults to 1.0.
            temp_t (float, optional): temperature for teacher. Defaults to 1.0.
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
        index
    ):
        # normalize the student's queries and keys
        queries_student = torch.nn.functional.normalize(queries_student, p=2, dim=-1)
        keys_student = torch.nn.functional.normalize(keys_student, p=2, dim=-1)
        
        # logits are exp(cosine_similarity / temperature)
        sim_teacher = queries_teacher @ keys_teacher.t()                                # shape: (Q, K)
        log_probs_teacher = torch.nn.functional.log_softmax(sim_teacher / self.temp_teacher, dim=-1)            
      
        sim_student = queries_student @ keys_student.t()                                # shape: (Q, K)
        log_probs_student = torch.nn.functional.log_softmax(sim_student / self.temp_student, dim=-1)

        # the teacher is fixed meaning the entropy H(teacher) is fixed, so we can miminize the KL divergence
        kl_loss = torch.nn.functional.kl_div(log_probs_student, log_probs_teacher, reduction="batchmean", log_target=True)
        return kl_loss


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


def distill_one_epoch(
    query_encoder: torch.nn.Module,
    key_encoder: torch.nn.Module,
    query_embeddings: torch.Tensor,
    keys_embeddings: torch.Tensor,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: KDLossNaive,
    regularizer_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: str,
    log_every_n_steps: int,
) -> list[dict[str, int | float]]:
    """Distill one epoch

    Args:
        query_encoder (torch.nn.Module): student model
        key_encoder (torch.nn.Module): trainable projector from teacher space to student space
        query_embeddings (torch.Tensor): teacher-embedded queries
        keys_embeddings (torch.Tensor): teacher-embedded keys
        dataloader (torch.utils.data.DataLoader): dataloader for queries (images in local features form)
        loss (torch.nn.Module): loss function
        optimizer (torch.optim.Optimizer): optimizer
        epoch (int): epoch
        device (str): device
        log_every_n_steps (int): log every n steps
    """
    query_encoder.train()
    key_encoder.train()

    device = torch.device(device)

    if not isinstance(query_embeddings, torch.Tensor):
        query_embeddings = torch.tensor(query_embeddings, device=device)
    if not isinstance(keys_embeddings, torch.Tensor):
        keys_embeddings = torch.tensor(keys_embeddings, device=device)

    logs = []

    for i, batch in enumerate(dataloader):
        images, indices = itemgetter("images", "indices")(batch)
        images = images.to(device)

        queries_student = query_encoder(images)
        keys_student = key_encoder(keys_embeddings)

        losses = loss_fn(queries_student, keys_student, query_embeddings[indices], keys_embeddings, indices)
        # only the query encoder is regularized
        loss = losses["loss"] + regularizer_fn(query_encoder)

        optimizer.zero_grad()
        losses["grad_estimator"].backward()
        optimizer.step()

        if i % log_every_n_steps == 0:
            logging.info(f"Epoch {epoch:>2d}, Step {i:>4d}, Loss: {loss.item():.5f}")
        logs.append({"epoch": epoch, "step": i, "loss": loss.item()})

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
    # weights = torch.randn(args.data_size)
    weights = None                              # default, equal weights
    loss = KDLoss(data_size=args.data_size, gamma1=args.gamma, gamma2=args.gamma, weights=weights).to(args.device)

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
