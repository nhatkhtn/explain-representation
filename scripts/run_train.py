import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    import os
    
    from pathlib import Path

    from dotenv import dotenv_values
    import wandb
    import torch
    from sklearn.model_selection import train_test_split
    from matplotlib import pyplot as plt

    from scripts.train_surrogate import load_data_from_runs, train_surrogate_experiment

    torch._logging.set_logs(dynamo=logging.DEBUG)
    
    random_state = 42

    device = "cuda:4"

    sweep_project_name = "sweep-exrep-downstream1"

    run = wandb.init(
        project=sweep_project_name,
        config={
            "job_type": "train_representation",
            "num_clusters": 40,
        },
        # reinit=True,
        save_code=True,
    )

    device = "cuda:7"

    model, logs = train_surrogate_experiment(run, device=device, save=False)

if __name__ == "__main__":
    import signal
    import wandb

    # def handler(signum, frame):
    #     print("Exception handler called!")
    #     wandb.finish(exit_code=1)
    #     raise RuntimeError("Run timeout")
        
    # signal.signal(signal.SIGALRM, handler)

    # signal.alarm(90)

    # try:
    #     main()
    # except RuntimeError as e:
    #     print(e)

    main()