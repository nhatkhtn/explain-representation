from dotenv import dotenv_values
import wandb

from exrep.registry import save_data, load_data

local_config = dotenv_values(".env")

run = wandb.init(
    project=local_config["WANDB_PROJECT"],
    config={
        "dataset": "imagenet-1k",
        "subset": "first-20-take-2000",
        "job_type": "data_preprocessing",
    },
    save_code=True,
)

def main():
    ds_name = f"{run.config.dataset}-{run.config.subset}"
    # full_dataset = datasets.load_dataset("imagenet-1k")["train"]
    full_dataset = load_data(base_name="imagenet-1k-first-20", phase="images", alias="latest", wandb_run=run, load_local=True)

    if run.config.subset == "first-20":
        subset_dataset = full_dataset.filter(lambda x: x['label'] < 20, keep_in_memory=True, num_proc=4)
    elif run.config.subset == "first-20-take-2000":
        subset_dataset = full_dataset.filter(lambda x: x['label'] < 20, keep_in_memory=True, num_proc=4).take(2000)
    else:
        raise ValueError("Invalid subset")

    save_data(
        dataset=subset_dataset,
        base_name=ds_name,
        phase="images",
        wandb_run=run,
    )

main()