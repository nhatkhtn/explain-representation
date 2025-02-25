import datasets
from dotenv import dotenv_values
import wandb

from exrep.registry import save_hf_dataset

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
    full_dataset = datasets.load_dataset("imagenet-1k")
    # full_dataset = load_data(base_name="imagenet-1k-first-20", phase="images", alias="latest", wandb_run=run, load_local=True)

    num_proc = 8

    if run.config.subset.startswith("first-20"):
        def filter_fn(x): return x['label'] < 20
        
    else:
        raise ValueError("Invalid subset")

    train_subset = full_dataset["train"].filter(filter_fn, num_proc=num_proc)
    val_subset = full_dataset["validation"].filter(filter_fn, num_proc=num_proc)
    
    if "take-2000" in run.config.subset:
        train_subset = train_subset.take(2000)
        
    subset_dataset = datasets.DatasetDict({
        "train": train_subset,
        "validation": val_subset,
    })

    print(subset_dataset)

    save_hf_dataset(
        dataset=subset_dataset,
        base_name=ds_name,
        phase="images",
        mode='write-new',
        wandb_run=run,
    )

if __name__ == "__main__":
    main()