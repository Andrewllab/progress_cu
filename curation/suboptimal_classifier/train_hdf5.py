"""Single-device HDF5 trainer without Octo, TensorFlow, DALI or ZMQ.

Keeps the released model/loss and temporal sampler for reproducibility.
Writes config.json and model_<step>.pth compatible with suboptimal_hdf5.py.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from absl import app, flags
from ml_collections import config_flags
from tqdm import trange

from curation.suboptimal_classifier.dataset.hdf5_dataset import HDF5Dataset
from curation.suboptimal_classifier.discriminator.discriminator import Discriminator

FLAGS = flags.FLAGS
flags.DEFINE_string("name", "robomimic", "Run subdirectory (must not already exist).")
config_flags.DEFINE_config_file(
    "config", str(Path(__file__).parent / "config/robomimic.py"), lock_config=False,
)


def main(_):
    config = FLAGS.config
    if not config.hdf5_dataset_kwargs.data_dir:
        raise ValueError("Set --config.hdf5_dataset_kwargs.data_dir to your HDF5 directory")
    if config.mixed_precision or config.grad_accum_steps != 1:
        raise ValueError("This direct trainer supports float32 and grad_accum_steps=1")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HDF5Dataset(config)
    model = Discriminator(**config.discriminator).to(device)
    if config.load_path:
        model.load_state_dict(torch.load(config.load_path, map_location=device, weights_only=True))
    run_dir = Path(config.save_dir) / FLAGS.name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(config.to_json(indent=2))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], **config.optimizer,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.optimizer.lr, total_steps=config.num_steps,
        pct_start=config.scheduler.pct_start,
    )
    model.train()
    if config.discriminator.frozen_encoder:
        model.encoder.eval()
    print(f"Training on {device}; checkpoints: {run_dir}")
    with (run_dir / "metrics.jsonl").open("w") as log:
        for step in trange(1, config.num_steps + 1):
            images, _, elapsed, *_ = next(dataset)
            result = model(images.to(device), None, None, score=elapsed.to(device))
            optimizer.zero_grad(set_to_none=True)
            result["loss"].backward()
            optimizer.step()
            scheduler.step()
            if step == 1 or step % config.log_interval == 0:
                log.write(json.dumps(dict(step=step, loss=result["loss"].item(),
                                          metrics=result["metrics"])) + "\n")
                log.flush()
            if step % config.save_interval == 0 or step == config.num_steps:
                torch.save(model.state_dict(), run_dir / f"model_{step}.pth")


if __name__ == "__main__":
    app.run(main)
