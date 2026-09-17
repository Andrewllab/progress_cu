"""ALOHA progress predictor with the released temporal bins at 50 Hz."""
from curation.suboptimal_classifier.config.robomimic import get_config as base_config


def get_config():
    config = base_config()
    config.hdf5_dataset_kwargs.freq = 50
    config.hdf5_dataset_kwargs.batch_size = 32
    config.hdf5_dataset_kwargs.num_workers = 0
    config.hdf5_dataset_kwargs.time_bins = [
        [0.0, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 5.0], [5.0, 10000.0]
    ]
    config.discriminator.rank_thres = config.hdf5_dataset_kwargs.time_bins
    config.discriminator.num_blocks = 4
    config.num_steps = 10000
    config.save_interval = 2000
    config.log_interval = 100
    config.save_dir = "runs/progress/aloha_unbox"
    return config
