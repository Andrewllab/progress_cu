"""ALOHA progress predictor with widened temporal bins at 50 Hz."""
from curation.suboptimal_classifier.config.aloha_50hz import get_config as base_config


def get_config():
    config = base_config()
    config.hdf5_dataset_kwargs.time_bins = [
        [0.0, 1.0], [1.0, 2.0], [2.0, 4.0], [4.0, 8.0], [8.0, 10000.0]
    ]
    config.discriminator.rank_thres = config.hdf5_dataset_kwargs.time_bins
    return config
