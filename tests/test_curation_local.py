"""Offline checks for the local HDF5 path; no pretrained downloads needed."""
import pickle

import h5py
import numpy as np
import torch

from curation.suboptimal_classifier.config.robomimic import get_config
from curation.suboptimal_classifier.dataset.hdf5_dataset import HDF5Dataset
from curation.video_encoding.suboptimal_hdf5 import Evaluator
from curation.semdedup.local import main as deduplicate
from curation.video_encoding.write_semdup_score import write_max_sim


def test_hdf5_split_noncontiguous_names_and_zero_workers(tmp_path):
    with h5py.File(tmp_path / "test.hdf5", "w") as file:
        for name in ["demo_2", "demo_9"]:
            demo = file.create_group(f"data/{name}")
            demo.attrs['num_samples'] = 120
            demo.create_dataset('actions', data=np.zeros((120, 7)))
            demo.create_dataset('obs/agentview_image', data=np.zeros((120, 84, 84, 3), dtype=np.uint8))
        file.create_dataset('mask/train', data=np.array([b'demo_9']))
    config = get_config()
    config.hdf5_dataset_kwargs.data_dir = str(tmp_path)
    config.hdf5_dataset_kwargs.filter_key = 'train'
    config.hdf5_dataset_kwargs.batch_size = 2
    loader = HDF5Dataset(config)
    assert loader.dataset.datasets[0].demo_keys == ['demo_9']
    image, _, elapsed, *_ = next(loader)
    assert image.shape == (2, 2, 3, 84, 84)
    assert torch.isfinite(image).all()
    assert torch.all((elapsed >= 0) & (elapsed <= 119 / 20))


def test_score_short_trajectories_and_one_frame_windows():
    evaluator = Evaluator.__new__(Evaluator)
    evaluator.config = get_config().to_dict()
    for length in (1, 3, 50):
        for window in (1, 40):
            evaluator.goal_time = [window / 20]
            distances = np.minimum(window, length - 1 - np.arange(length))[:, None]
            probabilities = np.full((length, 1, 5), 0.2)
            score = evaluator.rank_prob_to_score(probabilities, distances, 20)
            assert score.shape == (length,)
            assert np.isfinite(score).all()


def test_duplicate_scoring_and_hdf5_write(tmp_path):
    folder = tmp_path / 'embeddings' / 'test'
    folder.mkdir(parents=True)
    ids = np.array(['test-demo_0-0-40', 'test-demo_0-40-80', 'test-demo_0-80-120'])
    with (folder / 'features.pkl').open('wb') as handle:
        pickle.dump(dict(id=ids, image_embeds=np.ones((3, 8)), action=np.ones((3, 8, 7))), handle)
    out = tmp_path / 'dedup'
    deduplicate(str(folder.parent), str(out), ncentroids=1, niter=2)
    with (out / 'dedup.pkl').open('rb') as handle:
        scores = pickle.load(handle)
    assert len(scores) == 3
    assert np.count_nonzero(scores.max_sim.to_numpy() > 0.99) == 2
    with h5py.File(tmp_path / 'test.hdf5', 'w') as file:
        file.create_dataset('data/demo_0/actions', data=np.zeros((120, 7)))
    write_max_sim(str(tmp_path), str(out / 'dedup.pkl'))
    with h5py.File(tmp_path / 'test.hdf5') as file:
        assert file['data/demo_0/max_sim_idx'].shape == (3, 2)
        assert np.count_nonzero(file['data/demo_0/max_sim'][:] > 0.99) == 2
