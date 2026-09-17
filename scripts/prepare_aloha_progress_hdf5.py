"""Build a compact timestamp-aligned HDF5 index for ALOHA progress training."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import cv2


DEFAULT_ROOT = Path("/home/weiran/projects_wr/datasets/aloha_unbox")
DEFAULT_OUTPUT = Path("runs/aloha_progress/data/aloha_unbox_50hz_84.hdf5")
CAMERA = "CAM_TOP_orig"
FREQ = 50.0


def list_frames(episode: Path) -> tuple[list[Path], np.ndarray]:
    paths = sorted((episode / "images" / CAMERA).glob("*.jpg"))
    if not paths:
        raise ValueError(f"No {CAMERA} JPEGs in {episode}")
    times = np.asarray([float(path.stem) for path in paths], dtype=np.float64)
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"Non-increasing camera timestamps in {episode}")
    return paths, times


def grid_source_indices(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Include the initial time and every 20 ms target not later than episode end.
    count = int(np.floor((times[-1] - times[0]) * FREQ + 1e-8)) + 1
    grid_times = times[0] + np.arange(count, dtype=np.float64) / FREQ
    # Use the latest image already available at each target time, avoiding a
    # future-frame peek. The first target is exactly the first image time.
    indices = np.searchsorted(times, grid_times, side="right") - 1
    indices = np.clip(indices, 0, len(times) - 1).astype(np.uint32)
    return grid_times, indices


def add_group(h5: h5py.File, group_name: str, episodes: list[Path], demo_names: list[str]) -> None:
    demos = h5.require_group("data")
    for episode, demo_name in zip(episodes, demo_names, strict=True):
        image_paths, image_times = list_frames(episode)
        grid_times, source_indices = grid_source_indices(image_times)
        demo = demos.create_group(demo_name)
        demo.attrs["num_samples"] = len(grid_times)
        demo.attrs["source_group"] = group_name
        demo.attrs["source_episode"] = episode.name
        demo.attrs["freq"] = FREQ
        demo.attrs["original_freq_median_hz"] = 1.0 / np.median(np.diff(image_times))
        demo.create_dataset("actions", data=np.zeros((len(grid_times), 1), dtype=np.float32))
        obs = demo.create_group("obs")
        source_images = []
        for i, path in enumerate(image_paths):
            encoded = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError(f"Could not decode top-camera JPEG {path}")
            rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            source_images.append(cv2.resize(rgb, (84, 84), interpolation=cv2.INTER_AREA))
        # Materialize the actual 50 Hz training sequence in HDF5. This avoids
        # filesystem JPEG access and source-frame lookup during every batch.
        grid_images = np.stack(source_images, axis=0)[source_indices]
        obs.create_dataset(
            "agentview_image", data=grid_images, dtype=np.uint8,
            chunks=(16, 84, 84, 3), compression="lzf",
        )
        obs.create_dataset("timestamps", data=grid_times, compression="gzip")
        obs.create_dataset("source_image_timestamps", data=image_times, compression="gzip")
        obs.create_dataset("source_image_index", data=source_indices, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    close_eps = sorted(path for path in (args.data_root / "close").iterdir() if path.is_dir())
    fail_eps = sorted(path for path in (args.data_root / "fail").iterdir() if path.is_dir())
    if len(close_eps) != 120 or len(fail_eps) != 14:
        raise ValueError(f"Expected 120 close and 14 fail episodes, got {len(close_eps)} and {len(fail_eps)}")
    for episode in close_eps + fail_eps:
        list_frames(episode)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    close_names = [f"demo_{i:03d}" for i in range(len(close_eps))]
    fail_names = [f"demo_{i + len(close_eps):03d}" for i in range(len(fail_eps))]
    all_names = close_names + fail_names
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(args.output, "w") as h5:
        h5.attrs["source_root"] = str(args.data_root)
        h5.attrs["camera"] = CAMERA
        h5.attrs["sample_frequency_hz"] = FREQ
        add_group(h5, "close", close_eps, close_names)
        add_group(h5, "fail", fail_eps, fail_names)
        mask = h5.create_group("mask")
        mask.create_dataset("close", data=np.asarray(close_names, dtype=string_dtype))
        mask.create_dataset("close_fail", data=np.asarray(all_names, dtype=string_dtype))
    print(f"Wrote {args.output} with {len(close_eps)} close and {len(fail_eps)} recovered-failure episodes")


if __name__ == "__main__":
    main()
