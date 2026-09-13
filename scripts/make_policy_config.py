"""Prepare a small-machine policy config from the released Can/Square template."""
import json
from pathlib import Path
import tyro


def main(dataset: str, output: str, task: str = "can", mode: str = "subop",
         subop_keep: float = 0.707, dedup_keep: float = 0.997,
         mix_level: float = 0.5, seed: int = 1, rollouts: bool = True):
    if task not in ('can', 'square') or mode not in ('none', 'subop', 'dedup', 'both'):
        raise ValueError('task: can/square; mode: none/subop/dedup/both')
    if not (0 < subop_keep <= 1 and 0 < dedup_keep < 1 and 0 <= mix_level <= 1):
        raise ValueError('Invalid retention ratio or mixture weight')
    root = Path(__file__).resolve().parent.parent
    template = root / f'robomimic/robomimic/exps/curation_exps/robomimic/{task}/bc_curation.json'
    config = json.loads(template.read_text())
    config['train']['data'] = str(Path(dataset).resolve())
    config['train']['output_dir'] = str(root / 'runs/policy')
    config['train']['seed'] = seed
    config['train']['num_data_workers'] = 0
    config['experiment']['name'] = f'{task}_{mode}_seed{seed}'
    config['experiment']['rollout'].update(enabled=rollouts, batched=False, n=20)
    # The released loader applies curation to validation too. Use only mask/train
    # here and evaluate via rollouts; do not present curated validation loss as
    # an independent measure of curation quality.
    config['experiment']['validate'] = False
    config['train']['hdf5_filter_key'] = 'train'
    subop = config['train']['curation']['subop_curate']
    subop.update(enabled=mode in ('subop', 'both'), subop_percentile=subop_keep,
                 mix_level=mix_level)
    dedup = config['train']['curation']['dedup_curate']
    dedup.update(enabled=mode in ('dedup', 'both'), dedup_type='semantic', keep_ratio=dedup_keep)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as handle:
        json.dump(config, handle, indent=2)
    print(destination)


if __name__ == '__main__':
    tyro.cli(main)
