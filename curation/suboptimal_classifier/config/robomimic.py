"""Direct HDF5 setup: same model settings as the released robomimic launcher."""
from curation.suboptimal_classifier.config.config import get_config as base_config
from ml_collections.config_dict import placeholder


def get_config():
    config = base_config()
    config.future_image = True
    config.num_steps = 10000
    config.optimizer.lr = 1e-4
    config.save_interval = 2000
    config.save_dir = "runs/progress"
    config.hdf5_dataset_kwargs.batch_size = 32
    # Avoid inheriting open HDF5 handles across worker processes by default.
    config.hdf5_dataset_kwargs.num_workers = 0
    config.hdf5_dataset_kwargs.filter_key = placeholder(str)
    config.hdf5_dataset_kwargs.obs_keys = {"agentview_image": 84}
    config.discriminator_dataset_kwargs.image_key = "agentview_image"
    config.discriminator.update(dict(
        action_query_length=1, num_blocks=6, head_token="cls",
        loss_fn_type="cross_entropy", no_action_input=True,
        frozen_encoder=True, encoder_type="dinov2", d_model=768,
        fusion_blocks_type="self-attn", head_type="rank", no_text_input=True,
    ))
    return config
