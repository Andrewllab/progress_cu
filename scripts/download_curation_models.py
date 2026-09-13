"""Download only the DINOv2 backbone and the Cosmos encoder used by SCIZOR."""
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


if __name__ == "__main__":
    snapshot_download("facebook/dinov2-base", allow_patterns=[
        "config.json", "model.safetensors", "preprocessor_config.json",
    ])
    destination = Path(".cache/cosmos/checkpoints/Cosmos-0.1-Tokenizer-CV8x16x16")
    hf_hub_download("nvidia/Cosmos-0.1-Tokenizer-CV8x16x16", "encoder.jit",
                    local_dir=destination)
    print(f"Cosmos encoder: {destination / 'encoder.jit'}")
