import argparse
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pulmonary_embolism_segmentation import (
    PulmonaryEmbolismSegmentationConfig,
    PulmonaryEmbolismSegmentationModel,
)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("network_weights", "state_dict", "model_state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def normalize_key(key):
    for prefix in ("module.", "_orig_mod."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def normalize_state_dict(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        key = normalize_key(key)
        if not key.startswith("segmentation_model."):
            key = f"segmentation_model.{key}"
        normalized[key] = value
    return normalized


def save_weights(model, output_dir):
    for stale_name in ("model.safetensors", "pytorch_model.bin"):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    torch.save(model.state_dict(), output_dir / "pytorch_model.bin")
    return "pytorch_model.bin"


def copy_model_code(output_dir):
    package_src = Path(__file__).resolve().parents[1] / "pulmonary_embolism_segmentation"
    package_dst = output_dir / "pulmonary_embolism_segmentation"
    if package_dst.exists():
        shutil.rmtree(package_dst)
    shutil.copytree(package_src, package_dst)

    repo_root = Path(__file__).resolve().parents[1]
    shutil.copy2(repo_root / "configuration_pe_segmentation.py", output_dir / "configuration_pe_segmentation.py")
    shutil.copy2(repo_root / "modeling_pe_segmentation.py", output_dir / "modeling_pe_segmentation.py")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--skip-code-copy", action="store_true")
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help="Allow loading older trusted PyTorch checkpoints that require pickle.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = PulmonaryEmbolismSegmentationConfig()
    model = PulmonaryEmbolismSegmentationModel(config)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=not args.allow_unsafe_pickle,
    )
    state_dict = normalize_state_dict(extract_state_dict(checkpoint))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print("Checkpoint did not match the portable architecture exactly.")
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")
        if missing:
            print("First missing keys:", missing[:10])
        if unexpected:
            print("First unexpected keys:", unexpected[:10])
        raise SystemExit(1)

    config.save_pretrained(args.output_dir)
    weight_file = save_weights(model, args.output_dir)
    if not args.skip_code_copy:
        copy_model_code(args.output_dir)

    print(f"Saved Hugging Face model to {args.output_dir}")
    print(f"Weights: {weight_file}")


if __name__ == "__main__":
    main()
