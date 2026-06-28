import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pulmonary_embolism_segmentation import PulmonaryEmbolismSegmentationModel


DEFAULT_TILE_SIZES = (
    (64, 128, 128),
    (96, 128, 128),
    (96, 192, 192),
    (128, 192, 192),
    (128, 256, 256),
    (160, 256, 256),
    (224, 320, 320),
)


def parse_tile_size(value: str):
    parts = tuple(int(part) for part in value.lower().replace("x", ",").split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Tile size must have three dimensions, e.g. 96,160,160")
    return parts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--tile-size", action="append", type=parse_tile_size)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"CUDA device: {props.name}, memory={props.total_memory / 1024**3:.1f} GiB")

    model = PulmonaryEmbolismSegmentationModel.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16 if not args.no_amp else torch.float32,
    ).to(device)
    model.eval()

    tile_sizes = args.tile_size or DEFAULT_TILE_SIZES
    for tile_size in tile_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            x = torch.zeros((1, 1, *tile_size), device=device)
            with torch.inference_mode(), torch.autocast("cuda", enabled=not args.no_amp):
                y = model(x).logits
            torch.cuda.synchronize()
            peak_gib = torch.cuda.max_memory_allocated() / 1024**3
            print(f"OK  tile={tile_size} output={tuple(y.shape)} peak={peak_gib:.2f} GiB")
            del x, y
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"OOM tile={tile_size}")


if __name__ == "__main__":
    main()
