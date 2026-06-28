import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pulmonary_embolism_segmentation import (
    PulmonaryEmbolismSegmentationModel,
    SlidingWindowInferenceConfig,
    predict_volume,
    predict_volume_resampled,
)


def load_volume(path: Path):
    if path.is_dir():
        try:
            import SimpleITK as sitk
        except ImportError as exc:
            raise ImportError("Install SimpleITK to read DICOM directories: `pip install SimpleITK`.") from exc

        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(path))
        if not series_ids:
            raise ValueError(f"No DICOM series found in {path}")
        filenames = reader.GetGDCMSeriesFileNames(str(path), series_ids[0])
        reader.SetFileNames(filenames)
        image = reader.Execute()
        spacing_xyz = image.GetSpacing()
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        return sitk.GetArrayFromImage(image).astype(np.float32, copy=False), spacing_zyx

    suffixes = "".join(path.suffixes).lower()
    if path.suffix.lower() == ".npy":
        return np.load(path), None
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        if len(data.files) != 1:
            raise ValueError(f"NPZ input must contain exactly one array; found {data.files}")
        return data[data.files[0]], None
    if suffixes.endswith(".nii") or suffixes.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("Install nibabel to read NIfTI files: `pip install nibabel`.") from exc
        image = nib.load(str(path))
        spacing_xyz = tuple(float(v) for v in image.header.get_zooms()[:3])
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        data = np.asarray(image.get_fdata(dtype=np.float32))
        return np.moveaxis(data, (0, 1, 2), (2, 1, 0)), spacing_zyx

    try:
        import SimpleITK as sitk
    except ImportError as exc:
        raise ImportError(
            "Install SimpleITK for DICOM/MHA/NRRD input, or pass a .npy/.npz/.nii.gz file."
        ) from exc

    image = sitk.ReadImage(str(path))
    spacing_xyz = image.GetSpacing()
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    return sitk.GetArrayFromImage(image).astype(np.float32, copy=False), spacing_zyx


def save_segmentation(path: Path, segmentation: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npy":
        np.save(path, segmentation)
        return
    if path.suffix.lower() == ".npz":
        np.savez_compressed(path, segmentation=segmentation)
        return
    raise ValueError("Output must be .npy or .npz for now.")


def parse_tile_size(value: str):
    parts = tuple(int(part) for part in value.lower().replace("x", ",").split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Tile size must have three dimensions, e.g. 96,160,160")
    return parts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tile-size", default=(128, 256, 256), type=parse_tile_size)
    parser.add_argument("--min-tile-size", default=(64, 128, 128), type=parse_tile_size)
    parser.add_argument("--overlap", default=0.5, type=float)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--empty-cache-between-tiles", action="store_true")
    parser.add_argument("--no-resample", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"Using CUDA device: {props.name}, memory={props.total_memory / 1024**3:.1f} GiB")
    else:
        print(f"Using device: {device}")

    model = PulmonaryEmbolismSegmentationModel.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" and not args.no_amp else torch.float32,
    )
    volume, spacing_zyx = load_volume(args.input)
    inference_config = SlidingWindowInferenceConfig(
        tile_size=args.tile_size,
        min_tile_size=args.min_tile_size,
        overlap=args.overlap,
        use_amp=not args.no_amp,
        device=device,
        empty_cache_between_tiles=args.empty_cache_between_tiles,
    )
    if spacing_zyx is not None and not args.no_resample:
        print(f"Input shape={volume.shape}, spacing zyx={spacing_zyx}; resampling to plan spacing={model.config.spacing}")
        segmentation = predict_volume_resampled(model, volume, spacing_zyx, inference_config)
    else:
        print(f"Input shape={volume.shape}; running without spacing resampling")
        segmentation = predict_volume(model, volume, inference_config)
    save_segmentation(args.output, segmentation)
    print(f"Saved segmentation to {args.output}")


if __name__ == "__main__":
    main()
