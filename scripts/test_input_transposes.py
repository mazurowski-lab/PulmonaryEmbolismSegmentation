import argparse
import itertools
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pulmonary_embolism_segmentation import (  # noqa: E402
    PulmonaryEmbolismSegmentationModel,
    SlidingWindowInferenceConfig,
    predict_volume_resampled,
)
from scripts.run_inference import load_volume, save_segmentation  # noqa: E402


def load_ground_truth_zyx(case_root: Path, case: str) -> np.ndarray:
    nii = nib.load(str(case_root / "ground_truth" / f"{case}.nii.gz"))
    gt_xyz = np.asanyarray(nii.dataobj)
    return np.moveaxis(gt_xyz, (0, 1, 2), (2, 1, 0)).astype(bool)


def dice(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)
    denom = int(pred.sum() + gt.sum())
    return 1.0 if denom == 0 else 2.0 * int((pred & gt).sum()) / denom


def parse_perm(value: str):
    parts = tuple(int(part) for part in value.replace(",", "").replace(" ", ""))
    if sorted(parts) != [0, 1, 2]:
        raise argparse.ArgumentTypeError("Permutation must contain 0,1,2 once, for example 021")
    return parts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--model-dir", default=Path("hf_pe_segmentation_fold_all"), type=Path)
    parser.add_argument("--sample-root", default=Path("sample_data"), type=Path)
    parser.add_argument("--output-root", default=Path("outputs"), type=Path)
    parser.add_argument("--tile-size", default=(128, 256, 256))
    parser.add_argument("--perm", action="append", type=parse_perm)
    parser.add_argument("--save-all", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    case_root = args.sample_root / args.case
    volume, spacing_zyx = load_volume(case_root / "dicom")
    ground_truth = load_ground_truth_zyx(case_root, args.case)
    if ground_truth.shape != volume.shape:
        raise ValueError(f"GT shape {ground_truth.shape} does not match DICOM shape {volume.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PulmonaryEmbolismSegmentationModel.from_pretrained(args.model_dir, trust_remote_code=True)
    inference_config = SlidingWindowInferenceConfig(tile_size=args.tile_size, device=device)

    perms = args.perm or list(itertools.permutations((0, 1, 2)))
    results = []
    print(f"case={args.case} shape={volume.shape} spacing_zyx={spacing_zyx} device={device}")
    for perm in perms:
        start = time.time()
        transposed_volume = np.transpose(volume, perm)
        transposed_spacing = tuple(spacing_zyx[i] for i in perm)
        prediction_transposed = predict_volume_resampled(
            model,
            transposed_volume,
            transposed_spacing,
            inference_config,
        )
        inverse_perm = tuple(int(i) for i in np.argsort(perm))
        prediction = np.transpose(prediction_transposed, inverse_perm)
        if prediction.shape != ground_truth.shape:
            raise ValueError(f"Prediction shape {prediction.shape} does not match GT {ground_truth.shape} for {perm}")

        score = dice(prediction, ground_truth)
        elapsed = time.time() - start
        pred_voxels = int(prediction.sum())
        row = {
            "perm": "".join(str(i) for i in perm),
            "dice": score,
            "prediction_voxels": pred_voxels,
            "seconds": elapsed,
        }
        results.append(row)
        print(
            f"perm={row['perm']} dice={score:.6f} "
            f"pred_voxels={pred_voxels} seconds={elapsed:.1f}"
        )
        if args.save_all:
            save_segmentation(
                args.output_root / f"{args.case}_segmentation_perm{row['perm']}.npz",
                prediction,
            )
        del transposed_volume, prediction_transposed, prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best = max(results, key=lambda item: item["dice"])
    print(
        f"best perm={best['perm']} dice={best['dice']:.6f} "
        f"pred_voxels={best['prediction_voxels']}"
    )


if __name__ == "__main__":
    main()
