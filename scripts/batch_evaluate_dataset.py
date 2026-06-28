import argparse
import csv
import shutil
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

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


def discover_cases(zip_path: Path):
    cases = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".dcm"):
                continue
            parts = info.filename.split("/")
            if len(parts) < 4:
                continue
            case = parts[3]
            if case.endswith(("GE", "TS")):
                vendor = "GE" if case.endswith("GE") else "TOSHIBA"
                cases.setdefault(case, vendor)
    return dict(sorted(cases.items()))


def extract_case(zip_path: Path, case: str, output_root: Path):
    case_root = output_root / case
    dicom_dir = case_root / "dicom"
    gt_dir = case_root / "ground_truth"
    gt_nifti = gt_dir / f"{case}.nii.gz"
    gt_csv = gt_dir / f"{case}.csv"
    if dicom_dir.exists() and gt_nifti.exists():
        return case_root

    dicom_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            target = None
            if f"/{case}/" in name and name.lower().endswith(".dcm"):
                target = dicom_dir / PurePosixPath(name).name
            elif name.endswith(f"/{case}.nii.gz"):
                target = gt_nifti
            elif name.endswith(f"/{case}.csv") or name.endswith(f"/{case} .csv"):
                target = gt_csv
            if target is None or target.exists():
                continue
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    if not gt_nifti.exists():
        raise FileNotFoundError(f"Missing ground-truth NIfTI for {case}")
    return case_root


def load_ground_truth(path: Path):
    data_xyz = np.asanyarray(nib.load(str(path)).dataobj)
    return np.moveaxis(data_xyz, (0, 1, 2), (2, 1, 0)).astype(bool)


def compute_metrics(prediction: np.ndarray, ground_truth: np.ndarray):
    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)
    intersection = int((pred & gt).sum())
    pred_voxels = int(pred.sum())
    gt_voxels = int(gt.sum())
    dice = 1.0 if pred_voxels + gt_voxels == 0 else 2.0 * intersection / (pred_voxels + gt_voxels)
    return pred_voxels, gt_voxels, intersection, dice


def write_results(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case",
        "vendor",
        "shape",
        "spacing_zyx",
        "prediction_voxels",
        "ground_truth_voxels",
        "intersection_voxels",
        "dice",
        "seconds",
        "output",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=Path("downloads/data.zip"), type=Path)
    parser.add_argument("--model-dir", default=Path("hf_pe_segmentation_fold_all"), type=Path)
    parser.add_argument("--sample-root", default=Path("sample_data"), type=Path)
    parser.add_argument("--output-root", default=Path("outputs"), type=Path)
    parser.add_argument("--csv", default=Path("outputs/all_case_stats.csv"), type=Path)
    parser.add_argument("--tile-size", default=(128, 256, 256))
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cases = discover_cases(args.zip)
    if args.limit is not None:
        cases = dict(list(cases.items())[: args.limit])
    print(f"Discovered {len(cases)} cases")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = PulmonaryEmbolismSegmentationModel.from_pretrained(args.model_dir, trust_remote_code=True)
    inference_config = SlidingWindowInferenceConfig(tile_size=args.tile_size, device=device)

    rows = []
    for index, (case, vendor) in enumerate(cases.items(), start=1):
        print(f"[{index}/{len(cases)}] {case} ({vendor})")
        case_root = extract_case(args.zip, case, args.sample_root)
        output_path = args.output_root / f"{case}_segmentation_nnunet_preprocess.npz"
        volume = None
        spacing_zyx = None

        started = time.time()
        if output_path.exists() and not args.overwrite:
            prediction = np.load(output_path)["segmentation"]
            elapsed = 0.0
            print(f"  using existing prediction {output_path}")
        else:
            volume, spacing_zyx = load_volume(case_root / "dicom")
            if spacing_zyx is None:
                raise RuntimeError(f"No spacing for {case}")
            prediction = predict_volume_resampled(model, volume, spacing_zyx, inference_config)
            save_segmentation(output_path, prediction)
            elapsed = time.time() - started
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if volume is None:
            volume, spacing_zyx = load_volume(case_root / "dicom")
        gt = load_ground_truth(case_root / "ground_truth" / f"{case}.nii.gz")
        pred_voxels, gt_voxels, intersection, dice = compute_metrics(prediction, gt)
        row = {
            "case": case,
            "vendor": vendor,
            "shape": "x".join(str(i) for i in prediction.shape),
            "spacing_zyx": "x".join(f"{float(i):.6g}" for i in spacing_zyx),
            "prediction_voxels": pred_voxels,
            "ground_truth_voxels": gt_voxels,
            "intersection_voxels": intersection,
            "dice": f"{dice:.6f}",
            "seconds": f"{elapsed:.2f}",
            "output": str(output_path),
        }
        rows.append(row)
        write_results(args.csv, rows)
        print(f"  dice={dice:.6f} pred={pred_voxels} gt={gt_voxels} seconds={elapsed:.1f}")
        del volume

    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
