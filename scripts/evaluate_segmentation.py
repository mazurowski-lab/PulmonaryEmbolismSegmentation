import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_prediction(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        if "segmentation" in data.files:
            return data["segmentation"]
        if len(data.files) == 1:
            return data[data.files[0]]
        raise ValueError(f"Could not choose prediction array from {path}: {data.files}")
    raise ValueError("Prediction must be .npy or .npz.")


def load_ground_truth_nifti_zyx(path: Path) -> np.ndarray:
    data_xyz = np.asanyarray(nib.load(str(path)).dataobj)
    return np.moveaxis(data_xyz, (0, 1, 2), (2, 1, 0))


def dice_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)
    denominator = int(prediction.sum() + ground_truth.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * int((prediction & ground_truth).sum()) / denominator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    prediction = load_prediction(args.prediction)
    ground_truth = load_ground_truth_nifti_zyx(args.ground_truth)
    if prediction.shape != ground_truth.shape:
        raise ValueError(f"Shape mismatch: prediction={prediction.shape}, ground_truth={ground_truth.shape}")

    prediction_mask = prediction.astype(bool)
    ground_truth_mask = ground_truth.astype(bool)
    intersection = int((prediction_mask & ground_truth_mask).sum())
    print(f"prediction_shape={prediction.shape}")
    print(f"prediction_voxels={int(prediction_mask.sum())}")
    print(f"ground_truth_voxels={int(ground_truth_mask.sum())}")
    print(f"intersection_voxels={intersection}")
    print(f"dice={dice_score(prediction, ground_truth):.6f}")


if __name__ == "__main__":
    main()
