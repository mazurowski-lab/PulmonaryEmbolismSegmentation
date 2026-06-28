from dataclasses import dataclass
from itertools import product
from math import ceil
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes, gaussian_filter, map_coordinates
from skimage.transform import resize


ANISO_THRESHOLD = 3.0


def normalize_ct(volume: np.ndarray, config) -> np.ndarray:
    volume = volume.astype(np.float32, copy=False)
    volume = np.clip(volume, config.ct_clip_min, config.ct_clip_max)
    return (volume - config.ct_mean) / config.ct_std


def compute_new_shape(old_shape: Sequence[int], old_spacing: Sequence[float], new_spacing: Sequence[float]) -> Tuple[int, ...]:
    return tuple(int(round(spacing / target_spacing * size)) for size, spacing, target_spacing in zip(old_shape, old_spacing, new_spacing))


def get_lowres_axis(spacing: Sequence[float]) -> Optional[int]:
    spacing = np.asarray(spacing)
    axes = np.where(np.max(spacing) / spacing == 1)[0]
    return int(axes[0]) if len(axes) == 1 else None


def determine_separate_z_axis(current_spacing: Sequence[float], new_spacing: Sequence[float]) -> Optional[int]:
    current_spacing = np.asarray(current_spacing)
    new_spacing = np.asarray(new_spacing)
    if np.max(current_spacing) / np.min(current_spacing) > ANISO_THRESHOLD:
        return get_lowres_axis(current_spacing)
    if np.max(new_spacing) / np.min(new_spacing) > ANISO_THRESHOLD:
        return get_lowres_axis(new_spacing)
    return None


def create_nonzero_mask(data: np.ndarray) -> np.ndarray:
    mask = data[0] != 0
    for channel in range(1, data.shape[0]):
        mask |= data[channel] != 0
    return binary_fill_holes(mask)


def crop_to_nonzero(data: np.ndarray) -> Tuple[np.ndarray, Tuple[Tuple[int, int], ...]]:
    mask = create_nonzero_mask(data)
    coords = np.array(np.where(mask))
    if coords.size == 0:
        bbox = tuple((0, dim) for dim in data.shape[1:])
    else:
        mins = coords.min(axis=1)
        maxs = coords.max(axis=1) + 1
        bbox = tuple((int(lo), int(hi)) for lo, hi in zip(mins, maxs))
    slicer = (slice(None),) + tuple(slice(lo, hi) for lo, hi in bbox)
    return data[slicer], bbox


def insert_crop(segmentation: np.ndarray, bbox: Sequence[Tuple[int, int]], original_shape: Sequence[int]) -> np.ndarray:
    result = np.zeros(original_shape, dtype=segmentation.dtype)
    slicer = tuple(slice(lo, hi) for lo, hi in bbox)
    result[slicer] = segmentation
    return result


def _resize_segmentation(segmentation: np.ndarray, new_shape: Sequence[int], order: int) -> np.ndarray:
    if order == 0:
        return resize(segmentation, new_shape, order=0, mode="edge", anti_aliasing=False, preserve_range=True)
    result = np.zeros(new_shape, dtype=segmentation.dtype)
    for label in np.unique(segmentation):
        resized = resize(
            (segmentation == label).astype(float),
            new_shape,
            order=order,
            mode="edge",
            anti_aliasing=False,
            preserve_range=True,
        )
        result[resized >= 0.5] = label
    return result


def _resize_channel(channel: np.ndarray, new_shape: Sequence[int], is_seg: bool, order: int) -> np.ndarray:
    if is_seg:
        return _resize_segmentation(channel, new_shape, order)
    return resize(channel, new_shape, order=order, mode="edge", anti_aliasing=False, preserve_range=True)


def resample_nnunet(
    data: np.ndarray,
    new_shape: Sequence[int],
    current_spacing: Sequence[float],
    new_spacing: Sequence[float],
    is_seg: bool = False,
    order: int = 3,
    order_z: int = 0,
) -> np.ndarray:
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    assert data.ndim == 4, "data must be channel-first: (c, z, y, x)"

    old_shape = tuple(data.shape[1:])
    new_shape = tuple(int(i) for i in new_shape)
    if old_shape == new_shape:
        return data

    axis = determine_separate_z_axis(current_spacing, new_spacing)
    output = np.zeros((data.shape[0], *new_shape), dtype=data.dtype)
    data_float = data.astype(float, copy=False)

    if axis is None:
        for c in range(data.shape[0]):
            output[c] = _resize_channel(data_float[c], new_shape, is_seg, order)
        return output

    shape = np.array(old_shape)
    new_shape_array = np.array(new_shape)
    if axis == 0:
        plane_shape = new_shape_array[1:]
    elif axis == 1:
        plane_shape = new_shape_array[[0, 2]]
    else:
        plane_shape = new_shape_array[:-1]

    for c in range(data.shape[0]):
        intermediate_shape = new_shape_array.copy()
        intermediate_shape[axis] = shape[axis]
        intermediate = np.zeros(tuple(intermediate_shape), dtype=float)
        for idx in range(shape[axis]):
            if axis == 0:
                intermediate[idx] = _resize_channel(data_float[c, idx], plane_shape, is_seg, order)
            elif axis == 1:
                intermediate[:, idx] = _resize_channel(data_float[c, :, idx], plane_shape, is_seg, order)
            else:
                intermediate[:, :, idx] = _resize_channel(data_float[c, :, :, idx], plane_shape, is_seg, order)

        if shape[axis] == new_shape_array[axis]:
            output[c] = intermediate
            continue

        rows, cols, depth = new_shape
        orig_rows, orig_cols, orig_depth = intermediate.shape
        row_scale = orig_rows / rows
        col_scale = orig_cols / cols
        depth_scale = orig_depth / depth
        map_rows, map_cols, map_depth = np.mgrid[:rows, :cols, :depth]
        coord_map = np.array(
            [
                row_scale * (map_rows + 0.5) - 0.5,
                col_scale * (map_cols + 0.5) - 0.5,
                depth_scale * (map_depth + 0.5) - 0.5,
            ]
        )
        if not is_seg or order_z == 0:
            output[c] = map_coordinates(intermediate, coord_map, order=order_z, mode="nearest")
        else:
            for label in np.unique(intermediate):
                output[c][
                    np.round(map_coordinates((intermediate == label).astype(float), coord_map, order=order_z, mode="nearest")) > 0.5
                ] = label
    return output


def compute_steps(image_size: Sequence[int], tile_size: Sequence[int], overlap: float) -> Tuple[Tuple[int, ...], ...]:
    steps = []
    for image_dim, tile_dim in zip(image_size, tile_size):
        if image_dim <= tile_dim:
            steps.append((0,))
            continue
        target_step = max(1, int(tile_dim * (1.0 - overlap)))
        num_steps = int(ceil((image_dim - tile_dim) / target_step)) + 1
        actual_step = (image_dim - tile_dim) / max(1, num_steps - 1)
        steps.append(tuple(int(round(actual_step * i)) for i in range(num_steps)))
    return tuple(steps)


def gaussian_importance_map(tile_size: Sequence[int], sigma_scale: float = 1.0 / 8.0) -> torch.Tensor:
    tmp = np.zeros(tuple(tile_size), dtype=np.float32)
    tmp[tuple(size // 2 for size in tile_size)] = 1
    sigmas = [size * sigma_scale for size in tile_size]
    weight = gaussian_filter(tmp, sigmas, 0, mode="constant", cval=0)
    weight /= np.max(weight)
    tensor = torch.from_numpy(weight.astype(np.float32, copy=False))
    tensor[tensor == 0] = torch.min(tensor[tensor != 0])
    return tensor


def pad_to_tile_size(volume: torch.Tensor, tile_size: Sequence[int]) -> Tuple[torch.Tensor, Tuple[slice, slice, slice]]:
    spatial = volume.shape[-3:]
    pad_after = [max(tile - dim, 0) for dim, tile in zip(spatial, tile_size)]
    if any(pad_after):
        volume = F.pad(volume, (0, pad_after[2], 0, pad_after[1], 0, pad_after[0]))
    crop = tuple(slice(0, dim) for dim in spatial)
    return volume, crop


def iter_tiles(image_size: Sequence[int], tile_size: Sequence[int], overlap: float) -> Iterable[Tuple[slice, slice, slice]]:
    for z, y, x in product(*compute_steps(image_size, tile_size, overlap)):
        yield (
            slice(z, z + tile_size[0]),
            slice(y, y + tile_size[1]),
            slice(x, x + tile_size[2]),
        )


def shrink_tile(tile_size: Sequence[int], min_tile_size: Sequence[int]) -> Optional[Tuple[int, int, int]]:
    divisibility = (32, 64, 64)
    candidates = []
    for axis, value in enumerate(tile_size):
        if value > min_tile_size[axis]:
            next_tile = list(tile_size)
            step = divisibility[axis]
            next_value = max(min_tile_size[axis], int(value * 0.75) // step * step)
            next_tile[axis] = next_value
            candidates.append(tuple(next_tile))
    if not candidates:
        return None
    return min(candidates, key=np.prod)


def normalize_tile_size(
    image_size: Sequence[int],
    requested_tile_size: Sequence[int],
    divisibility: Sequence[int] = (32, 64, 64),
) -> Tuple[int, int, int]:
    tile_size = []
    for image_dim, requested_dim, divisor in zip(image_size, requested_tile_size, divisibility):
        if requested_dim % divisor != 0:
            raise ValueError(
                f"Tile size {tuple(requested_tile_size)} is incompatible with network strides. "
                f"Expected divisibility by {tuple(divisibility)} for z,y,x."
            )
        if image_dim >= requested_dim:
            tile_size.append(requested_dim)
        else:
            tile_size.append(int(ceil(image_dim / divisor) * divisor))
    return tuple(tile_size)


@dataclass
class SlidingWindowInferenceConfig:
    tile_size: Tuple[int, int, int] = (128, 256, 256)
    min_tile_size: Tuple[int, int, int] = (64, 128, 128)
    overlap: float = 0.5
    use_amp: bool = True
    device: Optional[str] = None
    empty_cache_between_tiles: bool = False


def _predict_with_tile_size(
    model,
    volume: torch.Tensor,
    tile_size: Sequence[int],
    overlap: float,
    device: torch.device,
    use_amp: bool,
    empty_cache_between_tiles: bool,
) -> torch.Tensor:
    volume, crop = pad_to_tile_size(volume, tile_size)
    image_size = volume.shape[-3:]
    num_classes = model.config.num_labels

    logits_sum = torch.zeros((num_classes, *image_size), dtype=torch.float32, device="cpu")
    weight_sum = torch.zeros(image_size, dtype=torch.float32, device="cpu")
    importance = gaussian_importance_map(tile_size).to(device)

    model.eval()
    with torch.inference_mode():
        for tile in iter_tiles(image_size, tile_size, overlap):
            patch = volume[(slice(None), slice(None), *tile)].to(device, non_blocking=True)
            amp_enabled = use_amp and device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(patch).logits[0]
            weighted_logits = (logits.float() * importance).cpu()
            logits_sum[(slice(None), *tile)] += weighted_logits
            weight_sum[tile] += importance.cpu()
            del patch, logits, weighted_logits
            if empty_cache_between_tiles and device.type == "cuda":
                torch.cuda.empty_cache()

    logits_sum /= torch.clamp(weight_sum.unsqueeze(0), min=1e-6)
    return logits_sum[(slice(None), *crop)]


def predict_logits(model, volume: np.ndarray, inference_config: Optional[SlidingWindowInferenceConfig] = None) -> torch.Tensor:
    inference_config = inference_config or SlidingWindowInferenceConfig()
    device = torch.device(inference_config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    tensor = torch.from_numpy(volume.astype(np.float32, copy=False)[None, None])
    tile_size = normalize_tile_size(tensor.shape[-3:], inference_config.tile_size)

    while True:
        try:
            return _predict_with_tile_size(
                model=model,
                volume=tensor,
                tile_size=tile_size,
                overlap=inference_config.overlap,
                device=device,
                use_amp=inference_config.use_amp,
                empty_cache_between_tiles=inference_config.empty_cache_between_tiles,
            )
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda":
                raise
            torch.cuda.empty_cache()
            next_tile_size = shrink_tile(tile_size, inference_config.min_tile_size)
            if next_tile_size is None:
                raise RuntimeError(
                    f"CUDA out of memory at minimum tile size {tile_size}. "
                    "Use CPU inference or lower min_tile_size."
                )
            tile_size = next_tile_size


def predict_volume(model, volume: np.ndarray, inference_config: Optional[SlidingWindowInferenceConfig] = None) -> np.ndarray:
    normalized = normalize_ct(volume, model.config)
    logits = predict_logits(model, normalized, inference_config)
    return torch.argmax(logits, dim=0).numpy().astype(np.uint8)


def predict_volume_resampled(
    model,
    volume: np.ndarray,
    source_spacing_zyx: Sequence[float],
    inference_config: Optional[SlidingWindowInferenceConfig] = None,
) -> np.ndarray:
    target_spacing_zyx = tuple(model.config.spacing)
    original_shape = volume.shape
    data = volume.astype(np.float32, copy=False)[None]
    cropped, bbox = crop_to_nonzero(data)
    cropped_shape = cropped.shape[1:]
    normalized = cropped.copy()
    normalized[0] = normalize_ct(normalized[0], model.config)
    resampled_shape = compute_new_shape(cropped_shape, source_spacing_zyx, target_spacing_zyx)
    resampled = resample_nnunet(
        normalized,
        new_shape=resampled_shape,
        current_spacing=source_spacing_zyx,
        new_spacing=target_spacing_zyx,
        is_seg=False,
        order=3,
        order_z=0,
    )
    logits = predict_logits(model, resampled[0], inference_config)
    logits_cropped = resample_nnunet(
        logits,
        new_shape=cropped_shape,
        current_spacing=target_spacing_zyx,
        new_spacing=source_spacing_zyx,
        is_seg=False,
        order=1,
        order_z=0,
    )
    segmentation_cropped = np.argmax(logits_cropped, axis=0).astype(np.uint8, copy=False)
    return insert_crop(segmentation_cropped, bbox, original_shape)
