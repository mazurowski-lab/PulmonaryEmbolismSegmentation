# Pulmonary Embolism Segmentation

This repository contains model weights and portable inference code for
pulmonary embolism segmentation from CT pulmonary angiography.

The released model is a 3D Residual Encoder U-Net trained with nnU-Net v2 on
430 patients and tested on 60 distinct patients. The original nnU-Net v2
workflow is still supported, and this repository now also includes a portable
plain-PyTorch/Hugging Face implementation that removes runtime dependencies on
`nnunetv2` and `dynamic-network-architectures`.

- Paper: https://link.springer.com/article/10.1007/s10278-026-01958-4
- Model weights: https://drive.google.com/drive/folders/1wvX-rz_VW2kHsvjlx7IPj00ENwo8Sv56
- Contact: Yixin Zhang, yz696@duke.edu

The model weights are released under a CC-BY 4.0-NC license. Researchers
interested in other licensing options should contact
[MinnHealth](https://www.minnhealth.com/).

## Intended Use

Researchers are welcome to use this model to generate segmentation predictions
or logits on their own CT datasets. Typical uses include:

- Running inference to obtain pulmonary embolism segmentation masks.
- Generating logits for analysis or knowledge distillation.

This model is intended for research use only. It is not a medical device and
should not be used for clinical decision-making without appropriate validation.

## Portable Hugging Face Model Code

The portable implementation is provided in:

- `pulmonary_embolism_segmentation/`
- `configuration_pe_segmentation.py`
- `modeling_pe_segmentation.py`
- `local_architecture.py`
- `config.json`

The architecture is implemented directly in PyTorch while matching the nnU-Net
v2 ResidualEncoderUNet checkpoint structure.

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "yzluka/PulmonaryEmbolismSegmentation",
    trust_remote_code=True,
)
```

The default configuration is copied from the upstream `3d_fullres` plan:

- network: `ResidualEncoderUNet`
- input channels: 1 CT volume channel
- output logits: 2 classes, background and pulmonary embolism
- patch size: `[224, 320, 320]`
- spacing: `[1.0, 0.7373045682907104, 0.7373045682907104]`
- encoder features: `[32, 64, 128, 256, 320, 320, 320]`
- encoder blocks: `[1, 3, 4, 6, 6, 6, 6]`
- decoder convolutions: `[1, 1, 1, 1, 1, 1]`
- CT clipping: `[-195, 305]`
- CT normalization mean/std: `37.060203552246094` / `92.34374237060547`

## Portable Inference

Install the lightweight package:

```bash
pip install -e .
```

For DICOM, NIfTI, or NumPy input, use:

```bash
python scripts/run_inference.py \
  --model-dir hf_pe_segmentation_fold_all \
  --input path/to/input \
  --output outputs/case_segmentation.npz \
  --tile-size 128,256,256
```

The inference helper follows the nnU-Net v2 preprocessing/export order:

1. read image and spacing
2. crop nonzero region
3. CT clip and normalize
4. resample image to the plan spacing
5. run sliding-window prediction
6. resample logits back to the cropped source grid
7. take argmax
8. insert the cropped prediction back into the original image shape

If CUDA runs out of memory, the predictor automatically shrinks the tile size
down to `64,128,128` by default and retries. For very constrained devices, pass
a smaller `--min-tile-size` or run with `--device cpu`.

## Convert nnU-Net Weights

After downloading an nnU-Net checkpoint, convert it into a Hugging Face model
folder:

```bash
python scripts/convert_nnunet_checkpoint.py \
  --checkpoint path/to/checkpoint_final.pth \
  --output-dir hf_pe_segmentation_fold_all
```

The converter accepts common nnU-Net checkpoint dictionaries such as
`network_weights`, `state_dict`, or a raw state dict.

## Original nnU-Net v2 Workflow

The original release uses nnU-Net v2 inference folders. To reproduce that
workflow:

1. Clone or download this repository.
2. Clone nnU-Net v2 from https://github.com/MIC-DKFZ/nnUNet and install it:

   ```bash
   cd nnUNet
   pip install -e .
   ```

3. Download the model weights and place the contents under:

   ```text
   dataset/nnUNet_results/Dataset256_PE/nnUNetTrainer_500epochs__nnUNetResEncUNetPlans_80G__3d_fullres/
   ```

4. Prepare input files in NIfTI format (`.nii` or `.nii.gz`) using nnU-Net's
   `caseID_channelID.nii.gz` convention, for example `PE001_0000.nii.gz`.
5. Fill in the paths in `pred_template.sh` and run inference.

Expected nnU-Net directory structure:

```text
PulmonaryEmbolismSegmentation/
  pred_template.sh
  dataset/
    nnUNet_results/
      Dataset256_PE/
        nnUNetTrainer_500epochs__nnUNetResEncUNetPlans_80G__3d_fullres/
          fold_0/
          fold_1/
          fold_2/
          fold_3/
          fold_4/
          fold_all/
          plans.json
          dataset.json
          dataset_fingerprint.json
```

## Validation

A 40-case validation pass was run on the pixel-level annotated dataset used for
testing portability. Dice scores are computed after resampling predictions back
to the original DICOM grid.

| Group | n | Mean Dice | Median Dice | Min | Max | Dice >= 0.5 | Dice >= 0.7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GE | 20 | 0.5934 | 0.6918 | 0.0016 | 0.8512 | 15 | 10 |
| Toshiba | 20 | 0.4780 | 0.7103 | 0.0000 | 0.8020 | 11 | 11 |
| All | 40 | 0.5357 | 0.7035 | 0.0000 | 0.8512 | 26 | 21 |

Additional input-axis permutation checks were run on low-performing cases
(`04TS`, `20GE`, `01TS`, `17TS`). The original `(z, y, x)` input order gave the
best Dice in each tested case, suggesting these failures are not explained by a
simple array transpose mismatch.

## Citation

If you use resources from this repository in your research, please cite:

```bibtex
@article{zhang2026rethinking,
  title={Rethinking Pulmonary Embolism Segmentation: A Study of Current Approaches and Challenges with an Open Weight Model},
  author={Zhang, Yixin and Chamberlain, Ryan and Ngo, Lawrence and Kramer, Kevin and Mazurowski, Maciej A},
  journal={Journal of Imaging Informatics in Medicine},
  pages={1--13},
  year={2026},
  publisher={Springer},
  doi={10.1007/s10278-026-01958-4},
  url={https://link.springer.com/article/10.1007/s10278-026-01958-4}
}
```
