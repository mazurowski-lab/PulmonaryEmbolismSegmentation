# Pulmonary Embolism Segmentation

This repository contains model weights and portable inference code for
pulmonary embolism segmentation from CT pulmonary angiography.

The released model is a 3D Residual Encoder U-Net trained with nnU-Net v2 on
430 patients and tested on 60 distinct patients. The original nnU-Net v2
workflow is still supported, and this repository now also includes a portable
plain-PyTorch/Hugging Face implementation that removes runtime dependencies on
`nnunetv2` and `dynamic-network-architectures`.

- Paper: https://link.springer.com/article/10.1007/s10278-026-01958-4
- Hugging Face model: https://huggingface.co/yzluka/PulmonaryEmbolismSegmentation
- Hugging Face model card: https://huggingface.co/yzluka/PulmonaryEmbolismSegmentation#model-card
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

## Two Supported Inference Channels

This repository supports two ways to run the model:

- **Hugging Face channel, recommended for most users:** portable PyTorch model
  loading and sliding-window inference without installing nnU-Net.
- **nnU-Net channel, for reproducing the original release:** original nnU-Net
  v2 folder layout, checkpoints, and `pred_template.sh`.

## Channel 1: Hugging Face

For most users, the easiest way to use the model is the Hugging Face version:

```bash
git clone https://github.com/mazurowski-lab/PulmonaryEmbolismSegmentation.git
cd PulmonaryEmbolismSegmentation
pip install -e .[inference]
```

Then run inference directly from the published model repo:

```bash
python scripts/run_inference.py \
  --model-dir yzluka/PulmonaryEmbolismSegmentation \
  --input path/to/dicom_or_nifti_or_numpy \
  --output outputs/case_segmentation.npz \
  --tile-size 128,256,256
```

This path downloads the converted model from Hugging Face and does not require
installing nnU-Net v2 or `dynamic-network-architectures`.

## Model Code

The portable implementation is provided in this repository:

- `pulmonary_embolism_segmentation/`
- `configuration_pe_segmentation.py`
- `modeling_pe_segmentation.py`
- `local_architecture.py`
- `config.json`

The architecture is implemented directly in PyTorch while matching the nnU-Net
v2 ResidualEncoderUNet checkpoint structure.

## Python Loading

To load the model architecture and weights in Python:

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

## Preprocessing

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

Manual conversion is only needed if you want to rebuild the Hugging Face model
folder from the original nnU-Net checkpoints. After downloading an nnU-Net
checkpoint, convert it with:

```bash
python scripts/convert_nnunet_checkpoint.py \
  --checkpoint path/to/checkpoint_final.pth \
  --output-dir hf_pe_segmentation_fold_all
```

The converter accepts common nnU-Net checkpoint dictionaries such as
`network_weights`, `state_dict`, or a raw state dict.

## Channel 2: Original nnU-Net v2 Workflow

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

## Evaluation

Evaluation details are described in the associated paper. This README does not
report standalone evaluation metrics because performance should be interpreted
in the context of the dataset, preprocessing, and intended research use.

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
