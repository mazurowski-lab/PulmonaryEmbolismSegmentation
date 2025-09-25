# Model Weights for Pulmonary Embolism (PE) Segmentation

This repository contains **model weights trained on 430, and tested on 60 distinct patients** for **Pulmonary Embolism (PE) segmentation** using [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet). The [model weights](https://drive.google.com/drive/folders/1wvX-rz_VW2kHsvjlx7IPj00ENwo8Sv56) are released for **Non-Commercial** use. 

Researchers are welcome to use this model to generate segmentation predictions (logits or masks) on their own CT datasets.  
Typical uses include:
- Running inference to obtain segmentation masks for analysis.
- Generating logits to regularize the training of other models as part of a knowledge distillation pipeline.

If you used resources from this repository in your research, please cite the following work:
```bibtex
@misc{zhang2025rethinkingpulmonaryembolismsegmentation,
      title={Rethinking Pulmonary Embolism Segmentation: A Study of Current Approaches and Challenges with an Open Weight Model}, 
      author={Yixin Zhang and Ryan Chamberlain and Lawrence Ngo and Kevin Kramer and Maciej A. Mazurowski},
      year={2025},
      eprint={2509.18308},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.18308}, 
}
```
---

## Environment

- Python $\geq$ 3.9  
- PyTorch $\geq$ 2.4.0
- CUDA $\geq$ 12.1, CUDNN 9, (Recommend GPU Memory $\geq$ 24 GB)
- nnUNetV2 (tested under commit `8c4184d`)
  
## Directory structure
```text
PulmonaryEmbolismSegmentation
└── pred_template.sh
└── nnUNet/
└── dataset/
   └── nnUNet_results/
       └── Dataset256_PE/
           └── nnUNetTrainer_500epochs__nnUNetResEncUNetPlans_80G__3d_fullres/
               └── fold_0
               └── fold_...
               └── fold_all
               └── plans.json
               └── dataset.json
               └── dataset_fingerprint.json

 ```            
## Get Started

1. **Download this repository**  
   Clone or download this repository so that the directory structure and bash scripts are available.
2. **clone nnUNETV2 from the [source](https://github.com/MIC-DKFZ/nnUNet) and install by running**    
   
   ```code
   cd nnUNet
   pip install -e .
   ```
3. **Download our model [weights](https://drive.google.com/drive/folders/1wvX-rz_VW2kHsvjlx7IPj00ENwo8Sv56) and place the content under** `nnUNetTrainer_500epochs__nnUNetResEncUNetPlans_80G__3d_fullres/`

Now, you are ready to run our model on your dataset:

1. **Prepare your data**  
   - Input files must be in **NIfTI format (`.nii` or `.nii.gz`)**.  
   - File naming convention: `caseID_channelID.nii.gz` (e.g., `PE001_0000.nii.gz`). The digt for caseID do not need to be consecutive).  
   - Channels must match the training setup defined in `dataset.json`.  

2. **Run inference**  
   You can now run inference after specifying the paths left blank in the script template `pred_template.sh`
   
