from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_pe_segmentation import PulmonaryEmbolismSegmentationConfig


@dataclass
class SegmentationModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: torch.Tensor = None
    deep_supervision_logits: Optional[Tuple[torch.Tensor, ...]] = None


class PulmonaryEmbolismSegmentationModel(PreTrainedModel):
    config_class = PulmonaryEmbolismSegmentationConfig
    base_model_prefix = "segmentation_model"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False
    _tied_weights_keys = []
    all_tied_weights_keys = {}
    _keys_to_ignore_on_load_missing = [
        r"segmentation_model\..*\.all_modules\..*",
        r"segmentation_model\.decoder\.encoder\..*",
    ]

    def __init__(self, config: PulmonaryEmbolismSegmentationConfig):
        super().__init__(config)
        self.segmentation_model = self._build_network(config)

    @staticmethod
    def _build_network(config: PulmonaryEmbolismSegmentationConfig) -> nn.Module:
        from .local_architecture import ResidualEncoderUNet

        return ResidualEncoderUNet(
            input_channels=config.input_channels,
            features_per_stage=config.features_per_stage,
            kernel_sizes=config.kernel_sizes,
            strides=config.strides,
            n_blocks_per_stage=config.n_blocks_per_stage,
            num_classes=config.num_labels,
            n_conv_per_stage_decoder=config.n_conv_per_stage_decoder,
            conv_bias=config.conv_bias,
            norm_eps=config.norm_eps,
            norm_affine=config.norm_affine,
            deep_supervision=config.deep_supervision,
        )

    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None):
        outputs = self.segmentation_model(pixel_values)
        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]
            deep_supervision_logits = tuple(outputs[1:])
        else:
            logits = outputs
            deep_supervision_logits = None

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels.long())

        return SegmentationModelOutput(
            loss=loss,
            logits=logits,
            deep_supervision_logits=deep_supervision_logits,
        )
