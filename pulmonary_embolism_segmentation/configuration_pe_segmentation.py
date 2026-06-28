from transformers import PretrainedConfig


class PulmonaryEmbolismSegmentationConfig(PretrainedConfig):
    model_type = "pulmonary-embolism-segmentation"

    def __init__(
        self,
        input_channels=1,
        num_labels=2,
        patch_size=None,
        spacing=None,
        features_per_stage=None,
        kernel_sizes=None,
        strides=None,
        n_blocks_per_stage=None,
        n_conv_per_stage_decoder=None,
        conv_bias=True,
        norm_eps=1e-5,
        norm_affine=True,
        leaky_relu_inplace=True,
        deep_supervision=False,
        label2id=None,
        id2label=None,
        ct_clip_min=-195.0,
        ct_clip_max=305.0,
        ct_mean=37.060203552246094,
        ct_std=92.34374237060547,
        **kwargs,
    ):
        label2id = label2id or {"background": 0, "pulmonary_embolism": 1}
        id2label = id2label or {str(v): k for k, v in label2id.items()}
        super().__init__(label2id=label2id, id2label=id2label, **kwargs)

        self.input_channels = input_channels
        self.num_labels = num_labels
        self.patch_size = patch_size or [224, 320, 320]
        self.spacing = spacing or [1.0, 0.7373045682907104, 0.7373045682907104]
        self.features_per_stage = features_per_stage or [32, 64, 128, 256, 320, 320, 320]
        self.kernel_sizes = kernel_sizes or [[3, 3, 3]] * 7
        self.strides = strides or [
            [1, 1, 1],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [1, 2, 2],
        ]
        self.n_blocks_per_stage = n_blocks_per_stage or [1, 3, 4, 6, 6, 6, 6]
        self.n_conv_per_stage_decoder = n_conv_per_stage_decoder or [1, 1, 1, 1, 1, 1]
        self.conv_bias = conv_bias
        self.norm_eps = norm_eps
        self.norm_affine = norm_affine
        self.leaky_relu_inplace = leaky_relu_inplace
        self.deep_supervision = deep_supervision

        self.ct_clip_min = ct_clip_min
        self.ct_clip_max = ct_clip_max
        self.ct_mean = ct_mean
        self.ct_std = ct_std
