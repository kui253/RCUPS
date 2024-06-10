import torch.nn.functional as F

from transformers.models.marian.modeling_marian import (
    MarianSinusoidalPositionalEmbedding as SinusoidalPositionalEmbedding,
)


from transformers import BartConfig


name_2_activation_fn_mapping = {
    "tanh": F.tanh,
    "relu": F.relu,
}


class RCUPSConfig(BartConfig):

    def __init__(
        self,
        backbone_model="../pretrained_model/bart_large",
        # all_bart_base config
        conv_activation_fn="relu",
        num_beams=5,
        rezero=1,
        max_length=100,
        min_length=5,
        utt_pooling="average",
        gt_pos_embed="",
        **kwargs,
    ):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                setattr(self, k, v)

        pretrained_model_config = BartConfig.from_pretrained(backbone_model)
        for k, v in vars(pretrained_model_config).items():
            if not hasattr(self, k):
                setattr(self, k, v)
        self.gt_pos_embed = gt_pos_embed
        self.conv_activation_fn = conv_activation_fn
        self.utt_pooling = utt_pooling
        self.backbone_model = backbone_model
        self.min_length = min_length
        self.num_beams = num_beams
        self.max_length = max_length
        self.rezero = rezero
