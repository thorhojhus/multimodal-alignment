import torch 
import torch.nn as nn
from sparktts.models.bicodec import BiCodec
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from numpy.typing import NDArray
from torch.types import Tensor
from typing import Dict

class BiCodecInherit(BiCodec):
    def __init__(self, *args, **kwargs):
        super(BiCodecInherit, self).__init__(*args, **kwargs)
    
    @torch.no_grad()
    def tokenize(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Tokenizes the input audio into semantic and global tokens.

        Args:
            batch (dict): The input audio features and reference waveform.

        Returns:
            tuple: Semantic tokens and global tokens.
        """
        feat = batch["feat"]

        z = self.encoder(feat.transpose(1, 2))
        semantic_tokens = self.quantizer.tokenize(z)

        return semantic_tokens

class BiCodecWrapper(nn.Module):
    def __init__(self, bicodec_dir="weights/BiCodec", wav2vec_model="facebook/wav2vec2-large-xlsr-53"):
        super(BiCodecWrapper, self).__init__()
        self.bicodec = BiCodecInherit.load_from_checkpoint(bicodec_dir).to("cuda", torch.bfloat16)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model)
        self.feature_extractor = Wav2Vec2Model.from_pretrained(wav2vec_model).to("cuda", torch.bfloat16)
        self.feature_extractor.config.output_hidden_states = True
        self.device = "cuda"
        self.dtype = torch.bfloat16

    def extract_wav2vec2_features(self, wavs: NDArray) -> NDArray:
        """extract wav2vec2 features"""

        feat = self.feature_extractor(wavs.to(self.feature_extractor.device))
        feats_mix = (
            feat.hidden_states[11] + feat.hidden_states[14] + feat.hidden_states[16]
        ) / 3

        return feats_mix

    def forward(self, wav : Tensor) -> Tensor:
        feat = self.extract_wav2vec2_features(wav)
        batch = {
            "wav": wav.unsqueeze(0).float().to(self.device),
            "feat": feat.to(self.device),
        }
        semantic_tokens = self.bicodec.tokenize(batch)
        return semantic_tokens