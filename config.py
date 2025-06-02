"""
Configuration utilities to be used elsewhere in the project.
"""
from pydantic import BaseModel
from typing import Literal, Optional

# Define Pydantic models for strong typing
class HydraJobConfig(BaseModel):
    chdir: bool

class HydraRunConfig(BaseModel):
    dir: str

class HydraConfig(BaseModel):
    job: HydraJobConfig
    run: HydraRunConfig

class WandbConfig(BaseModel):
    enabled: bool
    run_name: str
    upload_audio: bool
    project: Optional[str] = None  # Optional field, defaults to None

class ModelConfig(BaseModel):
    downsample_factor: int
    lm_model_name: str
    load_in_4bit: bool
    encoder_model_name: str
    projector: Literal["simple", "tinyalign", "linear", "conv", "transformer", "simple-linear"]
    token_length: int
    audio_tokenizer: bool
    audio_tokenizer_vocab_size: int
    train_llm: bool
    train_encoder: bool
    expand_embeds: bool
    train_encoder_wav2vec: bool
    train_encoder_full: bool
    use_hierarchical: bool = True

class TrainingConfig(BaseModel):
    num_epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    gradient_clip_norm: float
    lr: float
    warmup_steps: int
    checkpoint_dir: str
    evaluate_every_n_steps: int
    resume_from_checkpoint: bool
    resume_checkpoint_path: str
    load_lr_from_checkpoint: bool
    log_every_n_steps: int
    loss: Literal["ce", "kl_loss", "embed_loss"]
    kl_temp: float
    do_phonemize: bool
    seed: Optional[int]
    run_count: Optional[int]

class EvalConfig(BaseModel):
    n_samples: int
    batch_size: Optional[int]

class DatasetConfig(BaseModel):
    augment: str
    name: str
    split: str
    test_split: str
    do_filter: bool
    min_audio_length: float
    max_audio_length: float

class AppConfig(BaseModel):
    hydra: Optional[HydraConfig] = None
    wandb: WandbConfig
    model: ModelConfig
    training: TrainingConfig
    eval: EvalConfig
    dataset: DatasetConfig