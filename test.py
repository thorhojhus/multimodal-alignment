import logging
from datetime import datetime

import hydra
import torch
import wandb
from transformers import AutoProcessor
from config import AppConfig
from dataset.audio_dataset import AudioChatDataset
from models.audio_llm import AudioLLMModel
from omegaconf import DictConfig
from utils import (
    load_checkpoint,
    run_evaluation
)

# set up a module-level logger
logger = logging.getLogger(__name__)

@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(hydra_conf: DictConfig):
    torch.set_float32_matmul_precision('medium')
    torch.cuda.set_per_process_memory_fraction(0.95, device=torch.cuda.current_device())
    logger.info("Hydra Config:\n" + str(hydra_conf))

    eval_cfg = AppConfig(**hydra_conf)
    
    assert eval_cfg.training.resume_from_checkpoint, "Checkpoint path is required for evaluation"
    
    model = AudioLLMModel(eval_cfg)

    load_checkpoint(
        [model.projector, model.audio_encoder.audio_encoder], None, None,
        checkpoint_path=eval_cfg.training.resume_checkpoint_path,
        logger=logger,
        load_lr_from_checkpoint=False
    )

    model.lm = torch.compile(model.lm, dynamic=True)
    model.audio_encoder.audio_encoder = torch.compile(model.audio_encoder.audio_encoder, dynamic=True)
    model.projector = torch.compile(model.projector, dynamic=True)
    if eval_cfg.wandb.enabled:
        wandb.init(
            entity="borkeren",
            project="benchmarks",
            name=f"benchmark-{eval_cfg.model.lm_model_name.split('/', maxsplit=1)[-1]}-processor-{eval_cfg.model.encoder_model_name.split('/', maxsplit=1)[-1]}-dataset-{eval_cfg.dataset.name}-at-{datetime.now()}"
                if eval_cfg.wandb.run_name == "default"
                else eval_cfg.wandb.run_name,
            config=dict(eval_cfg)
        )

    if ("w2v-bert" in eval_cfg.model.encoder_model_name):
        from transformers import AutoFeatureExtractor
        processor = AutoFeatureExtractor.from_pretrained(eval_cfg.model.encoder_model_name, padding_value=0.0)
    else:
        try:
            processor = AutoProcessor.from_pretrained(eval_cfg.model.encoder_model_name)
        except Exception:
            from transformers import AutoFeatureExtractor
            processor = AutoFeatureExtractor.from_pretrained(eval_cfg.model.encoder_model_name)

    test_dataset = AudioChatDataset(
        lm_model=eval_cfg.model.lm_model_name,
        dataset=eval_cfg.dataset.name,
        split=eval_cfg.dataset.test_split,
        tokenizer=model.tokenizer,
        processor=processor,
        T=eval_cfg.model.token_length,
        config=eval_cfg,
    )

    wer, cer, corrected_wer_list = run_evaluation(
        model=model, 
        test_dataset=test_dataset, 
        tokenizer=model.tokenizer, 
        logger=logger, 
        n_samples=None, 
        config=eval_cfg,
        batch_size=eval_cfg.eval.batch_size,
    )

    logger.info(
        f"WER: {wer:.2f} (C: {(sum(corrected_wer_list) / len(corrected_wer_list)):0.4f}) | CER: {cer:.2f} | "
    )

    if eval_cfg.wandb.enabled:
        wandb.log({
            "eval/epoch_wer": wer,
            "eval/epoch_wer_corrected": sum(corrected_wer_list) / len(corrected_wer_list),
            "eval/epoch_cer": cer
        })

if __name__ == "__main__":
    main()