import functools
import logging
import time
from datetime import timedelta
import hydra
import torch
from torchinfo import summary
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_linear_schedule_with_warmup
from dataset.audio_dataset import AudioChatDataset, collate_fn
from models.audio_llm import AudioLLMModel
from utils import (
    print_parameter_count,
    run_evaluation,
    calculate_eval_loss,
    run_evaluation_with_loss
)

# set up a module-level logger
logger = logging.getLogger(__name__)

def setup_wandb(runtime_config: DictConfig) -> None:
    wandb.init(
        entity="borkeren",
        project="vocab-size-sweep",
        name=f"model-{runtime_config.model.lm_model_name.split('/', maxsplit=1)[-1]}-processor-{runtime_config.model.encoder_model_name.split('/', maxsplit=1)[-1]}-dataset-{runtime_config.dataset.name}"
             if runtime_config.wandb.run_name == "default"
             else runtime_config.wandb.run_name,
        config=dict(runtime_config),
    )

def train_multiple(cfg: DictConfig) -> None:
    if cfg.wandb.enabled:
        setup_wandb(cfg)
    OmegaConf.update(cfg, "model.audio_tokenizer_vocab_size", wandb.config["vocab_size"])
    if cfg.wandb.enabled:
        wandb.run.name = f"{cfg.wandb.run_name}-({wandb.config['vocab_size']})"

    model = AudioLLMModel(cfg)
    summary(model)
    
    # freeze non-trainable components
    for param in model.lm.parameters():
        param.requires_grad = False    

    # create dataset and processor
    if "w2v-bert" in cfg.model.encoder_model_name:
        from transformers import AutoFeatureExtractor
        processor = AutoFeatureExtractor.from_pretrained(cfg.model.encoder_model_name)
    else:
        processor = AutoProcessor.from_pretrained(cfg.model.encoder_model_name)
    
    dataset = AudioChatDataset(
        lm_model=cfg.model.lm_model_name,
        dataset=cfg.dataset.name,
        split=cfg.dataset.split,
        tokenizer=model.tokenizer,
        processor=processor,
        T=cfg.model.token_length,
    )
    
    test_dataset = AudioChatDataset(
        lm_model=cfg.model.lm_model_name,
        dataset=cfg.dataset.name,
        split="test.clean",
        tokenizer=model.tokenizer,
        processor=processor,
        T=cfg.model.token_length,
    )
    
    print_parameter_count(model, logger)
    
    trainable_params = [
        {"params": list(model.projector.parameters())},
    ]
    
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.training.lr)
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.training.batch_size, 
        collate_fn=functools.partial(collate_fn, tokenizer=model.tokenizer, T=dataset.T), 
        shuffle=True,
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True
    )
    
    # calculate total number of training steps
    total_steps = len(dataloader) * cfg.training.num_epochs
    
    # create the scheduler using warmup_steps from config
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=total_steps
    )

    # load checkpoint if specified
    global_step = 0
    
    for epoch in range(cfg.training.num_epochs):
        model.train()
        total_loss = 0
        epoch_start_time = time.monotonic()
        batch_start_time = time.monotonic()
        
        for batch_idx, batch in enumerate(dataloader):
            # move data to GPU
            input_ids = batch["input_ids"].to("cuda", non_blocking=True)
            if isinstance(batch["audio_features"], dict):
                # for wav2vec-bert/seamless models
                audio_features = {
                    'input_features': batch["audio_features"]["input_features"].to("cuda", torch.bfloat16, non_blocking=True),
                    'attention_mask': batch["audio_features"]["attention_mask"].to("cuda", non_blocking=True)
                }
            else:
                # for other models (Whisper, CLAP, wav2vec)
                audio_features = batch["audio_features"].to("cuda", torch.bfloat16, non_blocking=True)
            labels = batch["labels"].to("cuda", non_blocking=True)
            alt_input_ids = batch["alt_input_ids"].to("cuda", non_blocking=True)
            alt_labels = batch["alt_labels"].to("cuda", non_blocking=True)
            attention_mask = batch["attention_mask"].to("cuda", non_blocking=True)
            insertion_indices = batch["insertion_indices"]
            text_ids = batch["text_ids"].to("cuda", non_blocking=True)
            
            # forward pass
            outputs = model(
                input_ids=input_ids,
                audio_features=audio_features,
                labels=labels,
                alt_input_ids=alt_input_ids,
                alt_labels=alt_labels,
                attention_mask=attention_mask,
                insertion_indices=insertion_indices,
                text_ids=text_ids
            )

            # backward pass
            loss = outputs.loss
            loss.backward()
            
            # optimizer and scheduler steps
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            global_step += 1
            
            # calculate batch time
            batch_time = time.monotonic() - batch_start_time
            
            if global_step % cfg.training.log_every_n_steps == 0:
                current_lr = scheduler.get_last_lr()[0]
                memory_allocated = torch.cuda.memory_allocated("cuda") / 1024**2
                memory_reserved = torch.cuda.memory_reserved("cuda") / 1024**2
                #torch.cuda.empty_cache()
                logger.info(
                    f"Step {global_step} | Epoch {epoch+1} | Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f} | LR: {current_lr:.2e} | Batch Time: {batch_time:.2f}s | "
                    f"GPU Memory: Allocated {memory_allocated:.2f} MB, Reserved {memory_reserved:.2f} MB"
                )
                if cfg.wandb.enabled:
                    wandb.log({
                        "train/loss": loss.item(), "lr": current_lr, "train/batch_time": batch_time
                    })
            

            # run evaluation and loss calculation if needed
            if global_step % cfg.training.evaluate_every_n_steps == 0:
                logger.info("\nRunning evaluation at step %d", global_step)
                
                # calculate evaluation loss
                eval_loss = calculate_eval_loss(
                    model=model,
                    test_dataset=test_dataset,
                    tokenizer=model.tokenizer,
                    logger=logger,
                    n_samples=cfg.eval.n_samples,
                    batch_size=cfg.eval.batch_size
                )
                
                # run WER/CER evaluation
                wer, cer = run_evaluation(
                    model=model,
                    test_dataset=test_dataset,
                    tokenizer=model.tokenizer,
                    logger=logger,
                    n_samples=cfg.eval.n_samples,
                    config=cfg,
                    batch_size=cfg.eval.batch_size
                )
                
                logger.info(
                    f"Step {global_step} Evaluation | "
                    f"Eval Loss: {eval_loss:.4f} | "
                    f"WER: {wer:.2f} | CER: {cer:.2f}"
                )
                if cfg.wandb.enabled:
                    wandb.log({
                        "eval/loss": eval_loss,
                        "eval/wer": wer,
                        "eval/cer": cer
                    })

            # reset batch timer
            batch_start_time = time.monotonic()
        
        # calculate epoch statistics
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.monotonic() - epoch_start_time
        avg_batch_time = epoch_time / len(dataloader)
        
        # end of epoch evaluation
        logger.info("\nRunning end of epoch evaluation")
        eval_loss, wer, cer = run_evaluation_with_loss(
            model=model, 
            test_dataset=test_dataset, 
            tokenizer=model.tokenizer, 
            logger=logger, 
            n_samples=None, 
            config=cfg
        )
            
        logger.info(
            f"Epoch {epoch+1} Summary | "
            f"Average Train Loss: {avg_loss:.4f} | "
            f"Eval Loss: {eval_loss:.4f} | "
            f"WER: {wer:.2f} | CER: {cer:.2f} | "
            f"Epoch Time: {timedelta(seconds=int(epoch_time))} | "
            f"Average Batch Time: {avg_batch_time:.2f}s"
        )
        if cfg.wandb.enabled:
            wandb.log({
                "train/epoch_loss": avg_loss,
                "eval/epoch_loss": eval_loss,
                "eval/epoch_wer": wer,
                "eval/epoch_cer": cer
            })
    if cfg.wandb.enabled:
        wandb.finish(0)
        

@hydra.main(version_base="1.2", config_path="configs", config_name="config_sweep")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.cuda.set_per_process_memory_fraction(0.95, device=torch.cuda.current_device())
    # log the Hydra configuration for reproducibility
    logger.info("Hydra Config:")
    logger.info(str(cfg))
    # initialize components using config values from Hydra

    sweep_config = {
        "method": "grid",
        "metric": {"name": "eval/loss", "goal": "minimize"},
        "parameters": {
            "vocab_size": {
                "values": [512, 1024, 2048, 4096, 8192, 16384]
            },
        }
    }

    sweep_id = wandb.sweep(sweep_config, entity="borkeren", project="vocab-size-sweep")

    wandb.agent(sweep_id, function=functools.partial(train_multiple, cfg=cfg))

if __name__ == "__main__":
    main()
