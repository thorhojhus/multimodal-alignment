import os
import logging
import time
from datetime import timedelta
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_linear_schedule_with_warmup
from models.audio_llm import AudioLLMModel, AudioLLMConfig
from dataset.librispeech import LibriSpeechChatDataset, collate_fn
import hydra
from omegaconf import DictConfig
from liger_kernel.transformers import _apply_liger_kernel_to_instance

# set up a module-level logger
logger = logging.getLogger(__name__)

def count_trainable_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameter_count(model):
    """Print trainable and total parameter counts."""
    trainable_params = count_trainable_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("\nModel Parameters:")
    logger.info(f"Trainable parameters: \t{trainable_params:,}")
    logger.info(f"Total parameters: \t{total_params:,}")
    logger.info(f"Percentage trainable: \t{100 * trainable_params / total_params:.2f}%\n")

def save_checkpoint(model, optimizer, scheduler, global_step, loss, save_dir):
    """
    Save training checkpoint including model projector weights, optimizer state, 
    scheduler state, and training metadata.
    """
    checkpoint = {
        'global_step': global_step,
        'loss': loss,
        'projector_state': model.projector.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict()
    }
    
    checkpoint_path = os.path.join(save_dir, f'checkpoint_step_{global_step}.pt')
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save latest checkpoint separately for easy loading
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """
    Load training checkpoint.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    model.projector.load_state_dict(checkpoint['projector_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    return checkpoint['global_step'], checkpoint['loss']

def run_evaluation(model, test_dataset, tokenizer):
    """
    Run generation on a batch from test dataset and compare with ground truth.
    Only includes user prompt during generation.
    """
    model.eval()
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        collate_fn=collate_fn,
        shuffle=True
    )
    
    # get one batch
    batch = next(iter(test_loader))
    input_ids = batch["input_ids"].to("cuda")
    audio_features = batch["audio_features"].to("cuda", torch.bfloat16)
    
    # extract only the user part of the conversation for generation
    def get_user_prompt(ids):
        # find positions of BOS tokens
        bos_positions = (ids == tokenizer.bos_token_id).nonzero(as_tuple=True)[0]
        # get the last BOS token position (start of assistant's response)
        last_bos = bos_positions[-1].item()
        # return everything before the assistant's response
        return ids[:last_bos]
    
    # extract ground truth (assistant's response)
    def get_ground_truth(ids):
        # find positions of BOS and EOS tokens
        bos_positions = (ids == tokenizer.bos_token_id).nonzero(as_tuple=True)[0]
        eos_positions = (ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        
        # get the last BOS token (start of assistant response)
        last_bos = bos_positions[-1].item()
        
        # find the first EOS after last BOS
        eos_after_bos = eos_positions[eos_positions > last_bos]
        response_end = eos_after_bos[0].item() if len(eos_after_bos) > 0 else len(ids)
        
        # extract and decode the assistant's response
        response_ids = ids[last_bos+1:response_end]
        return tokenizer.decode(response_ids, skip_special_tokens=True)
    
    # prepare input for generation
    user_prompts = []
    ground_truths = []
    for i in range(len(input_ids)):
        user_prompts.append(get_user_prompt(input_ids[i]))
        ground_truths.append(get_ground_truth(input_ids[i]))
    
    # pad user prompts to same length
    max_prompt_len = max(len(p) for p in user_prompts)
    padded_prompts = []
    attention_masks = []
    for prompt in user_prompts:
        pad_len = max_prompt_len - len(prompt)
        padded_prompt = torch.nn.functional.pad(prompt, (0, pad_len), value=tokenizer.pad_token_id)
        attention_mask = torch.cat([torch.ones(len(prompt)), torch.zeros(pad_len)]).to(prompt.device)
        padded_prompts.append(padded_prompt)
        attention_masks.append(attention_mask)
    
    input_ids_for_gen = torch.stack(padded_prompts)
    attention_mask_for_gen = torch.stack(attention_masks)
    
    # generate completions
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids_for_gen,
            audio_features=audio_features,
            attention_mask=attention_mask_for_gen,
            max_new_tokens=100,
            do_sample=False,
        )
    
    # log results
    for i in range(len(outputs)):
        generated_text = tokenizer.decode(outputs[i][len(user_prompts[i]):], skip_special_tokens=True)
        
        logger.info("\nEvaluation Sample %d:", i + 1)
        logger.info("Ground truth: %s", ground_truths[i])
        logger.info("Generated  : %s", generated_text)
        logger.info("-" * 50)
    
    model.train()

def train(model, dataset, cfg):
    # create checkpoint directory
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    test_dataset = LibriSpeechChatDataset(
        split="test.clean",
        tokenizer=model.tokenizer,
        processor=AutoProcessor.from_pretrained(cfg.model.whisper_model_name)
    )
    
    print_parameter_count(model)
        
    trainable_params = [
        {"params": list(model.projector.parameters())},
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.training.lr)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.training.batch_size, 
        collate_fn=collate_fn, 
        shuffle=True, 
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
    if cfg.training.resume_from_checkpoint:
        global_step, _ = load_checkpoint(
            model, optimizer, scheduler, 
            cfg.training.resume_checkpoint_path
        )
        logger.info(f"Resuming training from step {global_step}")
    
    for epoch in range(cfg.training.num_epochs):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()
        batch_start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            # move data to GPU
            input_ids = batch["input_ids"].to("cuda", non_blocking=True)
            audio_features = batch["audio_features"].to("cuda", torch.bfloat16, non_blocking=True)
            labels = batch["labels"].to("cuda", non_blocking=True)
            
            # forward pass
            outputs = model(
                input_ids=input_ids,
                audio_features=audio_features,
                labels=labels
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
            batch_time = time.time() - batch_start_time
            
            if global_step % cfg.training.log_every_n_steps == 0:
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Step {global_step} | "
                    f"Epoch {epoch+1} | "
                    f"Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Batch Time: {batch_time:.2f}s"
                )
            
            # Save checkpoint if needed
            if global_step % cfg.training.save_every_n_steps == 0:
                save_checkpoint(
                    model, optimizer, scheduler, 
                    global_step, loss.item(), 
                    cfg.training.checkpoint_dir
                )
                
            # Run evaluation if needed
            if global_step % cfg.training.evaluate_every_n_steps == 0:
                logger.info("\nRunning evaluation at step %d", global_step)
                with torch.inference_mode():
                    run_evaluation(model, test_dataset, model.tokenizer)
            
            # reset batch timer
            batch_start_time = time.time()
        
        # calculate epoch statistics
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = epoch_time / len(dataloader)
        
        logger.info(
            f"Epoch {epoch+1} Summary | "
            f"Average Loss: {avg_loss:.4f} | "
            f"Epoch Time: {timedelta(seconds=int(epoch_time))} | "
            f"Average Batch Time: {avg_batch_time:.2f}s"
        )

@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision('high')
    # log the Hydra configuration for reproducibility
    logger.info("Hydra Config:\n" + str(cfg))
    
    # initialize components using config values from Hydra
    config = AudioLLMConfig(
        downsample_factor=cfg.model.downsample_factor,
        lm_model_name=cfg.model.lm_model_name,
        whisper_model_name=cfg.model.whisper_model_name
    )
    model = AudioLLMModel(config)
    
    #model.apply_liger_kernel()
    
    # create dataset and processor
    processor = AutoProcessor.from_pretrained(cfg.model.whisper_model_name)
    dataset = LibriSpeechChatDataset(
        split=cfg.dataset.split,
        tokenizer=model.tokenizer,
        processor=processor
    )
    
    # freeze non-trainable components
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
    for param in model.lm.parameters():
        param.requires_grad = False
    
    train(model, dataset, cfg)

if __name__ == "__main__":
    main()