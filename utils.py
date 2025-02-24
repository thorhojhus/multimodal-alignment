import functools
import logging
import os
import torch
from torch.utils.data import DataLoader
from dataset.audio_dataset import collate_fn
import jiwer
import wandb
from typing import Tuple, List

def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameter_count(model: torch.nn.Module, logger: logging.Logger) -> None:
    trainable_params = count_trainable_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("\nModel Parameters:")
    logger.info(f"Trainable parameters: \t{trainable_params:,}")
    logger.info(f"Total parameters: \t{total_params:,}")
    logger.info(f"Percentage trainable: \t{100 * trainable_params / total_params:.2f}%\n")

def save_checkpoint(
    models: list[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    global_step: int,
    loss: float,
    save_dir: str,
    logger: logging.Logger
) -> None:
    checkpoint = {
        'global_step': global_step,
        'loss': loss,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict()
    }
    checkpoint |= {model._get_name(): model.state_dict() for model in models}
    checkpoint_path = os.path.join(save_dir, f'checkpoint_step_{global_step}.pt')
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)

def load_checkpoint(
    models: list[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_path: str,
    logger: logging.Logger
) -> tuple:
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    for model in models:
        model.load_state_dict(checkpoint[model._get_name()])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    return checkpoint['global_step'], checkpoint['loss']

def get_user_prompts_batch(ids_batch, model_name: str, tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    token_id = tokenizer.bos_token_id
    if model_name == "gemma":
        token_id = 106  # <|start_of_turn|>

    prompts = []
    masks = []

    for ids in ids_batch:
        bos_positions = (ids == token_id).nonzero(as_tuple=True)[0]
        last_bos = bos_positions[-1].item()
        prompt = ids[:last_bos + 3]
        prompts.append(prompt)
        masks.append(torch.ones_like(prompt))

    max_len = max(len(p) for p in prompts)
    padded_prompts = []
    padded_masks = []

    for prompt, mask in zip(prompts, masks):
        pad_len = max_len - len(prompt)
        padded_prompts.append(
            torch.nn.functional.pad(prompt, (0, pad_len), value=tokenizer.pad_token_id)
        )
        padded_masks.append(torch.nn.functional.pad(mask, (0, pad_len), value=0))

    return torch.stack(padded_prompts), torch.stack(padded_masks)

def get_ground_truths_batch(ids_batch, model_name: str, tokenizer) -> List[str]:
    begin_of_turn = tokenizer.bos_token_id
    end_of_turn = tokenizer.eos_token_id
    if model_name == "gemma":
        begin_of_turn = 106  # <|start_of_turn|>
        end_of_turn = 107  # <|end_of_turn|>

    ground_truths = []
    for ids in ids_batch:
        bos_positions = (ids == begin_of_turn).nonzero(as_tuple=True)[0]
        eos_positions = (ids == end_of_turn).nonzero(as_tuple=True)[0]
        last_bos = bos_positions[-1].item()
        eos_after_bos = eos_positions[eos_positions > last_bos]
        response_end = eos_after_bos[0].item() if len(eos_after_bos) > 0 else len(ids)
        response_ids = ids[last_bos + 3:response_end]
        ground_truths.append(tokenizer.decode(response_ids, skip_special_tokens=True))

    return ground_truths

def generate_batch(
    model: torch.nn.Module,
    batch: dict,
    model_name: str,
    tokenizer,
    max_new_tokens: int = 100
) -> Tuple[List[str], List[str]]:
    user_prompts, user_prompt_masks = get_user_prompts_batch(batch["input_ids"], model_name, tokenizer)
    ground_truths = get_ground_truths_batch(batch["input_ids"], model_name, tokenizer)

    user_prompts = user_prompts.to("cuda")
    user_prompt_masks = user_prompt_masks.to("cuda")
    audio_features = batch["audio_features"].to("cuda", torch.bfloat16)
    insertion_indices = [indices.to("cuda") for indices in batch["insertion_indices"]]

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=user_prompts,
            audio_features=audio_features,
            insertion_indices=insertion_indices,
            attention_mask=user_prompt_masks,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    generated_texts = [
        tokenizer.decode(output.tolist(), skip_special_tokens=True)
        for output in outputs
    ]

    return generated_texts, ground_truths

def calculate_metrics(hypotheses: List[str], references: List[str]) -> Tuple[float, float]:
    wer = jiwer.wer(references, hypotheses)
    cer = jiwer.cer(references, hypotheses)
    return wer, cer

def run_evaluation(
    model: torch.nn.Module,
    test_dataset: torch.utils.data.Dataset,
    tokenizer,
    logger: logging.Logger,
    n_samples: int,
    config,
    batch_size: int = 64
) -> Tuple[float, float]:
    """
    Run batched evaluation on test dataset and calculate WER/CER metrics.
    
    Args:
        model: The AudioLLM model to evaluate
        test_dataset: Dataset to evaluate on
        tokenizer: Model tokenizer
        logger: Logger instance
        n_samples: Number of samples to evaluate (if > dataset size, uses full dataset)
        config: Model configuration
        batch_size: Batch size for evaluation
    
    Returns:
        Tuple of (WER, CER) metrics
    """
    model.eval()
    torch.cuda.empty_cache()
    model_name = config.model.lm_model_name.split("/")[-1].split("-")[0].lower()
    
    # adjust n_samples if it exceeds dataset size
    if n_samples is None:
        n_samples = len(test_dataset)
    else:
        n_samples = min(n_samples, len(test_dataset))
    logger.info(f"Evaluating on {n_samples} samples with batch size {batch_size}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=functools.partial(collate_fn, tokenizer=tokenizer),
        shuffle=True
    )
    
    all_generated = []
    all_ground_truths = []
    samples_processed = 0

    logger.info("Generating predictions...")
    for batch in test_loader:
        if samples_processed >= n_samples:
            break
            
        generated_texts, ground_truths = generate_batch(
            model, batch, model_name, tokenizer
        )
        
        # handle partial last batch if needed
        remaining_samples = n_samples - samples_processed
        if remaining_samples < len(generated_texts):
            generated_texts = generated_texts[:remaining_samples]
            ground_truths = ground_truths[:remaining_samples]
            
        all_generated.extend(generated_texts)
        all_ground_truths.extend(ground_truths)
        samples_processed += len(generated_texts)

    logger.info("\nCalculating metrics...")
    wer, cer = calculate_metrics(all_generated, all_ground_truths)
    i = 0
    for gen, gt in zip(all_generated, all_ground_truths):
        print(f"Generated: {gen.strip()}\nGround Truth: {gt.strip()}\n")
        i += 1
        if i == 10:
            break
    
    logger.info(f"\nResults over {len(all_generated)} samples:")
    logger.info(f"Word Error Rate (WER): {wer:.4f}")
    logger.info(f"Character Error Rate (CER): {cer:.4f}")
    
    if config.wandb.enabled:
            eval_table = wandb.Table(columns=["Generated", "Ground Truth"])
            for gen, gt in zip(all_generated, all_ground_truths):
                eval_table.add_data(gen.strip(), gt.strip())
            wandb.log({"evaluation_completions": eval_table})
    torch.cuda.empty_cache()
    model.train()
    return wer, cer


def calculate_eval_loss(
    model: torch.nn.Module,
    test_dataset: torch.utils.data.Dataset,
    tokenizer,
    logger: logging.Logger,
    n_samples: int = None,
    batch_size: int = 32
) -> float:
    """
    Calculate average evaluation loss on test dataset.
    
    Args:
        model: The AudioLLM model to evaluate
        test_dataset: Dataset to evaluate on
        tokenizer: Model tokenizer
        logger: Logger instance
        n_samples: Number of samples to evaluate (if None, uses full dataset)
        batch_size: Batch size for evaluation
    
    Returns:
        Average loss over the evaluated samples
    """
    model.eval()
    torch.cuda.empty_cache()
    
    # adjust n_samples if it exceeds dataset size
    if n_samples is None:
        n_samples = len(test_dataset)
    else:
        n_samples = min(n_samples, len(test_dataset))
    logger.info(f"Calculating evaluation loss on {n_samples} samples with batch size {batch_size}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=functools.partial(collate_fn, tokenizer=tokenizer),
        shuffle=False 
    )
    
    total_loss = 0.0
    samples_processed = 0
    num_batches = 0
    
    with torch.inference_mode():
        for batch in test_loader:
            if samples_processed >= n_samples:
                break
                
            # move data to GPU
            input_ids = batch["input_ids"].to("cuda", non_blocking=True)
            if isinstance(batch["audio_features"], dict):
                audio_features = {
                    'input_features': batch["audio_features"]["input_features"].to("cuda", torch.bfloat16, non_blocking=True),
                    'attention_mask': batch["audio_features"]["attention_mask"].to("cuda", non_blocking=True)
                }
            else:
                audio_features = batch["audio_features"].to("cuda", torch.bfloat16, non_blocking=True)
            labels = batch["labels"].to("cuda", non_blocking=True)
            alt_input_ids = batch["alt_input_ids"].to("cuda", non_blocking=True)
            alt_labels = batch["alt_labels"].to("cuda", non_blocking=True)
            attention_mask = batch["attention_mask"].to("cuda", non_blocking=True)
            insertion_indices = batch["insertion_indices"]
            
            # forward pass
            outputs = model(
                input_ids=input_ids,
                audio_features=audio_features,
                labels=labels,
                alt_input_ids=alt_input_ids,
                alt_labels=alt_labels,
                attention_mask=attention_mask,
                insertion_indices=insertion_indices
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
            samples_processed += input_ids.size(0)
            
            if samples_processed > n_samples:
                # adjust for partial last batch
                excess = samples_processed - n_samples
                batch_loss_contribution = loss.item() * (input_ids.size(0) - excess) / input_ids.size(0)
                total_loss = total_loss - loss.item() + batch_loss_contribution
                samples_processed = n_samples
                num_batches -= 1  # don't count partial batch fully
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info(f"Average Evaluation Loss: {avg_loss:.4f} over {samples_processed} samples")
    torch.cuda.empty_cache()
    model.train()
    return avg_loss