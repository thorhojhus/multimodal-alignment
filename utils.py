import os
import logging
import torch
from torch.utils.data import DataLoader
from dataset.librispeech import collate_fn


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameter_count(model: torch.nn.Module, logger: logging.Logger) -> None:
    trainable_params = count_trainable_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("\nModel Parameters:")
    logger.info(f"Trainable parameters: \t{trainable_params:,}")
    logger.info(f"Total parameters: \t{total_params:,}")
    logger.info(f"Percentage trainable: \t{100 * trainable_params / total_params:.2f}%\n")

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, global_step: int, loss: float, save_dir: str, logger: logging.Logger) -> None:
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

def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler, checkpoint_path: str, logger: logging.Logger) -> tuple:
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    model.projector.load_state_dict(checkpoint['projector_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    return checkpoint['global_step'], checkpoint['loss']

def run_evaluation(model: torch.nn.Module, test_dataset: torch.utils.data.Dataset, tokenizer, logger: logging.Logger) -> None:
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
        return ids[:last_bos+4]
    
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
        logger.info("Generated: %s", generated_text)
        logger.info("-" * 50)
    
    model.train()