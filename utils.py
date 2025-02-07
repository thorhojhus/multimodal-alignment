import os
import logging
import torch
from torch.utils.data import DataLoader
from dataset.audio_dataset import collate_fn


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
    """Save the checkpoints for specified models (LIST), optimizer and scheduler.
    
    The checkpoint savefile is saved based on the step parameter.
    """
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
    
    # Save latest checkpoint separately for easy loading
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)

def load_checkpoint(
    models: list[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_path: str,
    logger: logging.Logger
) -> tuple:
    "Load the checkpoint for specified models (LIST), optimizer and scheduler."
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    for model in models:
        model.load_state_dict(checkpoint[model._get_name()])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    
    return checkpoint['global_step'], checkpoint['loss']

def run_evaluation(model: torch.nn.Module, test_dataset: torch.utils.data.Dataset, tokenizer, logger: logging.Logger) -> None:
    """
    Run generation sequentially on samples from test dataset and compare with ground truth.
    Only includes user prompt during generation.
    """
    model.eval()
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Changed to 1 for sequential processing
        collate_fn=collate_fn,
        shuffle=True
    )
    
    # get two samples (keeping same number of examples as original)
    samples = []
    for i, batch in enumerate(test_loader):
        if i >= 2:  # Process only 2 samples
            break
        samples.append(batch)
    
    for i, batch in enumerate(samples):
        input_ids = batch["input_ids"].to("cuda")
        audio_features = batch["audio_features"].to("cuda", torch.bfloat16)
        
        # extract only the user part of the conversation for generation
        # find positions of BOS tokens
        bos_positions = (input_ids[0] == tokenizer.bos_token_id).nonzero(as_tuple=True)[0]
        # get the last BOS token position (start of assistant's response)
        last_bos = bos_positions[-1].item()
        # get user prompt (everything before assistant's response)
        user_prompt = input_ids[0, :last_bos+4]
        
        # extract ground truth (assistant's response)
        # find positions of EOS tokens
        eos_positions = (input_ids[0] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        # find the first EOS after last BOS
        eos_after_bos = eos_positions[eos_positions > last_bos]
        response_end = eos_after_bos[0].item() if len(eos_after_bos) > 0 else len(input_ids[0])
        # extract and decode the assistant's response
        response_ids = input_ids[0, last_bos+4:response_end]
        ground_truth = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # generate completion
        with torch.inference_mode():
            output = model.generate(
                input_ids=user_prompt.unsqueeze(0),
                audio_features=audio_features,
                max_new_tokens=100,
                do_sample=False,
            )
        
        # decode and log results
        generated_text = tokenizer.decode(output[0][len(user_prompt):], skip_special_tokens=False)
        
        logger.info("\nEvaluation Sample %d:", i + 1)
        logger.info("User prompt: %s", tokenizer.decode(user_prompt, skip_special_tokens=False))
        logger.info("Ground truth: %s", ground_truth)
        logger.info("Generated: %s", generated_text)
        logger.info("-" * 50)
    
    model.train()