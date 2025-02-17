import functools
import logging
import os

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

def run_evaluation(model: torch.nn.Module, test_dataset: torch.utils.data.Dataset, tokenizer, logger: logging.Logger, n_samples: int, config) -> None:
    """
    Run generation on a batch from test dataset and compare with ground truth.
    Only includes user prompt during generation.
    """
    model.eval()

    test_loader = DataLoader(
        test_dataset,
        batch_size=n_samples,
        collate_fn=functools.partial(collate_fn, tokenizer=tokenizer),
        shuffle=True
    )
    model_name = config.model.lm_model_name.split("/")[-1].split("-")[0].lower()
    # get one batch
    batch = next(iter(test_loader))

    # extract only the user part of the conversation for generation
    def get_user_prompt(ids, model_name):
        token_id = tokenizer.bos_token_id 
        if model_name == "gemma":
            token_id = 106 # <|start_of_turn|>
        # find positions of BOS tokens
        bos_positions = (ids == token_id).nonzero(as_tuple=True)[0]
        # get the last BOS token position (start of assistant's response)
        last_bos = bos_positions[-1].item()
        # return everything before the assistant's response
        return ids[:last_bos+3]
    
    # extract ground truth (assistant's response)
    def get_ground_truth(ids, model_name):
        begin_of_turn = tokenizer.bos_token_id
        end_of_turn = tokenizer.eos_token_id 
        if model_name == "gemma":
            begin_of_turn = 106 # <|start_of_turn|>
            end_of_turn = 107 # <|end_of_turn|>

        # find positions of BOS and EOS tokens
        bos_positions = (ids == begin_of_turn).nonzero(as_tuple=True)[0]
        eos_positions = (ids == end_of_turn).nonzero(as_tuple=True)[0]
        
        # get the last BOS token (start of assistant response)
        last_bos = bos_positions[-1].item()
        
        # find the first EOS after last BOS
        eos_after_bos = eos_positions[eos_positions > last_bos]
        response_end = eos_after_bos[0].item() if len(eos_after_bos) > 0 else len(ids)
        
        # extract and decode the assistant's response
        response_ids = ids[last_bos:response_end]
        return tokenizer.decode(response_ids, skip_special_tokens=True)

    
    for i in range(n_samples):
        user_prompt = get_user_prompt(batch["input_ids"][i], model_name)
        ground_truth = get_ground_truth(batch["input_ids"][i], model_name)
        user_prompt_mask = torch.ones_like(user_prompt)
        
        if isinstance(batch["audio_features"], dict):
            audio_features = {
                'input_features': batch["audio_features"]["input_features"][i].to("cuda", torch.bfloat16).unsqueeze(0),
                'attention_mask': batch["audio_features"]["attention_mask"][i].to("cuda").unsqueeze(0)
            }
        else:
            audio_features = batch["audio_features"][i].to("cuda", torch.bfloat16).unsqueeze(0)

        # generate completions
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=user_prompt.unsqueeze(0).to("cuda"),
                audio_features=audio_features,
                insertion_indices=batch["insertion_indices"][i].to("cuda").unsqueeze(0),
                attention_mask=user_prompt_mask.unsqueeze(0).to("cuda"),
                max_new_tokens=100,
                do_sample=False,
            )
    
        generated_text = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=False)
        
        logger.info("\nEvaluation Sample %d:", i + 1)
        logger.info("User prompt: %s", tokenizer.decode(user_prompt, skip_special_tokens=False))
        logger.info("Ground truth: %s", ground_truth)
        logger.info("Generated: %s", generated_text)
        logger.info("-" * 50)
    
    model.train()