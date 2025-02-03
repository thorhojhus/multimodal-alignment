import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_linear_schedule_with_warmup
from models.audio_llm import AudioLLMModel, AudioLLMConfig
from dataset.librispeech import LibriSpeechChatDataset, collate_fn
import time
from datetime import timedelta

def count_trainable_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_parameter_count(model):
    """Print trainable and total parameter counts."""
    trainable_params = count_trainable_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters:")
    print(f"Trainable parameters: \t{trainable_params:,}")
    print(f"Total parameters: \t{total_params:,}")
    print(f"Percentage trainable: \t{100 * trainable_params / total_params:.2f}%\n")

def train(model, dataset, num_epochs=3, batch_size=4):
    # print parameter counts before training
    print_parameter_count(model)
    
    # create optimizer only for trainable parameters
    trainable_params = [
        {"params": model.lm.get_input_embeddings().parameters()},
        {"params": model.projector.parameters()}
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, pin_memory=True)
    
    # calculate total number of training steps
    total_steps = len(dataloader) * num_epochs
    
    # create the scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=50,
        num_training_steps=total_steps
    )
    
    for epoch in range(num_epochs):
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
            
            # calculate batch time
            batch_time = time.time() - batch_start_time
            
            if batch_idx % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch+1} | "
                    f"Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Batch Time: {batch_time:.2f}s"
                )
            
            # reset batch timer
            batch_start_time = time.time()
        
        # calculate epoch statistics
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start_time
        avg_batch_time = epoch_time / len(dataloader)
        
        print(
            f"Epoch {epoch+1} Summary | "
            f"Average Loss: {avg_loss:.4f} | "
            f"Epoch Time: {timedelta(seconds=int(epoch_time))} | "
            f"Average Batch Time: {avg_batch_time:.2f}s"
        )

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    # initialize components
    config = AudioLLMConfig()
    model = AudioLLMModel(config)
    
    # create dataset and processor
    processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    dataset = LibriSpeechChatDataset(
        split="train.clean.360",
        tokenizer=model.tokenizer,
        processor=processor
    )
    
    # freeze non-trainable components
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
    for param in model.lm.parameters():
        param.requires_grad = False
    
    # start training
    train(model, dataset, num_epochs=1, batch_size=16)