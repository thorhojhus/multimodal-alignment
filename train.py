import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from models.audio_llm import AudioLLMModel, AudioLLMConfig
from dataset.librispeech import LibriSpeechChatDataset, collate_fn

def train(model, dataset, num_epochs=3, batch_size=4):
    # create optimizer only for trainable parameters
    trainable_params = [
        {"params": model.lm.get_input_embeddings().parameters()},
        {"params": model.projector.parameters()}
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=5e-5)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True, pin_memory=True)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
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
            
            # optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    # initialize components
    config = AudioLLMConfig()
    model = AudioLLMModel(config)
    
    # create dataset and processor
    processor = AutoProcessor.from_pretrained("openai/whisper-base.en")
    dataset = LibriSpeechChatDataset(
        split="train.clean.100",
        tokenizer=model.tokenizer,
        processor=processor
    )
    torch.set_float32_matmul_precision('high')
    #model.compile() # cannot be used with liger_kernel, should be config in the future
    
    # freeze non-trainable components
    for param in model.audio_encoder.parameters():
        param.requires_grad = False
    for param in model.lm.parameters():
        param.requires_grad = False
    model.lm.get_input_embeddings().weight.requires_grad = True
    
    # start training
    train(model, dataset, num_epochs=3, batch_size=16)