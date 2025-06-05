import os, random, warnings, numpy as np
from functools import partial

import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import matplotlib.pyplot as plt
from PIL import Image

from datasets import DatasetDict
from dataset.load_librispeech import librispeech_dataset_dict
from jiwer import wer, cer

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    TrainerCallback,
)

from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from accelerate import Accelerator
from accelerate.utils import set_seed

warnings.filterwarnings("ignore")
set_seed(42)

def audio_to_rgb_spec(
    audio_np: np.ndarray,
    sr: int,
    res: int = 896,
) -> Image.Image:
    # Convert to tensor
    wav = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
    
    # Use exact Whisper parameters from the HF configuration
    mel = MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=80,
        f_min=0,
        f_max=8000,
        power=2.0,
    )(wav)
    
    # Log-scale mel spectrogram (standard Whisper preprocessing)
    log_mel = torch.log10(torch.clamp(mel, min=1e-10))
    
    # Normalize following Whisper's approach
    log_mel = torch.clamp((log_mel - log_mel.mean()) / (log_mel.std() + 1e-6), -4.0, 4.0)
    log_mel = (log_mel + 4.0) / 8.0  # Scale to [0,1]
    
    # Convert to numpy for visualization
    mel_np = log_mel.squeeze(0).cpu().numpy()
    
    # Use viridis colormap for visualization
    colored = plt.colormaps['viridis'](mel_np)[:, :, :3]
    
    # Handle segmentation for long spectrograms
    h, w = colored.shape[0:2]
    aspect = w / h
    
    if aspect > 10:
        num_segments = 3
    elif aspect > 5:
        num_segments = 2
    else:
        num_segments = 1
    
    if num_segments > 1:
        segment_width = w // num_segments
        stacked_height = h * num_segments
        stacked_rgb = np.zeros((stacked_height, segment_width, 3))
        
        for i in range(num_segments):
            start_col = i * segment_width
            end_col = min((i + 1) * segment_width, w)
            segment = colored[:, start_col:end_col, :]
            stacked_rgb[i*h:(i+1)*h, 0:(end_col-start_col), :] = segment
        
        colored = stacked_rgb
        h, w = colored.shape[0:2]
        aspect = w / h
    
    # Resize and pad
    if aspect > 1:
        new_w = res
        new_h = int(res / aspect)
        padding = (0, (res - new_h) // 2, 0, res - new_h - (res - new_h) // 2)
    else:
        new_h = res
        new_w = int(res * aspect)
        padding = ((res - new_w) // 2, 0, res - new_w - (res - new_w) // 2, 0)
    
    img = Image.fromarray((colored * 255).astype(np.uint8))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    
    padded_img = Image.new("RGB", (res, res), (0, 0, 0))
    padded_img.paste(img, (padding[0], padding[1]))
    padded_img = padded_img.rotate(180)
    
    return padded_img

# Initialize accelerator early
accelerator = Accelerator()

# Main function to contain all training logic
def main():
    global accelerator
    
    librispeech: DatasetDict = librispeech_dataset_dict
    train_ds, test_ds = librispeech["train"], librispeech["test"]

    MODEL_ID = "google/gemma-3-4b-it"
    DTYPE = torch.bfloat16

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE,
        bnb_4bit_quant_storage=DTYPE,
    )

    # Load model without device_map
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE, 
        quantization_config=bnb_cfg,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    lora_cfg = LoraConfig(
        r=32, lora_alpha=16, lora_dropout=0.05, use_rslora=True,
        bias="none", task_type="CAUSAL_LM",
        target_modules="all-linear",
        modules_to_save=["embed_tokens", "lm_head"],
    )

    SYS_MSG = "You are an expert speech-to-text assistant."
    USR_TMPL = (
        "Please transcribe the speech contained in the attached spectrogram image.\n"
        "Return only the transcript."
    )

    def collate_fn(batch, proc: AutoProcessor):
        texts, images = [], []
        for sample in batch:
            spec = audio_to_rgb_spec(sample["audio"]["array"],
                                    sample["audio"]["sampling_rate"])
            chat = [
                {"role": "system", "content": [{"type": "text", "text": SYS_MSG}]},
                {"role": "user",   "content": [
                    {"type": "image", "image": spec},
                    {"type": "text",  "text": USR_TMPL},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": sample["text"].lower()}]},
            ]
            prompt = proc.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
            texts.append(prompt.strip())
            images.append([spec])   # list-of-list for multimodal batch
        batch_enc = proc(text=texts, images=images, padding=True, return_tensors="pt")
        labels = batch_enc.input_ids.clone()
        pad_id = proc.tokenizer.pad_token_id
        boi_id = proc.tokenizer.convert_tokens_to_ids(proc.tokenizer.special_tokens_map["boi_token"])
        labels[labels == pad_id] = -100
        labels[labels == boi_id] = -100
        labels[labels == 262144] = -100
        batch_enc["labels"] = labels
        return batch_enc

    data_collator = partial(collate_fn, proc=processor)

    @torch.inference_mode()
    def generate_transcript(sample, mdl=model, proc=processor):
        spec = audio_to_rgb_spec(sample["audio"]["array"],
                                sample["audio"]["sampling_rate"])
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYS_MSG}]},
            {"role": "user", "content": [
                {"type": "image", "image": spec},
                {"type": "text", "text": USR_TMPL},
            ]},
        ]
        prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[prompt], images=[spec], return_tensors="pt")
        # Move to the appropriate device as needed
        inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
        gen_ids = mdl.generate(
            **inputs, max_new_tokens=256, top_p=1.0, temperature=0.8,
            eos_token_id=[proc.tokenizer.eos_token_id,
                        proc.tokenizer.convert_tokens_to_ids("<end_of_turn>")],
        )
        return proc.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:],
                                skip_special_tokens=True)[0].strip()

    def compute_wer(model_, subset=None):
        refs, hyps = [], []
        iterable = subset if subset is not None else test_ds
        for sample in iterable:
            refs.append(sample["text"].lower().strip())
            hyps.append(generate_transcript(sample, mdl=model_).lower().strip())
        return wer(refs, hyps), cer(refs, hyps)

    class SaveBestWERCallback(TrainerCallback):
        def __init__(self, out_dir="gemma-s2t-librispeech-best"):
            self.best_wer = float("inf")
            self.out_dir  = out_dir
            os.makedirs(out_dir, exist_ok=True)

        def on_epoch_end(self, args, state, control, **kwargs):
            mdl = kwargs["model"]
            print("\nComputing WER / CER on test split …")
            cur_wer, cur_cer = compute_wer(mdl)
            print(f"Epoch {state.epoch:.0f}  WER={cur_wer:.4%}  CER={cur_cer:.4%}")
            if cur_wer < self.best_wer:
                self.best_wer = cur_wer
                print(f"★ New best WER - saving adapter to {self.out_dir}")
                mdl.save_pretrained(self.out_dir)
                processor.save_pretrained(self.out_dir)

    # Configure for multi-GPU training
    cfg = SFTConfig(
        output_dir="gemma-s2t-librispeech",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=2e-4,
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        bf16=True,
        lr_scheduler_type="constant",
        logging_steps=1,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=8,
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        push_to_hub=False,
        
        # Important settings for FSDP/DDP
        ddp_find_unused_parameters=False,
        tf32=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        data_collator=data_collator,
        processing_class=processor,
        peft_config=lora_cfg,
        callbacks=[SaveBestWERCallback()],
    )

    trainer.train()
    print("\nTraining done. Best adapter is in gemma-s2t-librispeech-best/")

if __name__ == "__main__":
    main()