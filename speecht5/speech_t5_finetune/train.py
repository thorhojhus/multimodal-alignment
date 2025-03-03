# modified from https://github.com/JackismyShephard/hugging-face-audio-course/blob/main/notebooks/training/finetuned-nst-da-training.ipynb
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Union

import torch
from datasets import load_from_disk
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    SpeechT5ForTextToSpeech,
    SpeechT5Processor,
)

checkpoint = "microsoft/speecht5_tts"
dataset_path = "/home/thorh/datasets/nst-da"
output_dir = "/home/thorh/checkpoints/speecht5_finetuned_nst"

processor = SpeechT5Processor.from_pretrained(checkpoint)

model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
# disable cache during training since it's incompatible with gradient checkpointing
model.config.use_cache = False
# set language and task for generation and re-enable cache
model.generate = partial(model.generate, use_cache=True)


train_dataset = load_from_disk(os.path.join(dataset_path, "train"))
test_dataset = load_from_disk(os.path.join(dataset_path, "test"))
dataset = {"train": train_dataset, "test": test_dataset}


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = processor.pad(
            input_ids=input_ids, labels=label_features, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(feature["input_values"]) for feature in label_features]
            )
            target_lengths = target_lengths.new(
                [
                    length - length % model.config.reduction_factor
                    for length in target_lengths
                ]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch
    
data_collator = TTSDataCollatorWithPadding(processor=processor)


training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=20,
    gradient_checkpointing=False,
    bf16=True,
    tf32=True,
    logging_steps=25,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    per_device_eval_batch_size=16,
    report_to=["wandb"],
    run_name="speecht5-nst",
    load_best_model_at_end=True,
    label_names=["labels"],
    push_to_hub=False,
    dataloader_num_workers=24,
    dataloader_pin_memory=True,
)


trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor,
)

trainer.train()