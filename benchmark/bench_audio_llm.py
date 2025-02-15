from models.audio_llm import AudioLLMModel
from types import SimpleNamespace
import torch
from dataset.audio_dataset import AudioChatDataset, collate_fn
from torch.utils.data import DataLoader
import functools
from transformers import AutoProcessor
import jiwer
from tqdm import tqdm

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.set_per_process_memory_fraction(0.9, device=torch.cuda.current_device())
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Config
config = SimpleNamespace(
    model=SimpleNamespace(
        lm_model_name="google/gemma-2-9b-it",
        encoder_model_name="syvai/hviske-v2",
        downsample_factor=10,
        load_in_4bit=True,
    ),
    training=SimpleNamespace(kl_loss=True),
)


def get_user_prompts_batch(ids_batch, model_name, tokenizer):
    token_id = tokenizer.bos_token_id
    if model_name == "gemma":
        token_id = 106  # <|start_of_turn|>

    prompts = []
    masks = []

    for ids in ids_batch:
        bos_positions = (ids == token_id).nonzero(as_tuple=True)[0]
        last_bos = bos_positions[-1].item()
        prompt = ids[: last_bos + 3]
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


def get_ground_truths_batch(ids_batch, model_name, tokenizer):
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
        response_ids = ids[last_bos + 3 : response_end]
        ground_truths.append(tokenizer.decode(response_ids, skip_special_tokens=True))

    return ground_truths


def generate_batch(model, batch, model_name, tokenizer, max_new_tokens=100):
    user_prompts, user_prompt_masks = get_user_prompts_batch(
        batch["input_ids"], model_name, tokenizer
    )
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


def calculate_metrics(hypotheses, references):
    wer = jiwer.wer(
        references,
        hypotheses,
    )
    cer = jiwer.cer(
        references,
        hypotheses,
    )

    return wer, cer


# Initialize model
model = AudioLLMModel(config)
checkpoint = torch.load(
    "/zhome/5c/8/186818/multimodal-alignment/outputs/2025-02-15/12-42-21/checkpoints/checkpoint_latest.pt"
)
model.projector.load_state_dict(checkpoint["Projector"])
model.eval()

# Initialize tokenizer and processor
tokenizer = model.tokenizer
processor = AutoProcessor.from_pretrained("syvai/hviske-v2")

# Create test dataset and dataloader
test_dataset = AudioChatDataset(
    lm_model="google/gemma-2-9b-it",
    dataset="coral",
    tokenizer=tokenizer,
    processor=processor,
    split="test",
)

batch_size = 64
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=functools.partial(collate_fn, tokenizer=tokenizer),
    shuffle=False,
)

model_name = config.model.lm_model_name.split("/")[-1].split("-")[0].lower()

# Lists to store all generated texts and ground truths
all_generated = []
all_ground_truths = []

# Process all batches
print("Generating predictions...")
for batch in tqdm(test_loader):
    generated_texts, ground_truths = generate_batch(
        model, batch, model_name, tokenizer
    )
    all_generated.extend(generated_texts)
    all_ground_truths.extend(ground_truths)

# Calculate metrics
print("\nCalculating WER and CER...")
wer, cer = calculate_metrics(all_generated, all_ground_truths)

# Print results
print(f"\nResults over {len(all_generated)} samples:")
print(f"Word Error Rate (WER): {wer:.4f}")
print(f"Character Error Rate (CER): {cer:.4f}")

# Save some example predictions
print("\nSaving example predictions...")
num_examples = len(all_generated)
with open("prediction_examples.txt", "w") as f:
    for i in range(num_examples):
        f.write(f"\nExample {i + 1}\n")
        f.write(f"Generated: {all_generated[i]}\n")
        f.write(f"Ground truth: {all_ground_truths[i]}\n")
        f.write("-" * 80 + "\n")