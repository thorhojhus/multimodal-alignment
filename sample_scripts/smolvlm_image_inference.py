import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, TextStreamer
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

image = load_image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Ptychochromis_insolitus_-_Wilhelma_01.jpg/2560px-Ptychochromis_insolitus_-_Wilhelma_01.jpg")

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")

model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda",
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Can you describe this image?"}
        ]
    },
]

prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
inputs = inputs.to(DEVICE)

streamer = TextStreamer(processor.tokenizer, skip_special_tokens=True)

generated_ids = model.generate(**inputs, max_new_tokens=500, streamer=streamer)

generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)