# Modified version of example script provided in https://huggingface.co/ibm-granite/granite-vision-3.1-2b-preview

from transformers import AutoProcessor, AutoModelForVision2Seq
from huggingface_hub import hf_hub_download
from PIL import Image
import torch
import os

prompt = os.environ['prompt']
input = os.environ['input']

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "ibm-granite/granite-vision-3.1-2b-preview"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForVision2Seq.from_pretrained(model_path).to(device)

# prepare image and text prompt, using the appropriate prompt template

img_path = Image.open(input).convert("RGB")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": img_path},
            {"type": "text", "text": prompt},
        ],
    },
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(device)


# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=1000)
print(processor.decode(output[0], skip_special_tokens=True))
