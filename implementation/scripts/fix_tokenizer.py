"""Fix tokenizer and config for HF LlavaForConditionalGeneration."""
import json

MODEL_DIR = "/data/user/mzhang630/data/mllm/models/llava-v1.5-7b"

# Fix config.json
with open(f"{MODEL_DIR}/config.json") as f:
    cfg = json.load(f)
cfg["image_token_index"] = 32000
cfg["vocab_size"] = 32064
if "text_config" in cfg:
    cfg["text_config"]["vocab_size"] = 32064
with open(f"{MODEL_DIR}/config.json", "w") as f:
    json.dump(cfg, f, indent=2)
print("Fixed config.json: image_token_index=32000, vocab_size=32064")

# Fix tokenizer
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
tok.add_tokens(["<image>"], special_tokens=True)
print(f"Added <image> token, id={tok.convert_tokens_to_ids('<image>')}, vocab={len(tok)}")
tok.save_pretrained(MODEL_DIR)
print("Tokenizer saved")

# Test processor
from transformers import AutoProcessor
from PIL import Image

proc = AutoProcessor.from_pretrained(MODEL_DIR)
print(f"patch_size: {proc.patch_size}")

img = Image.new("RGB", (336, 336), color="red")
inputs = proc(text="USER: <image>\nDescribe.\nASSISTANT:", images=img, return_tensors="pt")
print(f"input_ids shape: {inputs['input_ids'].shape}")
print(f"pixel_values shape: {inputs['pixel_values'].shape}")
print("PROCESSOR WORKS!")
