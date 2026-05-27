"""Resize embedding weights to match config vocab_size=32064."""
import torch
import json

MODEL_DIR = "/data/user/mzhang630/data/mllm/models/llava-v1.5-7b"

# Load config
with open(f"{MODEL_DIR}/config.json") as f:
    cfg = json.load(f)

target_vocab = cfg.get("vocab_size", 32064)
print(f"Target vocab size: {target_vocab}")

# Load shard containing embeddings (shard 1)
shard1_path = f"{MODEL_DIR}/pytorch_model-00001-of-00002.bin"
print(f"Loading {shard1_path}...")
state = torch.load(shard1_path, map_location="cpu", weights_only=True)

embed_key = "model.language_model.embed_tokens.weight"
if embed_key in state:
    old_shape = state[embed_key].shape
    print(f"Current embed_tokens shape: {old_shape}")
    if old_shape[0] < target_vocab:
        new_embed = torch.zeros(target_vocab, old_shape[1], dtype=state[embed_key].dtype)
        new_embed[:old_shape[0]] = state[embed_key]
        state[embed_key] = new_embed
        print(f"Resized embed_tokens: {old_shape} -> {new_embed.shape}")
    torch.save(state, shard1_path)
    print("Saved shard 1")

# Load shard containing lm_head (shard 2)
shard2_path = f"{MODEL_DIR}/pytorch_model-00002-of-00002.bin"
print(f"Loading {shard2_path}...")
state2 = torch.load(shard2_path, map_location="cpu", weights_only=True)

lm_head_key = "lm_head.weight"
if lm_head_key in state2:
    old_shape = state2[lm_head_key].shape
    print(f"Current lm_head shape: {old_shape}")
    if old_shape[0] < target_vocab:
        new_lm = torch.zeros(target_vocab, old_shape[1], dtype=state2[lm_head_key].dtype)
        new_lm[:old_shape[0]] = state2[lm_head_key]
        state2[lm_head_key] = new_lm
        print(f"Resized lm_head: {old_shape} -> {new_lm.shape}")
    torch.save(state2, shard2_path)
    print("Saved shard 2")

# Full test
print("\nTesting full model + processor load...")
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_DIR, device_map="cpu", low_cpu_mem_usage=True,
)

w = model.language_model.layers[0].self_attn.q_proj.weight
print(f"LLM: std={w.std():.6f} (loaded={w.std() > 0.001})")
v = model.vision_tower.vision_model.encoder.layers[0].self_attn.q_proj.weight
print(f"Vision: std={v.std():.6f} (loaded={v.std() > 0.001})")
p = model.multi_modal_projector.linear_1.weight
print(f"Projector: std={p.std():.6f} (loaded={p.std() > 0.001})")
e = model.language_model.embed_tokens.weight
print(f"Embeddings: shape={e.shape}, std={e.std():.6f}")

proc = AutoProcessor.from_pretrained(MODEL_DIR)
img = Image.new("RGB", (336, 336), color="red")
inputs = proc(text="USER: <image>\nDescribe.\nASSISTANT:", images=img, return_tensors="pt")
print(f"input_ids: {inputs['input_ids'].shape}, pixel_values: {inputs['pixel_values'].shape}")

with torch.no_grad():
    out = model(**inputs)
    print(f"logits shape: {out.logits.shape}")

print("\nFULL MODEL WORKS END-TO-END!")
