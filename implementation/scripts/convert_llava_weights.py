"""Convert original LLaVA checkpoint to HuggingFace LlavaForConditionalGeneration format.

Original format keys:
  model.layers.X... -> language_model weights
  model.embed_tokens... -> language_model embeddings
  model.norm... -> language_model norm
  lm_head... -> language_model lm_head
  model.mm_projector... -> multi_modal_projector

HF format keys:
  model.language_model.model.layers.X...
  model.language_model.model.embed_tokens...
  model.language_model.model.norm...
  model.language_model.lm_head...
  model.multi_modal_projector...
"""

import torch
import json
import os
import sys
from pathlib import Path
from collections import OrderedDict

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "/data/user/mzhang630/data/mllm/models/llava-v1.5-7b"

print(f"Converting weights in {MODEL_DIR}")

idx_path = os.path.join(MODEL_DIR, "pytorch_model.bin.index.json")
with open(idx_path) as f:
    index = json.load(f)

weight_map = index["weight_map"]
shard_files = set(weight_map.values())

KEY_MAPPING = {
    "model.embed_tokens.": "model.language_model.embed_tokens.",
    "model.layers.": "model.language_model.layers.",
    "model.norm.": "model.language_model.norm.",
    "lm_head.": "lm_head.",
    "model.mm_projector.": "model.multi_modal_projector.",
}

def remap_key(key):
    for old_prefix, new_prefix in KEY_MAPPING.items():
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix):]
    return key

# Fix the mm_projector key mapping:
# Original: model.mm_projector.0.weight -> HF: model.multi_modal_projector.linear_1.weight
MM_PROJ_MAP = {
    "model.multi_modal_projector.0.weight": "model.multi_modal_projector.linear_1.weight",
    "model.multi_modal_projector.0.bias": "model.multi_modal_projector.linear_1.bias",
    "model.multi_modal_projector.2.weight": "model.multi_modal_projector.linear_2.weight",
    "model.multi_modal_projector.2.bias": "model.multi_modal_projector.linear_2.bias",
}

new_weight_map = {}
for shard_file in sorted(shard_files):
    shard_path = os.path.join(MODEL_DIR, shard_file)
    print(f"Processing {shard_file}...")
    state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = remap_key(key)
        if new_key in MM_PROJ_MAP:
            new_key = MM_PROJ_MAP[new_key]
        new_state_dict[new_key] = value
        new_weight_map[new_key] = shard_file

    torch.save(new_state_dict, shard_path)
    print(f"  Saved {len(new_state_dict)} tensors")

# Also load and convert mm_projector.bin into the main shards
proj_path = os.path.join(MODEL_DIR, "mm_projector.bin")
if os.path.exists(proj_path):
    print("Processing mm_projector.bin...")
    proj_weights = torch.load(proj_path, map_location="cpu", weights_only=True)

    # Check if projector keys are already in the main shards
    proj_already_in = any("multi_modal_projector" in k for k in new_weight_map)
    if not proj_already_in:
        # Load the last shard and add projector weights
        last_shard = sorted(shard_files)[-1]
        last_shard_path = os.path.join(MODEL_DIR, last_shard)
        state_dict = torch.load(last_shard_path, map_location="cpu", weights_only=True)

        for key, value in proj_weights.items():
            new_key = remap_key(key)
            if new_key in MM_PROJ_MAP:
                new_key = MM_PROJ_MAP[new_key]
            state_dict[new_key] = value
            new_weight_map[new_key] = last_shard

        torch.save(state_dict, last_shard_path)
        print(f"  Added {len(proj_weights)} projector tensors to {last_shard}")

# Update the index file
new_index = {
    "metadata": index.get("metadata", {"total_size": 0}),
    "weight_map": new_weight_map,
}
with open(idx_path, "w") as f:
    json.dump(new_index, f, indent=2)

# Update config.json - set model_type to llava_next or keep llava
# and ensure architectures is correct
config_path = os.path.join(MODEL_DIR, "config.json")
with open(config_path) as f:
    config = json.load(f)

config["architectures"] = ["LlavaForConditionalGeneration"]
config["model_type"] = "llava"

# Add text_config for the language model
if "text_config" not in config:
    config["text_config"] = {
        "model_type": "llama",
        "hidden_size": config.get("hidden_size", 4096),
        "intermediate_size": config.get("intermediate_size", 11008),
        "num_attention_heads": config.get("num_attention_heads", 32),
        "num_hidden_layers": config.get("num_hidden_layers", 32),
        "num_key_value_heads": config.get("num_key_value_heads", 32),
        "vocab_size": config.get("vocab_size", 32000),
        "rms_norm_eps": config.get("rms_norm_eps", 1e-5),
        "max_position_embeddings": config.get("max_position_embeddings", 4096),
    }

# Add vision_config
if "vision_config" not in config:
    config["vision_config"] = {
        "model_type": "clip_vision_model",
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "image_size": 336,
        "patch_size": 14,
    }

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"\nConversion complete!")
print(f"Total keys: {len(new_weight_map)}")
print(f"Sample new keys:")
for k in sorted(new_weight_map.keys())[:5]:
    print(f"  {k}")
