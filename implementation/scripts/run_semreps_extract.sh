#!/bin/bash
#SBATCH --job-name=sr_extract
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/sr_extract_%j.log

# Extract LLaVA features for 70 SemReps test images.
# Saves mean-pooled (576 tokens averaged) features at multiple layers.

export PROJECT_DIR=/data/user/mzhang630/data/mllm
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python
export PYTHONUNBUFFERED=1

echo "Node: $(hostname)  GPU: $(nvidia-smi -L 2>/dev/null | head -1)  Date: $(date)"

$PY << 'PYEOF'
import sys, os, json, logging, numpy as np, torch, gc
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('sr_extract')

from pathlib import Path
from PIL import Image

PROJECT_DIR = '/data/user/mzhang630/data/mllm'
SEMREPS_DIR = '/data/user/mzhang630/data/semreps'
LLAVA_DIR = f'{PROJECT_DIR}/models/llava-v1.5-7b'
LAYERS = [0, 4, 8, 12, 14, 16, 20, 24, 31]
N_VIS = 576

# Load manifest
with open(f'{SEMREPS_DIR}/test_manifest.json') as f:
    manifest = json.load(f)
test_coco_ids = manifest['test_coco_ids']
n_images = len(test_coco_ids)
logger.info(f'{n_images} test images to process')

# Load model
logger.info('Loading LLaVA...')
from transformers import CLIPVisionModel, CLIPImageProcessor, LlamaForCausalLM, LlamaConfig, AutoTokenizer

with open(os.path.join(LLAVA_DIR, 'config.json')) as f:
    cfg = json.load(f)

clip_path = cfg.get('mm_vision_tower', '')
dtype = torch.float16
device = 'cuda'

vision_tower = CLIPVisionModel.from_pretrained(clip_path, torch_dtype=dtype).to(device)
vision_tower.eval()
image_processor = CLIPImageProcessor.from_pretrained(clip_path)

mm_hidden_size = cfg.get('mm_hidden_size', 1024)
hidden_size = cfg.get('hidden_size', 4096)
projector = torch.nn.Sequential(
    torch.nn.Linear(mm_hidden_size, hidden_size),
    torch.nn.GELU(),
    torch.nn.Linear(hidden_size, hidden_size),
).to(device, dtype=dtype)

idx_path = os.path.join(LLAVA_DIR, 'pytorch_model.bin.index.json')
with open(idx_path) as f:
    idx = json.load(f)
proj_weights = {}
llm_state = {}
for shard_file in sorted(set(idx['weight_map'].values())):
    shard = torch.load(os.path.join(LLAVA_DIR, shard_file), map_location='cpu', weights_only=True)
    for k, v in shard.items():
        clean_k = k
        for prefix in ['model.language_model.', 'model.language_model.model.']:
            if clean_k.startswith(prefix):
                clean_k = clean_k[len(prefix):]
                if not clean_k.startswith('model.'):
                    clean_k = 'model.' + clean_k
                break
        if 'mm_projector' in clean_k or 'multi_modal_projector' in clean_k:
            proj_weights[clean_k] = v
        elif clean_k.startswith('model.vision_tower.'):
            pass
        elif clean_k.startswith('model.') or clean_k == 'lm_head.weight':
            llm_state[clean_k] = v
    del shard

proj_state = {}
for k, v in proj_weights.items():
    clean = k
    for prefix in ('model.mm_projector.', 'model.multi_modal_projector.'):
        if clean.startswith(prefix):
            clean = clean[len(prefix):]
            break
    if clean.startswith('linear_1'): clean = clean.replace('linear_1', '0')
    if clean.startswith('linear_2'): clean = clean.replace('linear_2', '2')
    proj_state[clean] = v
projector.load_state_dict(proj_state)
projector.eval()

llama_cfg = LlamaConfig(
    vocab_size=cfg.get('vocab_size', 32000),
    hidden_size=cfg.get('hidden_size', 4096),
    intermediate_size=cfg.get('intermediate_size', 11008),
    num_hidden_layers=cfg.get('num_hidden_layers', 32),
    num_attention_heads=cfg.get('num_attention_heads', 32),
    num_key_value_heads=cfg.get('num_key_value_heads', 32),
    max_position_embeddings=cfg.get('max_position_embeddings', 4096),
    rms_norm_eps=cfg.get('rms_norm_eps', 1e-5),
    torch_dtype=dtype,
)
model = LlamaForCausalLM(llama_cfg)
model.load_state_dict(llm_state, strict=False)
model = model.to(dtype=dtype).to(device)
model.eval()
del llm_state, proj_weights
gc.collect()

tokenizer = AutoTokenizer.from_pretrained(LLAVA_DIR, use_fast=False)
logger.info('Model loaded')

# Hooks
hooks = []
token_acts = {}
def make_hook(layer_idx):
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        token_acts[layer_idx] = hidden.detach()
    return hook_fn

for L in LAYERS:
    h = model.model.layers[L].register_forward_hook(make_hook(L))
    hooks.append(h)

# Also extract CLIP features (pre-decoder baseline)
clip_features = np.zeros((n_images, 1024), dtype=np.float32)  # CLIP CLS-like (mean-pooled patches)
proj_features = np.zeros((n_images, hidden_size), dtype=np.float32)

# Per-layer mean-pooled features
layer_features = {L: np.zeros((n_images, hidden_size), dtype=np.float32) for L in LAYERS}

text = 'Describe this image in detail.'
prompt = f'USER: \n{text}\nASSISTANT:'

logger.info('Extracting...')
for idx, cid in enumerate(test_coco_ids):
    img_path = f'{SEMREPS_DIR}/images/{cid:012d}.jpg'
    img = Image.open(img_path).convert('RGB')
    pixel_values = image_processor(images=img, return_tensors='pt')['pixel_values'].to(device, dtype=dtype)

    with torch.no_grad():
        vision_out = vision_tower(pixel_values, output_hidden_states=True)
        clip_patches = vision_out.hidden_states[-2][:, 1:, :]  # (1, 576, 1024)
        image_features = projector(clip_patches)  # (1, 576, 4096)

        clip_features[idx] = clip_patches[0].cpu().float().mean(dim=0).numpy()
        proj_features[idx] = image_features[0].cpu().float().mean(dim=0).numpy()

        text_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
        text_embeds = model.get_input_embeddings()(text_ids)
        inputs_embeds = torch.cat([image_features, text_embeds], dim=1)

        token_acts.clear()
        model(inputs_embeds=inputs_embeds)

        for L in LAYERS:
            h_out = token_acts[L]
            if h_out.dim() == 3:
                h_out = h_out[0]
            layer_features[L][idx] = h_out[:N_VIS].cpu().float().mean(dim=0).numpy()

    if (idx + 1) % 10 == 0:
        logger.info(f'  {idx+1}/{n_images}')

for h in hooks:
    h.remove()

# Save
out_dir = Path(f'{PROJECT_DIR}/results_semreps')
out_dir.mkdir(parents=True, exist_ok=True)

save_dict = {
    'coco_ids': np.array(test_coco_ids),
    'clip_features': clip_features,
    'proj_features': proj_features,
}
for L in LAYERS:
    save_dict[f'layer_{L}'] = layer_features[L]

np.savez(out_dir / 'semreps_test_features.npz', **save_dict)
logger.info(f'Saved to {out_dir / "semreps_test_features.npz"}')
logger.info(f'Shapes: clip={clip_features.shape}, L14={layer_features[14].shape}')
logger.info('Extraction complete')
PYEOF

echo "Done: $(date)"
