#!/bin/bash
#SBATCH --job-name=sr_captext
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/sr_captext_%j.log

# Extract TEXT-only LLaVA features for the 70 SemReps test CAPTIONS.
# Mirrors run_semreps_extract.sh (image side) so the two are directly comparable:
#   - same 9 decoder layers, mean-pooled over caption tokens
#   - ordered by the SAME coco_ids as semreps_test_features.npz (image side)
# Output: results_semreps/semreps_test_caption_features.npz

export PROJECT_DIR=/data/user/mzhang630/data/mllm
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python
export PYTHONUNBUFFERED=1

echo "Node: $(hostname)  GPU: $(nvidia-smi -L 2>/dev/null | head -1)  Date: $(date)"

$PY << 'PYEOF'
import sys, os, json, logging, numpy as np, torch, gc
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('sr_captext')

PROJECT_DIR = '/data/user/mzhang630/data/mllm'
SEMREPS_DIR = '/data/user/mzhang630/data/semreps'
LLAVA_DIR = f'{PROJECT_DIR}/models/llava-v1.5-7b'
LAYERS = [0, 4, 8, 12, 14, 16, 20, 24, 31]

# ---- Order captions by the SAME coco_ids as the image-feature npz ----
img_npz = np.load(f'{PROJECT_DIR}/results_semreps/semreps_test_features.npz')
coco_ids = [int(c) for c in img_npz['coco_ids']]
logger.info(f'{len(coco_ids)} coco_ids (ordered to match image features)')

with open(f'{SEMREPS_DIR}/test_captions.json') as f:
    cap_map = json.load(f)  # {"146411": "A cat ...", ...}  keys are str
captions = []
for cid in coco_ids:
    key = str(cid)
    if key not in cap_map:
        raise KeyError(f'caption missing for coco_id {cid}')
    captions.append(cap_map[key])
n = len(captions)
logger.info(f'{n} captions aligned. e.g. [0] {coco_ids[0]}: "{captions[0]}"')

# ---- Load model (same loader as image-side script) ----
logger.info('Loading LLaVA (LLaMA decoder + tokenizer)...')
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer

with open(os.path.join(LLAVA_DIR, 'config.json')) as f:
    cfg = json.load(f)
dtype = torch.float16
device = 'cuda'
hidden_size = cfg.get('hidden_size', 4096)

idx_path = os.path.join(LLAVA_DIR, 'pytorch_model.bin.index.json')
with open(idx_path) as f:
    idx = json.load(f)
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
            pass
        elif clean_k.startswith('model.vision_tower.'):
            pass
        elif clean_k.startswith('model.') or clean_k == 'lm_head.weight':
            llm_state[clean_k] = v
    del shard

llama_cfg = LlamaConfig(
    vocab_size=cfg.get('vocab_size', 32000),
    hidden_size=hidden_size,
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
del llm_state
gc.collect()

tokenizer = AutoTokenizer.from_pretrained(LLAVA_DIR, use_fast=False)
logger.info('Model loaded')

# ---- Hooks on same layers ----
hooks = []
token_acts = {}
def make_hook(layer_idx):
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        token_acts[layer_idx] = hidden.detach()
    return hook_fn
for L in LAYERS:
    hooks.append(model.model.layers[L].register_forward_hook(make_hook(L)))

embed_features = np.zeros((n, hidden_size), dtype=np.float32)            # input word-embedding mean (non-contextual text baseline)
layer_features = {L: np.zeros((n, hidden_size), dtype=np.float32) for L in LAYERS}

logger.info('Extracting caption text features...')
for i, cap in enumerate(captions):
    # Pure caption text: BOS + caption tokens. Pool over caption content tokens (skip BOS).
    ids = tokenizer(cap, return_tensors='pt')['input_ids'].to(device)  # (1, T) includes BOS
    with torch.no_grad():
        embeds = model.get_input_embeddings()(ids)  # (1, T, 4096)
        # pool indices: skip BOS at position 0
        pool_slice = slice(1, ids.shape[1]) if ids.shape[1] > 1 else slice(0, ids.shape[1])
        embed_features[i] = embeds[0, pool_slice].cpu().float().mean(dim=0).numpy()

        token_acts.clear()
        model(inputs_embeds=embeds)
        for L in LAYERS:
            h = token_acts[L]
            if h.dim() == 3:
                h = h[0]
            layer_features[L][i] = h[pool_slice].cpu().float().mean(dim=0).numpy()
    if (i + 1) % 10 == 0:
        logger.info(f'  {i+1}/{n}')

for h in hooks:
    h.remove()

out_path = f'{PROJECT_DIR}/results_semreps/semreps_test_caption_features.npz'
save_dict = {'coco_ids': np.array(coco_ids), 'embed_features': embed_features}
for L in LAYERS:
    save_dict[f'layer_{L}'] = layer_features[L]
np.savez(out_path, **save_dict)
logger.info(f'Saved {out_path}')
logger.info(f'Shapes: embed={embed_features.shape}, L14={layer_features[14].shape}')
logger.info('Caption text extraction complete')
PYEOF

echo "Done: $(date)"
