#!/bin/bash
#SBATCH --job-name=exp_attn_sp
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/exp_attn_sp_%j.log

# Direction B: Extract attention maps and analyze spatial structure.
# For each layer, extract the 576×576 attention matrix (image tokens only),
# then test whether spatially adjacent tokens attend more to each other
# (analogous to lateral connections in visual cortex).

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python
export PYTHONUNBUFFERED=1

echo "Node: $(hostname)  GPU: $(nvidia-smi -L 2>/dev/null | head -1)  Date: $(date)"

$PY -c "
import sys, os, json, logging, numpy as np, gc, time, torch
sys.path.insert(0, '$PROJECT_DIR/implementation')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('exp_attn_sp')

from pathlib import Path
from PIL import Image
from scipy import stats

LLAVA_DIR = '$PROJECT_DIR/models/llava-v1.5-7b'
LAYERS_TO_EXTRACT = [0, 4, 8, 12, 14, 16, 20, 24, 31]
N_VIS = 576
GRID_H, GRID_W = 24, 24

# ========== Load model ==========
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
llama_cfg._attn_implementation = 'eager'
model = LlamaForCausalLM(llama_cfg)
model.load_state_dict(llm_state, strict=False)
model = model.to(dtype=dtype).to(device)
model.eval()
del llm_state, proj_weights
gc.collect()

tokenizer = AutoTokenizer.from_pretrained(LLAVA_DIR, use_fast=False)
logger.info('Model loaded')

# ========== Hook to capture attention weights ==========
attn_weights_per_layer = {}

def make_attn_hook(layer_idx):
    def hook_fn(module, args, kwargs, output):
        # In eager mode, LlamaAttention returns (attn_output, attn_weights, past_kv)
        # when output_attentions=True
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            # attn_weights: (batch, n_heads, seq_len, seq_len)
            aw = output[1].detach()
            # Average across heads, keep only image block: (N_VIS, N_VIS)
            aw_img = aw[0, :, :N_VIS, :N_VIS].mean(dim=0).cpu().float()
            if layer_idx not in attn_weights_per_layer:
                attn_weights_per_layer[layer_idx] = []
            attn_weights_per_layer[layer_idx].append(aw_img)  # (576, 576)
    return hook_fn

hooks = []
for i, layer in enumerate(model.model.layers):
    if i in LAYERS_TO_EXTRACT:
        h = layer.self_attn.register_forward_hook(make_attn_hook(i), with_kwargs=True)
        hooks.append(h)
logger.info(f'Hooks on {len(hooks)} layers for attention capture')

# ========== Extract attention maps ==========
manifest_path = Path('$PROJECT_DIR/results_v2/stimuli_shared1000/manifest.json')
with open(manifest_path) as f:
    manifest = json.load(f)
image_paths = [str(Path('$PROJECT_DIR/results_v2/stimuli_shared1000') / p) for p in manifest['paths']]

N_SAMPLE = 200  # use 200 images (enough for stable average)
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(image_paths), size=N_SAMPLE, replace=False)
sample_paths = [image_paths[i] for i in sample_idx]

text = 'Describe this image in detail.'
prompt = f'USER: \n{text}\nASSISTANT:'

logger.info(f'Extracting attention maps from {N_SAMPLE} images...')

for idx, img_path in enumerate(sample_paths):
    img = Image.open(img_path).convert('RGB')
    pixel_values = image_processor(images=img, return_tensors='pt')['pixel_values'].to(device, dtype=dtype)

    with torch.no_grad():
        vision_out = vision_tower(pixel_values, output_hidden_states=True)
        image_features = vision_out.hidden_states[-2][:, 1:, :]
        image_features = projector(image_features)
        text_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
        text_embeds = model.get_input_embeddings()(text_ids)
        inputs_embeds = torch.cat([image_features, text_embeds], dim=1)
        model(inputs_embeds=inputs_embeds, output_attentions=True)

    if (idx + 1) % 50 == 0:
        logger.info(f'  {idx+1}/{N_SAMPLE}')
        torch.cuda.empty_cache()

for h in hooks:
    h.remove()
del model, vision_tower, projector
torch.cuda.empty_cache()
gc.collect()

# ========== Compute average attention maps ==========
logger.info('Computing average attention maps...')
out_dir = Path('$PROJECT_DIR/results_v2/attention_spatial')
out_dir.mkdir(parents=True, exist_ok=True)

# Pre-compute spatial distance matrix for 24x24 grid
rows = np.arange(GRID_H).repeat(GRID_W)  # [0,0,...,0, 1,1,...,1, ...]
cols = np.tile(np.arange(GRID_W), GRID_H)  # [0,1,...,23, 0,1,...,23, ...]
# Euclidean distance between each pair of tokens
dist_matrix = np.sqrt((rows[:, None] - rows[None, :]) ** 2 + (cols[:, None] - cols[None, :]) ** 2)

results = {}
for L in LAYERS_TO_EXTRACT:
    if L not in attn_weights_per_layer or len(attn_weights_per_layer[L]) == 0:
        logger.warning(f'No attention weights for layer {L}')
        continue

    # Stack and average: (N_SAMPLE, 576, 576) -> (576, 576)
    all_attn = torch.stack(attn_weights_per_layer[L], dim=0)  # (N, 576, 576)
    head_avg_attn = all_attn.mean(dim=0).numpy()  # (576, 576) — already head-averaged

    # Only look at causal positions (j <= i)
    mask_causal = np.tril(np.ones((N_VIS, N_VIS), dtype=bool))

    # Flatten causal positions
    attn_flat = head_avg_attn[mask_causal]
    dist_flat = dist_matrix[mask_causal]

    # Remove self-attention (distance=0) for correlation
    nonself = dist_flat > 0
    attn_nonself = attn_flat[nonself]
    dist_nonself = dist_flat[nonself]

    # Spearman correlation: attention vs distance
    rho, pval = stats.spearmanr(dist_nonself, attn_nonself)

    # Bin by distance and compute mean attention
    dist_bins = np.arange(0, 35, 1)
    bin_means = []
    for b in range(len(dist_bins) - 1):
        mask_bin = (dist_nonself >= dist_bins[b]) & (dist_nonself < dist_bins[b+1])
        if mask_bin.sum() > 0:
            bin_means.append((dist_bins[b] + 0.5, attn_nonself[mask_bin].mean()))
    bin_means = np.array(bin_means)

    logger.info(f'  L{L}: Spearman rho={rho:.4f} (p={pval:.2e}), nearby(d<3) attn={attn_nonself[dist_nonself<3].mean():.6f}, far(d>15) attn={attn_nonself[dist_nonself>15].mean():.6f}')

    results[L] = {
        'spearman_rho': rho,
        'spearman_p': pval,
        'nearby_attn': float(attn_nonself[dist_nonself < 3].mean()),
        'far_attn': float(attn_nonself[dist_nonself > 15].mean()),
        'ratio': float(attn_nonself[dist_nonself < 3].mean() / max(attn_nonself[dist_nonself > 15].mean(), 1e-10)),
    }

    np.savez(out_dir / f'layer_{L}_attn_spatial.npz',
        head_avg_attn=head_avg_attn,
        dist_matrix=dist_matrix,
        bin_means=bin_means,
        spearman_rho=rho,
        spearman_p=pval)

    del all_attn
    gc.collect()

# Summary
logger.info('=== Summary: Attention spatial structure ===')
logger.info(f'{\"Layer\":<8} {\"Spearman_rho\":<15} {\"Nearby(d<3)\":<15} {\"Far(d>15)\":<15} {\"Ratio\":<10}')
for L in sorted(results.keys()):
    r = results[L]
    logger.info(f'L{L:<6} {r[\"spearman_rho\"]:>12.4f}   {r[\"nearby_attn\"]:>12.6f}   {r[\"far_attn\"]:>12.6f}   {r[\"ratio\"]:>8.2f}x')

# Save summary
import json as json_mod
with open(out_dir / 'summary.json', 'w') as f:
    json_mod.dump(results, f, indent=2, default=str)

logger.info('Direction B (attention spatial analysis) complete')
"

echo "Done: $(date)"
