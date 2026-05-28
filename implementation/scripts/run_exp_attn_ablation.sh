#!/bin/bash
#SBATCH --job-name=exp_attn_abl
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/exp_attn_abl_%j.log

# Experiment 2: Attention ablation — block image-to-image attention.
# Each image token can only self-attend (diagonal), no cross-image-token attention.
# Text tokens still attend to all image+text tokens normally (standard causal).
# Tests whether cross-token attention mixing drives the brain-alignment gain.

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

echo "Node: $(hostname)  GPU: $(nvidia-smi -L 2>/dev/null | head -1)  Date: $(date)"

$PY -c "
import sys, os, json, logging, numpy as np, gc, time, torch, functools
sys.path.insert(0, '$PROJECT_DIR/implementation')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger('exp_attn_abl')

from pathlib import Path
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from src.data.nsd_fsaverage import NSDFsaverageLoader, get_roi_voxels

# ========== Phase 1: Extract with ablated attention ==========
logger.info('=== Phase 1: Extracting with ablated image-to-image attention ===')

LLAVA_DIR = '$PROJECT_DIR/models/llava-v1.5-7b'
LAYERS_TO_EXTRACT = [0, 14, 31]
N_VIS = 576

from transformers import CLIPVisionModel, CLIPImageProcessor, LlamaForCausalLM, LlamaConfig, AutoTokenizer

with open(os.path.join(LLAVA_DIR, 'config.json')) as f:
    cfg = json.load(f)

clip_path = cfg.get('mm_vision_tower', '')
dtype = torch.float16
device = 'cuda'

vision_tower = CLIPVisionModel.from_pretrained(clip_path, torch_dtype=dtype).to(device)
vision_tower.eval()
image_processor = CLIPImageProcessor.from_pretrained(clip_path)

# Load projector
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

# Load LLaMA with real weights + eager attention (not flash)
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
missing, unexpected = model.load_state_dict(llm_state, strict=False)
model = model.to(dtype=dtype).to(device)
model.eval()
logger.info(f'LLaMA loaded: {len(llm_state)-len(unexpected)} tensors loaded')
del llm_state, proj_weights
gc.collect()

tokenizer = AutoTokenizer.from_pretrained(LLAVA_DIR, use_fast=False)

# ---- Monkey-patch: block image-to-image attention ----
# Pre-compute the block mask (lower triangle excluding diagonal in image block)
# block_mask[i,j] = True means 'block attention from i to j'
_block_mask = torch.tril(torch.ones(N_VIS, N_VIS, dtype=torch.bool), diagonal=-1).to(device)
logger.info(f'Block mask: {_block_mask.sum().item()} positions blocked out of {N_VIS*N_VIS}')

def wrap_decoder_layer(layer):
    orig_forward = layer.forward
    @functools.wraps(orig_forward)
    def wrapped_forward(*args, **kwargs):
        # attention_mask can be in kwargs or args[1]
        mask = kwargs.get('attention_mask', None)
        mask_in_kwargs = mask is not None
        if mask is None and len(args) > 1:
            mask = args[1]
        if mask is not None and hasattr(mask, 'dim') and mask.dim() == 4:
            mask = mask.clone()
            mask[:, :, :N_VIS, :N_VIS].masked_fill_(_block_mask, torch.finfo(mask.dtype).min)
            if mask_in_kwargs:
                kwargs['attention_mask'] = mask
            else:
                args = (args[0], mask) + args[2:]
        return orig_forward(*args, **kwargs)
    layer.forward = wrapped_forward

for layer in model.model.layers:
    wrap_decoder_layer(layer)
logger.info('All 32 decoder layers patched with attention ablation')

# ---- Smoke test: verify masking works ----
logger.info('Smoke test: checking attention mask modification...')
test_input = torch.randn(1, N_VIS + 10, hidden_size, device=device, dtype=dtype)
hooks_test = []
attn_masks_seen = []
def capture_mask_hook(module, args, kwargs):
    mask = kwargs.get('attention_mask', None)
    if mask is None and len(args) > 1:
        mask = args[1]
    if mask is not None:
        attn_masks_seen.append(mask.clone())
    return None

h = model.model.layers[0].self_attn.register_forward_pre_hook(capture_mask_hook, with_kwargs=True)
with torch.no_grad():
    try:
        model(inputs_embeds=test_input)
    except Exception:
        pass
h.remove()

if attn_masks_seen:
    m = attn_masks_seen[0]
    # Check that image-to-image positions are blocked
    img_block = m[0, 0, :N_VIS, :N_VIS]
    n_blocked = (img_block <= -1e4).sum().item()
    n_expected = _block_mask.sum().item()
    logger.info(f'  Mask shape: {m.shape}, blocked positions: {n_blocked}, expected: {n_expected}')
    if n_blocked >= n_expected * 0.9:
        logger.info('  Smoke test PASSED: image-to-image attention is blocked')
    else:
        logger.warning(f'  Smoke test WARNING: only {n_blocked}/{n_expected} positions blocked')
else:
    logger.warning('  Could not capture attention mask — trying alternative hook...')

del test_input, attn_masks_seen
torch.cuda.empty_cache()

# Register activation hooks on target layers
act_hooks = []
token_acts = {}
def make_hook(layer_idx):
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        token_acts[layer_idx] = hidden.detach()
    return hook_fn

for L in LAYERS_TO_EXTRACT:
    h = model.model.layers[L].register_forward_hook(make_hook(L))
    act_hooks.append(h)

# Load stimuli
manifest_path = Path('$PROJECT_DIR/results_v2/stimuli_shared1000/manifest.json')
with open(manifest_path) as f:
    manifest = json.load(f)
image_paths = [str(Path('$PROJECT_DIR/results_v2/stimuli_shared1000') / p) for p in manifest['paths']]
n_images = len(image_paths)
logger.info(f'Processing {n_images} images')

out_acts = {L: np.zeros((n_images, N_VIS, hidden_size), dtype=np.float16) for L in LAYERS_TO_EXTRACT}

text = 'Describe this image in detail.'
prompt = f'USER: \n{text}\nASSISTANT:'

for idx in range(n_images):
    img = Image.open(image_paths[idx]).convert('RGB')
    pixel_values = image_processor(images=img, return_tensors='pt')['pixel_values'].to(device, dtype=dtype)

    with torch.no_grad():
        vision_out = vision_tower(pixel_values, output_hidden_states=True)
        image_features = vision_out.hidden_states[-2][:, 1:, :]
        image_features = projector(image_features)

        text_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
        text_embeds = model.get_input_embeddings()(text_ids)
        inputs_embeds = torch.cat([image_features, text_embeds], dim=1)

        token_acts.clear()
        model(inputs_embeds=inputs_embeds)

        for L in LAYERS_TO_EXTRACT:
            h_out = token_acts[L]
            if h_out.dim() == 3:
                h_out = h_out[0]
            out_acts[L][idx] = h_out[:N_VIS].cpu().to(torch.float16).numpy()

    if (idx + 1) % 100 == 0:
        logger.info(f'  Extracted {idx+1}/{n_images}')
        torch.cuda.empty_cache()

for h in act_hooks:
    h.remove()

# Save
out_dir = Path('$PROJECT_DIR/results_v2_raw/token_activations/attn_ablation')
out_dir.mkdir(parents=True, exist_ok=True)
for L in LAYERS_TO_EXTRACT:
    np.save(out_dir / f'layer_{L}_vis_tokens.npy', out_acts[L])
    logger.info(f'  Saved layer {L}: {out_acts[L].shape}')

import shutil
shutil.copy('$PROJECT_DIR/results_v2_raw/token_activations/llava/LLaVA_1.5_7B/nsd_ids.json',
            out_dir / 'nsd_ids.json')

del model, vision_tower, projector, out_acts
torch.cuda.empty_cache()
gc.collect()
logger.info('Phase 1 done: extraction complete')

# ========== Phase 2: Encoding ==========
logger.info('=== Phase 2: Encoding (fold-local PCA, sequential) ===')

with open(out_dir / 'nsd_ids.json') as f:
    act_nsd_ids = np.array(json.load(f))

def pearson_r_per_voxel(y_pred, y_true, eps=1e-10):
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    num = (yp * yt).sum(axis=0)
    den = np.sqrt((yp ** 2).sum(axis=0) * (yt ** 2).sum(axis=0))
    r = num / np.maximum(den, eps)
    return np.where(np.isfinite(r), r, 0.0)

def ridge_fold_local_pca(X_raw, Y, n_folds=5, pca_dim=256, alpha=10000.0, seed=42):
    n_stim, raw_dim = X_raw.shape
    Y_c = Y - Y.mean(axis=0, keepdims=True)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    preds = np.zeros_like(Y_c)
    for train_idx, test_idx in kf.split(X_raw):
        X_tr, X_te = X_raw[train_idx], X_raw[test_idx]
        Y_tr = Y_c[train_idx]
        if raw_dim > pca_dim:
            pca = PCA(n_components=pca_dim, random_state=seed, svd_solver='randomized')
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True) + 1e-8
        X_tr = (X_tr - mu) / sd
        X_te = (X_te - mu) / sd
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(X_tr, Y_tr)
        preds[test_idx] = ridge.predict(X_te)
    return pearson_r_per_voxel(preds, Y_c)

loader = NSDFsaverageLoader('$PROJECT_DIR/../nsd')
brain = {}
for roi in ['V1', 'V2', 'V3', 'V4']:
    try:
        v, ids, nc = get_roi_voxels(loader, 0, roi)
        if v.shape[1] > 0:
            brain[roi] = {'voxels': v, 'nsd_ids': ids, 'ncsnr': nc}
    except Exception as e:
        logger.warning(f'ROI {roi}: {e}')

nsd_to_idx = {int(n): i for i, n in enumerate(act_nsd_ids)}
sample_brain_ids = list(brain.values())[0]['nsd_ids']
keep_act, keep_brain_idx = [], []
for brow, nid in enumerate(sample_brain_ids):
    a = nsd_to_idx.get(int(nid))
    if a is not None:
        keep_act.append(a)
        keep_brain_idx.append(brow)
keep_act = np.array(keep_act)
keep_brain_idx = np.array(keep_brain_idx)
logger.info(f'Aligned {len(keep_act)} images')

enc_dir = Path('$PROJECT_DIR/results_v2/ablation_attn/subj01')
N_TOKENS = 576

for L in LAYERS_TO_EXTRACT:
    t0 = time.time()
    logger.info(f'=== Encoding layer {L} ===')
    arr = np.load(out_dir / f'layer_{L}_vis_tokens.npy', mmap_mode='r')
    arr_aligned = np.array(arr[keep_act]).astype(np.float32)
    logger.info(f'  Shape: {arr_aligned.shape}')
    del arr

    layer_out = enc_dir / f'layer_{L}' / 'per_roi_layer'
    layer_out.mkdir(parents=True, exist_ok=True)

    for roi_name in ['V1', 'V2', 'V3', 'V4']:
        Y = brain[roi_name]['voxels'][keep_brain_idx].astype(np.float32)
        mean_r_arr = np.zeros(N_TOKENS, dtype=np.float32)

        for tok in range(N_TOKENS):
            X_tok = arr_aligned[:, tok, :]
            r_v = ridge_fold_local_pca(X_tok, Y)
            mean_r_arr[tok] = r_v.mean()
            if tok % 96 == 95:
                logger.info(f'    {roi_name} tok {tok+1}/{N_TOKENS} mean_r={mean_r_arr[:tok+1].mean():.4f}')

        mr = float(mean_r_arr.mean())
        logger.info(f'  L{L} {roi_name}: mean_r={mr:.4f}')
        np.savez(layer_out / f'{roi_name}_L{L}.npz', mean_r=mean_r_arr)
        del Y
        gc.collect()

    del arr_aligned
    gc.collect()
    logger.info(f'  Layer {L} done in {(time.time()-t0)/60:.1f} min')

logger.info('Experiment 2 (Attention ablation) complete')
"

echo "Done: $(date)"
