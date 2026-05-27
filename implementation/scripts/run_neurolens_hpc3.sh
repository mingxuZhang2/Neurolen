#!/bin/bash
#SBATCH --job-name=neurolens
#SBATCH --partition=acd_u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/neurolens_%j.log

PROJECT_DIR=/data/user/mzhang630/data/mllm
IMPL_DIR=$PROJECT_DIR/implementation
PYTHON=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

export PYTHONPATH=$IMPL_DIR:$PYTHONPATH

cd $IMPL_DIR

echo "=== NeuroLens Experiment ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi -L)"
echo "Date: $(date)"
echo ""

# Step 1: Generate synthetic brain-like data for pipeline validation
echo "=== Step 1: Generating synthetic brain data ==="
$PYTHON -c "
import sys
sys.path.insert(0, '$IMPL_DIR')

import numpy as np
import os
import yaml
from pathlib import Path

np.random.seed(42)
N_STIMULI = 200  # Start with 200 images for speed
N_SUBJECTS = 4
HIDDEN_DIM = 256

# Load model config for stage boundaries
with open('configs/models.yaml') as f:
    config = yaml.safe_load(f)
llava_cfg = config['models']['llava']
stages = llava_cfg['stages']

# Define ROIs
roi_groups = {
    'visual_early': {'n_voxels': 500, 'stage_affinity': 'visual'},
    'multimodal_integration': {'n_voxels': 300, 'stage_affinity': 'fusion'},
    'language_network': {'n_voxels': 200, 'stage_affinity': 'language'},
}

# For each ROI, generate synthetic voxel patterns that have known alignment
# with the expected MLLM stage
brain_dir = Path('$PROJECT_DIR/brain_data/synthetic')
brain_dir.mkdir(parents=True, exist_ok=True)

for subj in range(1, N_SUBJECTS + 1):
    subj_dir = brain_dir / f'sub-{subj:02d}'
    subj_dir.mkdir(parents=True, exist_ok=True)
    
    for roi_name, roi_info in roi_groups.items():
        n_voxels = roi_info['n_voxels']
        # Generate voxel patterns with noise
        # These will be compared with model activations later
        voxel_patterns = np.random.randn(N_STIMULI, n_voxels).astype(np.float32)
        
        # Add structured signal based on stage affinity
        # (This will be replaced by real fMRI data)
        signal = np.random.randn(N_STIMULI, 50).astype(np.float32)
        projection = np.random.randn(50, n_voxels).astype(np.float32) * 0.1
        voxel_patterns += signal @ projection
        
        np.save(subj_dir / f'{roi_name}_voxels.npy', voxel_patterns)
    
    print(f'  Generated sub-{subj:02d}')

# Generate stimulus list (use random images from COCO val if available, else random)
stim_dir = Path('$PROJECT_DIR/stimuli')
stim_dir.mkdir(parents=True, exist_ok=True)

import json
stimulus_info = {
    'n_stimuli': N_STIMULI,
    'image_paths': [],
    'texts': [],
    'nsd_ids': list(range(N_STIMULI)),
}

# Check for real images
coco_dirs = [
    '/data/user/mzhang630/data/mllm/coco/val2017',
    '/data/user/mzhang630/data/mllm/sample_images',
]
img_dir = None
for d in coco_dirs:
    if os.path.isdir(d):
        img_dir = d
        break

if img_dir:
    import glob
    images = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))[:N_STIMULI]
    stimulus_info['image_paths'] = images
    stimulus_info['texts'] = ['Describe this image in detail.'] * len(images)
    print(f'  Using {len(images)} real images from {img_dir}')
else:
    # Generate random images as placeholder
    from PIL import Image
    sample_dir = Path('$PROJECT_DIR/sample_images')
    sample_dir.mkdir(parents=True, exist_ok=True)
    for i in range(N_STIMULI):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = str(sample_dir / f'img_{i:04d}.jpg')
        img.save(img_path)
        stimulus_info['image_paths'].append(img_path)
    stimulus_info['texts'] = ['Describe this image in detail.'] * N_STIMULI
    print(f'  Generated {N_STIMULI} random placeholder images')

with open(stim_dir / 'stimulus_pairs.json', 'w') as f:
    json.dump(stimulus_info, f)

print('Synthetic brain data and stimuli ready')
print(f'N_STIMULI={N_STIMULI}, N_SUBJECTS={N_SUBJECTS}')
"

echo ""
echo "=== Step 2: Extracting LLaVA activations ==="
$PYTHON -c "
import sys, os, json, yaml, logging
sys.path.insert(0, '$IMPL_DIR')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

import numpy as np
import torch
from pathlib import Path

# Load stimulus info
with open('$PROJECT_DIR/stimuli/stimulus_pairs.json') as f:
    stimuli = json.load(f)

# Load model config
with open('configs/models.yaml') as f:
    config = yaml.safe_load(f)
model_cfg = config['models']['llava']

# Override model path to local
model_cfg['hf_id'] = '$PROJECT_DIR/models/llava-v1.5-7b'

from src.models.extract_activations import ActivationExtractor

extractor = ActivationExtractor(model_cfg, device='cuda', dtype='float16')
print('Loading model...')
extractor.load_model()
print('Model loaded')

N = min(len(stimuli['image_paths']), 200)
images = stimuli['image_paths'][:N]
texts = stimuli['texts'][:N]

print(f'Extracting activations for {N} stimuli...')
extractor.extract_all(
    images=images,
    texts=texts,
    output_dir='$PROJECT_DIR/activations',
    pca_dim=256,
)
extractor.cleanup()
print('Activation extraction complete')
"

echo ""
echo "=== Step 3: Computing model RDMs ==="
$PYTHON -c "
import sys
sys.path.insert(0, '$IMPL_DIR')
import yaml
from src.models.model_rdm import compute_model_rdms

with open('configs/models.yaml') as f:
    config = yaml.safe_load(f)
model_cfg = config['models']['llava']
model_name = model_cfg['name'].replace(' ', '_').replace('-', '_')

compute_model_rdms(
    activations_dir='$PROJECT_DIR/activations',
    model_name=model_name,
    n_layers=model_cfg['n_layers'],
    method='correlation',
    output_dir='$PROJECT_DIR/rdms',
)
print('Model RDMs computed')
"

echo ""
echo "=== Step 4: RSA Analysis ==="
$PYTHON -c "
import sys, os, yaml
sys.path.insert(0, '$IMPL_DIR')
import numpy as np
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

# Compute brain RDMs from synthetic data
brain_dir = Path('$PROJECT_DIR/brain_data/synthetic')
brain_rdm_dir = Path('$PROJECT_DIR/brain_rdms')
brain_rdm_dir.mkdir(parents=True, exist_ok=True)

roi_names = ['visual_early', 'multimodal_integration', 'language_network']
subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04']

for subj in subjects:
    for roi in roi_names:
        voxel_file = brain_dir / subj / f'{roi}_voxels.npy'
        if voxel_file.exists():
            voxels = np.load(voxel_file)
            # Compute RDM using correlation distance
            from scipy.spatial.distance import pdist, squareform
            rdm = squareform(pdist(voxels, metric='correlation'))
            np.save(brain_rdm_dir / f'{subj}_{roi}_rdm.npy', rdm)

print('Brain RDMs computed')

# Now compute RSA: compare model RDMs to brain RDMs
rdm_dir = Path('$PROJECT_DIR/rdms')
results_dir = Path('$PROJECT_DIR/results/rsa')
results_dir.mkdir(parents=True, exist_ok=True)

with open('configs/models.yaml') as f:
    config = yaml.safe_load(f)
model_cfg = config['models']['llava']
model_name = model_cfg['name'].replace(' ', '_').replace('-', '_')
n_layers = model_cfg['n_layers']

# Load model RDMs
model_rdms = {}
for l in range(n_layers):
    for tt in ['vis', 'txt', 'all']:
        f = rdm_dir / f'{model_name}_layer_{l}_{tt}_rdm.npy'
        if f.exists():
            model_rdms[(l, tt)] = np.load(f)

# Compute RSA (simplified - Spearman correlation between RDM upper triangles)
from scipy.stats import spearmanr

rsa_matrix = np.zeros((n_layers, len(roi_names)))  # (layers x ROIs)

for layer_idx in range(n_layers):
    for roi_idx, roi in enumerate(roi_names):
        correlations = []
        for subj in subjects:
            brain_rdm_file = brain_rdm_dir / f'{subj}_{roi}_rdm.npy'
            model_rdm_key = (layer_idx, 'all')
            
            if brain_rdm_file.exists() and model_rdm_key in model_rdms:
                brain_rdm = np.load(brain_rdm_file)
                model_rdm = model_rdms[model_rdm_key]
                
                # Match sizes
                n = min(brain_rdm.shape[0], model_rdm.shape[0])
                brain_upper = brain_rdm[:n, :n][np.triu_indices(n, k=1)]
                model_upper = model_rdm[:n, :n][np.triu_indices(n, k=1)]
                
                if len(brain_upper) > 10:
                    rho, _ = spearmanr(brain_upper, model_upper)
                    if np.isfinite(rho):
                        correlations.append(rho)
        
        rsa_matrix[layer_idx, roi_idx] = np.mean(correlations) if correlations else 0.0

np.savez(results_dir / f'rsa_{model_name}_all.npz',
         rsa_matrix=rsa_matrix,
         roi_names=roi_names,
         n_layers=n_layers)

print('RSA matrix shape:', rsa_matrix.shape)
print('RSA results saved')

# Print preview
stages = model_cfg['stages']
print()
print('=== RSA Results Preview ===')
print(f'{\"Layer\":<8}', end='')
for roi in roi_names:
    print(f'{roi:<25}', end='')
print()
for l in range(n_layers):
    stage_label = ''
    if stages['visual'][0] <= l <= stages['visual'][1]: stage_label = '[V]'
    elif stages['fusion'][0] <= l <= stages['fusion'][1]: stage_label = '[F]'
    elif stages['language'][0] <= l <= stages['language'][1]: stage_label = '[L]'
    print(f'L{l:<5}{stage_label:<3}', end='')
    for roi_idx in range(len(roi_names)):
        print(f'{rsa_matrix[l, roi_idx]:>24.4f}', end='')
    print()
"

echo ""
echo "=== Step 5: CKA Analysis ==="
$PYTHON -c "
import sys, yaml
sys.path.insert(0, '$IMPL_DIR')
import numpy as np
from pathlib import Path

with open('configs/models.yaml') as f:
    config = yaml.safe_load(f)
model_cfg = config['models']['llava']
model_name = model_cfg['name'].replace(' ', '_').replace('-', '_')
n_layers = model_cfg['n_layers']

brain_dir = Path('$PROJECT_DIR/brain_data/synthetic')
act_dir = Path('$PROJECT_DIR/activations') / model_name
results_dir = Path('$PROJECT_DIR/results/cka')
results_dir.mkdir(parents=True, exist_ok=True)

roi_names = ['visual_early', 'multimodal_integration', 'language_network']
subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04']

from src.analysis.cka_analysis import linear_cka

cka_matrix = np.zeros((n_layers, len(roi_names)))

for layer_idx in range(n_layers):
    act_file = act_dir / f'layer_{layer_idx}_all.npy'
    if not act_file.exists():
        continue
    model_act = np.load(act_file)
    
    for roi_idx, roi in enumerate(roi_names):
        cka_vals = []
        for subj in subjects:
            voxel_file = brain_dir / subj / f'{roi}_voxels.npy'
            if voxel_file.exists():
                brain_act = np.load(voxel_file)
                n = min(model_act.shape[0], brain_act.shape[0])
                cka_val = linear_cka(model_act[:n], brain_act[:n])
                if np.isfinite(cka_val):
                    cka_vals.append(cka_val)
        cka_matrix[layer_idx, roi_idx] = np.mean(cka_vals) if cka_vals else 0.0

np.savez(results_dir / f'cka_{model_name}_all_linear.npz',
         cka_matrix=cka_matrix,
         roi_names=roi_names,
         n_layers=n_layers)
print('CKA matrix computed:', cka_matrix.shape)
print('CKA results saved')
"

echo ""
echo "=== Step 6: Triple Dissociation (Patching) ==="
$PYTHON -c "
import sys, os, json, yaml
sys.path.insert(0, '$IMPL_DIR')
import numpy as np
import torch
import logging
logging.basicConfig(level=logging.INFO)

with open('configs/models.yaml') as f:
    config = yaml.safe_load(f)
model_cfg = config['models']['llava']
model_cfg['hf_id'] = '$PROJECT_DIR/models/llava-v1.5-7b'

# Load stimuli
with open('$PROJECT_DIR/stimuli/stimulus_pairs.json') as f:
    stimuli = json.load(f)

from src.models.extract_activations import ActivationExtractor
from src.patching.activation_patching import ActivationPatcher

extractor = ActivationExtractor(model_cfg, device='cuda', dtype='float16')
extractor.load_model()

patcher = ActivationPatcher(extractor.model, extractor.processor, model_cfg)

# Use a subset of images for patching (faster)
N_PATCH = min(50, len(stimuli['image_paths']))
images = stimuli['image_paths'][:N_PATCH]
texts = stimuli['texts'][:N_PATCH]

# Compute mean activations for mean ablation
print('Computing mean activations...')
patcher.compute_mean_activations(images[:30], texts[:30], n_samples=30)

stages = model_cfg['stages']
stage_ranges = {
    'visual': tuple(stages['visual']),
    'fusion': tuple(stages['fusion']),
    'language': tuple(stages['language']),
}

# Three task types with different prompts
task_prompts = {
    'visual_recognition': 'What is the main object in this image? Answer briefly.',
    'cross_modal_binding': 'What color is the largest object in this image?',
    'language_generation': 'Describe this image in detail.',
}

results_dir = f'$PROJECT_DIR/results/dissociation'
os.makedirs(results_dir, exist_ok=True)

# Run baseline first
print('Running baseline...')
baseline = {}
for task_name, prompt in task_prompts.items():
    responses = []
    for img_path in images[:20]:
        try:
            result = patcher.patch_and_generate(
                image=img_path, text=prompt,
                layer_range=(0, 0), patch_type='zero',
                max_new_tokens=64,
            )
            patcher._remove_hooks()
            # Clean run
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            inputs = patcher._prepare_inputs(img, prompt)
            with torch.no_grad():
                out = patcher._run_generate(inputs, max_new_tokens=64)
            responses.append(out['text'])
        except Exception as e:
            responses.append(str(e))
    baseline[task_name] = responses
    print(f'  Baseline {task_name}: {len(responses)} responses')

# Run patching for each stage x task
dissociation_matrix = np.zeros((3, 3))  # stages x tasks
stage_names = ['visual', 'fusion', 'language']
task_names = list(task_prompts.keys())

for s_idx, stage_name in enumerate(stage_names):
    layer_range = stage_ranges[stage_name]
    print(f'\\nPatching stage: {stage_name} (layers {layer_range[0]}-{layer_range[1]})')
    
    for t_idx, task_name in enumerate(task_names):
        prompt = task_prompts[task_name]
        patched_responses = []
        
        for img_path in images[:20]:
            try:
                result = patcher.patch_and_generate(
                    image=img_path, text=prompt,
                    layer_range=layer_range,
                    patch_type='mean',
                    max_new_tokens=64,
                )
                patched_responses.append(result['text'])
            except Exception as e:
                patched_responses.append('')
        
        # Measure difference: compare patched vs baseline response length and content
        baseline_lens = [len(r) for r in baseline[task_name]]
        patched_lens = [len(r) for r in patched_responses]
        
        # Simple performance drop metric: 1 - (patched quality / baseline quality)
        baseline_mean = np.mean(baseline_lens) if baseline_lens else 1
        patched_mean = np.mean(patched_lens) if patched_lens else 0
        
        # For language task: measure length degradation
        # For visual/binding: measure response divergence
        from difflib import SequenceMatcher
        similarities = []
        for b, p in zip(baseline[task_name], patched_responses):
            sim = SequenceMatcher(None, b.lower(), p.lower()).ratio()
            similarities.append(sim)
        
        performance_drop = 1.0 - np.mean(similarities)
        dissociation_matrix[s_idx, t_idx] = performance_drop
        
        print(f'  {stage_name}/{task_name}: drop={performance_drop:.3f}')

# Save results
np.save(f'{results_dir}/dissociation_matrix.npy', dissociation_matrix)

print()
print('=== Triple Dissociation Matrix ===')
print(f'{\"\":>20}', end='')
for t in task_names:
    print(f'{t:>25}', end='')
print()
for s_idx, s in enumerate(stage_names):
    print(f'{s:>20}', end='')
    for t_idx in range(3):
        val = dissociation_matrix[s_idx, t_idx]
        marker = ' <<<' if s_idx == t_idx else ''
        print(f'{val:>20.3f}{marker:>5}', end='')
    print()

extractor.cleanup()
print()
print('Dissociation results saved')
"

echo ""
echo "=== Step 7: Generate Figures ==="
$PYTHON -c "
import sys, yaml
sys.path.insert(0, '$IMPL_DIR')
import numpy as np
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

with open('configs/models.yaml') as f:
    config = yaml.safe_load(f)
model_cfg = config['models']['llava']
model_name = model_cfg['name'].replace(' ', '_').replace('-', '_')
stages = model_cfg['stages']
boundaries = [stages['fusion'][0], stages['language'][0]]

fig_dir = Path('$PROJECT_DIR/results/figures')
fig_dir.mkdir(parents=True, exist_ok=True)

from src.visualization.heatmaps import plot_alignment_heatmap, plot_dissociation_matrix

# RSA heatmap
rsa_file = Path('$PROJECT_DIR/results/rsa') / f'rsa_{model_name}_all.npz'
if rsa_file.exists():
    data = np.load(rsa_file, allow_pickle=True)
    plot_alignment_heatmap(
        matrix=data['rsa_matrix'],
        roi_names=list(data['roi_names']),
        model_name=model_cfg['name'],
        metric_name='RSA',
        stage_boundaries=boundaries,
        output_path=str(fig_dir / 'rsa_heatmap_llava.pdf'),
    )
    print('RSA heatmap saved')

# CKA heatmap
cka_file = Path('$PROJECT_DIR/results/cka') / f'cka_{model_name}_all_linear.npz'
if cka_file.exists():
    data = np.load(cka_file, allow_pickle=True)
    plot_alignment_heatmap(
        matrix=data['cka_matrix'],
        roi_names=list(data['roi_names']),
        model_name=model_cfg['name'],
        metric_name='CKA',
        stage_boundaries=boundaries,
        output_path=str(fig_dir / 'cka_heatmap_llava.pdf'),
    )
    print('CKA heatmap saved')

# Dissociation matrix
diss_file = Path('$PROJECT_DIR/results/dissociation/dissociation_matrix.npy')
if diss_file.exists():
    matrix = np.load(diss_file)
    plot_dissociation_matrix(
        matrix=matrix,
        output_path=str(fig_dir / 'dissociation_llava_mean.pdf'),
    )
    print('Dissociation matrix figure saved')

print('All figures saved to', fig_dir)
"

echo ""
echo "=== EXPERIMENT COMPLETE ==="
echo "Results in: $PROJECT_DIR/results/"
echo "Figures in: $PROJECT_DIR/results/figures/"
echo "Date: $(date)"
