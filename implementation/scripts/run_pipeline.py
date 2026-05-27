"""NeuroLens unified pipeline — runs all steps in a single process."""
import sys, os, json, yaml, logging
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger("neurolens")

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", "/data/user/mzhang630/data/mllm"))
IMPL_DIR = PROJECT_DIR / "implementation"
sys.path.insert(0, str(IMPL_DIR))

with open(IMPL_DIR / "configs" / "models.yaml") as f:
    models_config = yaml.safe_load(f)
model_cfg = models_config["models"]["llava"]
model_cfg["hf_id"] = str(PROJECT_DIR / "models" / "llava-v1.5-7b")

N_STIMULI = 200
N_SUBJECTS = 4
STAGES = model_cfg["stages"]

# ============================================================
# Step 1: Generate synthetic brain data
# ============================================================
logger.info("=== Step 1: Generating synthetic brain data ===")
np.random.seed(42)
brain_dir = PROJECT_DIR / "brain_data" / "synthetic"
stim_dir = PROJECT_DIR / "stimuli"
brain_dir.mkdir(parents=True, exist_ok=True)
stim_dir.mkdir(parents=True, exist_ok=True)

roi_groups = ["visual_early", "multimodal_integration", "language_network"]
roi_voxels = {"visual_early": 500, "multimodal_integration": 300, "language_network": 200}

for subj in range(1, N_SUBJECTS + 1):
    subj_dir = brain_dir / f"sub-{subj:02d}"
    subj_dir.mkdir(parents=True, exist_ok=True)
    for roi in roi_groups:
        voxels = np.random.randn(N_STIMULI, roi_voxels[roi]).astype(np.float32)
        signal = np.random.randn(N_STIMULI, 50).astype(np.float32)
        voxels += signal @ (np.random.randn(50, roi_voxels[roi]).astype(np.float32) * 0.1)
        np.save(subj_dir / f"{roi}_voxels.npy", voxels)
    logger.info(f"  Generated sub-{subj:02d}")

# Stimulus list
sample_dir = PROJECT_DIR / "sample_images"
if not sample_dir.exists():
    sample_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image as PILImage
    for i in range(N_STIMULI):
        img = PILImage.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(sample_dir / f"img_{i:04d}.jpg")
    logger.info(f"  Generated {N_STIMULI} placeholder images")

import glob
images = sorted(glob.glob(str(sample_dir / "*.jpg")))[:N_STIMULI]
texts = ["Describe this image in detail."] * N_STIMULI
stimulus_info = {"image_paths": images, "texts": texts, "n_stimuli": N_STIMULI}
with open(stim_dir / "stimulus_pairs.json", "w") as f:
    json.dump(stimulus_info, f)

# ============================================================
# Step 2: Extract LLaVA activations
# ============================================================
logger.info("=== Step 2: Extracting LLaVA activations ===")
from src.models.extract_activations import ActivationExtractor

extractor = ActivationExtractor(model_cfg, device="cuda", dtype="float16")
extractor.load_model()

extractor.extract_all(images=images, texts=texts,
                      output_dir=str(PROJECT_DIR / "activations"), pca_dim=256)

# Verify activations
act_dir = PROJECT_DIR / "activations" / model_cfg["name"].replace(" ", "_").replace("-", "_")
test_file = act_dir / "layer_0_all.npy"
if test_file.exists():
    d = np.load(test_file)
    logger.info(f"  Activation check: layer_0_all shape={d.shape}, std={d.std():.4f}")
else:
    logger.error("  NO ACTIVATION FILES GENERATED")

# ============================================================
# Step 3: Compute model RDMs
# ============================================================
logger.info("=== Step 3: Computing model RDMs ===")
model_name = model_cfg["name"].replace(" ", "_").replace("-", "_")
n_layers = model_cfg["n_layers"]
rdm_dir = PROJECT_DIR / "rdms"
rdm_dir.mkdir(parents=True, exist_ok=True)

for l in range(n_layers):
    for tt in ["vis", "txt", "all"]:
        f = act_dir / f"layer_{l}_{tt}.npy"
        if f.exists():
            data = np.load(f)
            if data.shape[0] > 1:
                rdm = squareform(pdist(data, metric="correlation"))
                np.save(rdm_dir / f"{model_name}_layer_{l}_{tt}_rdm.npy", rdm)
logger.info("  Model RDMs computed")

# ============================================================
# Step 4: RSA Analysis
# ============================================================
logger.info("=== Step 4: RSA Analysis ===")
brain_rdm_dir = PROJECT_DIR / "brain_rdms"
brain_rdm_dir.mkdir(parents=True, exist_ok=True)
results_dir = PROJECT_DIR / "results"
(results_dir / "rsa").mkdir(parents=True, exist_ok=True)

subjects = [f"sub-{i:02d}" for i in range(1, N_SUBJECTS + 1)]
for subj in subjects:
    for roi in roi_groups:
        vf = brain_dir / subj / f"{roi}_voxels.npy"
        if vf.exists():
            v = np.load(vf)
            np.save(brain_rdm_dir / f"{subj}_{roi}_rdm.npy", squareform(pdist(v, metric="correlation")))

rsa_matrix = np.zeros((n_layers, len(roi_groups)))
for l in range(n_layers):
    for ri, roi in enumerate(roi_groups):
        corrs = []
        model_rdm_f = rdm_dir / f"{model_name}_layer_{l}_all_rdm.npy"
        if not model_rdm_f.exists():
            continue
        model_rdm = np.load(model_rdm_f)
        for subj in subjects:
            brain_rdm_f = brain_rdm_dir / f"{subj}_{roi}_rdm.npy"
            if not brain_rdm_f.exists():
                continue
            brain_rdm = np.load(brain_rdm_f)
            n = min(model_rdm.shape[0], brain_rdm.shape[0])
            mu = model_rdm[:n, :n][np.triu_indices(n, k=1)]
            bu = brain_rdm[:n, :n][np.triu_indices(n, k=1)]
            if len(mu) > 10:
                rho, _ = spearmanr(mu, bu)
                if np.isfinite(rho):
                    corrs.append(rho)
        rsa_matrix[l, ri] = np.mean(corrs) if corrs else 0.0

np.savez(results_dir / "rsa" / f"rsa_{model_name}_all.npz",
         rsa_matrix=rsa_matrix, roi_names=roi_groups, n_layers=n_layers)
logger.info(f"  RSA matrix: shape={rsa_matrix.shape}, range=[{rsa_matrix.min():.4f}, {rsa_matrix.max():.4f}]")

# ============================================================
# Step 5: CKA Analysis
# ============================================================
logger.info("=== Step 5: CKA Analysis ===")
(results_dir / "cka").mkdir(parents=True, exist_ok=True)
from src.analysis.cka_analysis import linear_cka

cka_matrix = np.zeros((n_layers, len(roi_groups)))
for l in range(n_layers):
    af = act_dir / f"layer_{l}_all.npy"
    if not af.exists():
        continue
    ma = np.load(af)
    for ri, roi in enumerate(roi_groups):
        vals = []
        for subj in subjects:
            vf = brain_dir / subj / f"{roi}_voxels.npy"
            if vf.exists():
                ba = np.load(vf)
                n = min(ma.shape[0], ba.shape[0])
                c = linear_cka(ma[:n], ba[:n])
                if np.isfinite(c):
                    vals.append(c)
        cka_matrix[l, ri] = np.mean(vals) if vals else 0.0

np.savez(results_dir / "cka" / f"cka_{model_name}_all_linear.npz",
         cka_matrix=cka_matrix, roi_names=roi_groups, n_layers=n_layers)
logger.info(f"  CKA matrix: shape={cka_matrix.shape}, range=[{cka_matrix.min():.4f}, {cka_matrix.max():.4f}]")

# ============================================================
# Step 6: Triple Dissociation (Patching)
# ============================================================
logger.info("=== Step 6: Triple Dissociation ===")
from src.patching.activation_patching import ActivationPatcher
from difflib import SequenceMatcher

patcher = ActivationPatcher(extractor.model, extractor.processor, model_cfg)
if hasattr(extractor, '_vision_tower'):
    patcher.set_vision_components(extractor._vision_tower, extractor._image_processor, extractor._mm_projector)

N_PATCH = min(20, len(images))
patch_images = images[:N_PATCH]

# Compute mean activations
logger.info("  Computing mean activations...")
patcher.compute_mean_activations(patch_images, texts[:N_PATCH], n_samples=min(15, N_PATCH))

task_prompts = {
    "visual_recognition": "What is the main object in this image? Answer briefly.",
    "cross_modal_binding": "What color is the largest object in this image?",
    "language_generation": "Describe this image in detail.",
}
stage_ranges = {s: tuple(STAGES[s]) for s in ["visual", "fusion", "language"]}
stage_names = ["visual", "fusion", "language"]
task_names = list(task_prompts.keys())

# Baseline
logger.info("  Running baseline...")
baseline = {}
for task_name, prompt in task_prompts.items():
    responses = []
    for img_path in patch_images:
        try:
            from PIL import Image as PILImage
            img = PILImage.open(img_path).convert("RGB")
            inp = patcher._prepare_inputs(img, prompt)
            with torch.no_grad():
                out = patcher._run_generate(inp, max_new_tokens=64)
            responses.append(out["text"])
        except Exception as e:
            responses.append(str(e)[:100])
    baseline[task_name] = responses
    logger.info(f"    Baseline {task_name}: {len(responses)} responses, sample='{responses[0][:60]}...'")

# Patching
diss_matrix = np.zeros((3, 3))
(results_dir / "dissociation").mkdir(parents=True, exist_ok=True)

for si, stage in enumerate(stage_names):
    lr = stage_ranges[stage]
    logger.info(f"  Patching stage: {stage} (layers {lr[0]}-{lr[1]})")
    for ti, task_name in enumerate(task_names):
        prompt = task_prompts[task_name]
        patched = []
        for img_path in patch_images:
            try:
                result = patcher.patch_and_generate(
                    image=img_path, text=prompt,
                    layer_range=lr, patch_type="mean", max_new_tokens=64)
                patched.append(result["text"])
            except Exception as e:
                patched.append("")
        sims = [SequenceMatcher(None, b.lower(), p.lower()).ratio()
                for b, p in zip(baseline[task_name], patched)]
        drop = 1.0 - np.mean(sims)
        diss_matrix[si, ti] = drop
        logger.info(f"    {stage}/{task_name}: drop={drop:.3f}")

np.save(results_dir / "dissociation" / "dissociation_matrix.npy", diss_matrix)

logger.info("\n=== Triple Dissociation Matrix ===")
header = f"{'':>20}" + "".join(f"{t:>25}" for t in task_names)
logger.info(header)
for si, s in enumerate(stage_names):
    row = f"{s:>20}" + "".join(
        f"{diss_matrix[si, ti]:>20.3f}{'  <<<' if si == ti else '     '}"
        for ti in range(3))
    logger.info(row)

# ============================================================
# Step 7: Generate Figures
# ============================================================
logger.info("\n=== Step 7: Generate Figures ===")
import matplotlib
matplotlib.use("Agg")
from src.visualization.heatmaps import plot_alignment_heatmap, plot_dissociation_matrix

fig_dir = results_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)
boundaries = [STAGES["fusion"][0], STAGES["language"][0]]

rsa_f = results_dir / "rsa" / f"rsa_{model_name}_all.npz"
if rsa_f.exists():
    d = np.load(rsa_f, allow_pickle=True)
    plot_alignment_heatmap(d["rsa_matrix"], list(d["roi_names"]), model_cfg["name"],
                           "RSA", boundaries, str(fig_dir / "rsa_heatmap.pdf"))
    logger.info("  RSA heatmap saved")

cka_f = results_dir / "cka" / f"cka_{model_name}_all_linear.npz"
if cka_f.exists():
    d = np.load(cka_f, allow_pickle=True)
    plot_alignment_heatmap(d["cka_matrix"], list(d["roi_names"]), model_cfg["name"],
                           "CKA", boundaries, str(fig_dir / "cka_heatmap.pdf"))
    logger.info("  CKA heatmap saved")

diss_f = results_dir / "dissociation" / "dissociation_matrix.npy"
if diss_f.exists():
    plot_dissociation_matrix(np.load(diss_f), str(fig_dir / "dissociation.pdf"))
    logger.info("  Dissociation matrix saved")

# Cleanup
extractor.cleanup()
logger.info("\n=== EXPERIMENT COMPLETE ===")
logger.info(f"Results: {results_dir}")
logger.info(f"Figures: {fig_dir}")
