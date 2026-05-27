#!/bin/bash
#SBATCH --job-name=noise_ceil
#SBATCH --partition=acd_u
#SBATCH --account=d_yings_team
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/data/user/mzhang630/data/mllm/logs/noise_ceil_%j.log

# Compute noise ceiling per voxel from NSD ncsnr, then normalize
# the existing spatial encoding mean_r.

export PROJECT_DIR=/data/user/mzhang630/data/mllm
export PYTHONPATH=$PROJECT_DIR/implementation:$PYTHONPATH
PY=/data/user/mzhang630/miniconda3/envs/alphasteer/bin/python

echo "Node: $(hostname)  Date: $(date)"

$PY -c "
import sys, json, logging, numpy as np
sys.path.insert(0, '$PROJECT_DIR/implementation')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

from src.data.nsd_fsaverage import NSDFsaverageLoader, get_roi_voxels
from pathlib import Path

loader = NSDFsaverageLoader('$PROJECT_DIR/../nsd')
visual_rois = ['V1', 'V2', 'V3', 'V4']
language_rois = ['Broca', 'IFG_extended', 'auditory_assoc', 'temporal_pole']
all_rois = visual_rois + language_rois

# Compute noise ceiling per ROI from ncsnr
# NSD ncsnr = signal std / noise std per voxel
# Noise ceiling for Pearson r with n_reps averaged:
#   NC_r = ncsnr / sqrt(ncsnr^2 + 1/n_reps)
# For shared1000 with ~3 reps averaged (but we have partial coverage):
n_reps = 3  # NSD design: each shared1000 image shown 3x

out_dir = Path('$PROJECT_DIR/results_v2/statistical_tests')
out_dir.mkdir(parents=True, exist_ok=True)

nc_per_roi = {}
for roi in all_rois:
    try:
        voxels, brain_ids, ncsnr = get_roi_voxels(loader, 0, roi)
    except Exception as e:
        print(f'ROI {roi} failed: {e}')
        continue
    if ncsnr.shape[0] == 0:
        continue
    # Noise ceiling: expected max correlation
    nc_r = ncsnr / np.sqrt(ncsnr**2 + 1.0/n_reps)
    nc_per_roi[roi] = {
        'nc_mean': float(nc_r.mean()),
        'nc_median': float(np.median(nc_r)),
        'nc_std': float(nc_r.std()),
        'n_voxels': int(ncsnr.shape[0]),
        'ncsnr_mean': float(ncsnr.mean()),
    }
    print(f'{roi:>20s}: NC_r mean={nc_r.mean():.4f} median={np.median(nc_r):.4f} '
          f'ncsnr mean={ncsnr.mean():.3f} n_vox={ncsnr.shape[0]}')

# Now normalize spatial encoding
spat = np.load('$PROJECT_DIR/results_v2/token_spatial_real/subj01/spatial_encoding.npz', allow_pickle=True)
mean_r = spat['mean_r']  # (576, 32, 4)
rois = list(spat['roi_names'])
layers = spat['layer_indices']

print()
print('=== NC-normalized mean_r (mean_r / NC_r_mean) ===')
print(f'{\"Layer\":>6s}', end='')
for roi in rois:
    print(f'{roi:>10s}', end='')
print()

nc_norm_r = np.zeros_like(mean_r)
for ri, roi in enumerate(rois):
    if roi in nc_per_roi:
        nc = nc_per_roi[roi]['nc_mean']
        nc_norm_r[:, :, ri] = mean_r[:, :, ri] / nc if nc > 0 else 0
    else:
        nc_norm_r[:, :, ri] = mean_r[:, :, ri]

for li in [0, 7, 14, 21, 31]:
    print(f'L{li:>4d}', end='')
    for ri in range(len(rois)):
        print(f'{nc_norm_r[:, li, ri].mean():>10.4f}', end='')
    print()

np.savez(
    out_dir / 'noise_ceiling.npz',
    roi_names=np.array(list(nc_per_roi.keys())),
    nc_mean=np.array([v['nc_mean'] for v in nc_per_roi.values()]),
    nc_median=np.array([v['nc_median'] for v in nc_per_roi.values()]),
    n_voxels=np.array([v['n_voxels'] for v in nc_per_roi.values()]),
    nc_norm_spatial=nc_norm_r,
    spatial_layers=layers,
    spatial_rois=np.array(rois),
)
print(f'Saved to {out_dir}/noise_ceiling.npz')
"

echo "Done: $(date)"
