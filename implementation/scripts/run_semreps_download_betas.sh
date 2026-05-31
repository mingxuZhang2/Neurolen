#!/usr/bin/env bash
# Download SemReps-8K (ds007272) surface betas from OpenNeuro S3 to HPC2.
#
# Each subject has DISJOINT train stimuli, so a per-subject encoder needs that
# subject's own train betas. sub-01/02/03 LH train betas were fetched earlier;
# this pulls additional subjects/conditions/hemispheres on demand.
#
# Usage:
#   SR_SUBS="sub-04 sub-05 sub-07" SR_HEMI=left SR_CONDS="betas_train_image betas_train_caption" \
#     bash run_semreps_download_betas.sh
#
# Env (all optional, defaults shown):
#   SR_SUBS  = "sub-04 sub-05 sub-07"
#   SR_HEMI  = "left"                       # left | right
#   SR_CONDS = "betas_train_image betas_train_caption"
set -uo pipefail

DEST=/hpc2hdd/home/mzhang630/data/semreps/surface
S3=s3://openneuro.org/ds007272/derivatives/betas/surface
SUBS=${SR_SUBS:-"sub-04 sub-05 sub-07"}
HEMI=${SR_HEMI:-left}
CONDS=${SR_CONDS:-"betas_train_image betas_train_caption"}

export AWS_MAX_ATTEMPTS=10
# bump parallelism for many small files
aws configure set default.s3.max_concurrent_requests 24 2>/dev/null || true

echo "=== SemReps beta download :: $(date) ==="
echo "subs=[$SUBS] hemi=$HEMI conds=[$CONDS]"

for s in $SUBS; do
  for c in $CONDS; do
    src="$S3/$HEMI/$s/$c/"
    dst="$DEST/$HEMI/$s/$c/"
    mkdir -p "$dst"
    echo "--- sync $s/$c ($HEMI) ---"
    aws s3 sync --no-sign-request "$src" "$dst" --only-show-errors
    n=$(ls "$dst"/*.gii 2>/dev/null | wc -l)
    echo "    $s/$c [$HEMI]: $n .gii files"
  done
done

echo "=== done :: $(date) ==="
