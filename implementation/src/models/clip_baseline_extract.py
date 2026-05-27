"""Extract CLIP-ViT patch tokens and MLP projector outputs as baselines.

Produces two "pseudo-layers" to compare against LLaMA decoder layers:
  clip_patches  — CLIP-ViT hidden_states[-2][:, 1:, :] (576 tokens, 1024-d)
  projector_out — MLP projector output (576 tokens, 4096-d)

These are the two stages BEFORE the image tokens enter LLaMA. By comparing
their brain encoding power to LLaMA L0..L31, we can isolate what the
language decoder adds vs what CLIP already provides.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

logger = logging.getLogger(__name__)


def extract_clip_baselines(
    llava_dir: str,
    image_paths: list[str],
    output_dir: str,
    pca_dim: int = 256,
):
    """Extract CLIP patch tokens and projector outputs for a list of images.

    Saves:
      clip_patches_vis_tokens.npy     (n_img, 576, pca_dim)
      projector_out_vis_tokens.npy    (n_img, 576, pca_dim)
    """
    from transformers import CLIPVisionModel, CLIPImageProcessor

    llava_dir = Path(llava_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load config to find CLIP path
    with open(llava_dir / "config.json") as f:
        cfg = json.load(f)
    clip_path = cfg.get("mm_vision_tower", "")
    logger.info(f"Loading CLIP from {clip_path}")

    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vision_tower = CLIPVisionModel.from_pretrained(clip_path, torch_dtype=dtype).to(device)
    vision_tower.eval()
    image_processor = CLIPImageProcessor.from_pretrained(clip_path)

    # Load projector
    mm_hidden_size = cfg.get("mm_hidden_size", 1024)
    hidden_size = cfg.get("hidden_size", 4096)
    projector = torch.nn.Sequential(
        torch.nn.Linear(mm_hidden_size, hidden_size),
        torch.nn.GELU(),
        torch.nn.Linear(hidden_size, hidden_size),
    ).to(device, dtype=dtype)

    # Load projector weights
    import os
    proj_weights = {}
    mm_proj_path = llava_dir / "mm_projector.bin"
    if mm_proj_path.exists():
        proj_weights = torch.load(mm_proj_path, map_location="cpu", weights_only=True)
    else:
        idx_path = llava_dir / "pytorch_model.bin.index.json"
        with open(idx_path) as f:
            idx = json.load(f)
        for shard_file in sorted(set(idx["weight_map"].values())):
            shard = torch.load(llava_dir / shard_file, map_location="cpu", weights_only=True)
            for k, v in shard.items():
                if "mm_projector" in k or "multi_modal_projector" in k:
                    proj_weights[k] = v
            del shard

    proj_state = {}
    for k, v in proj_weights.items():
        clean = k
        for prefix in ("model.mm_projector.", "model.multi_modal_projector."):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
                break
        if clean.startswith("linear_1"):
            clean = clean.replace("linear_1", "0")
        if clean.startswith("linear_2"):
            clean = clean.replace("linear_2", "2")
        proj_state[clean] = v
    projector.load_state_dict(proj_state)
    projector.eval()
    logger.info(f"Loaded projector ({len(proj_state)} tensors)")

    n_images = len(image_paths)
    clip_buf = []
    proj_buf = []

    for idx in tqdm(range(n_images), desc="CLIP baseline extraction"):
        img = Image.open(image_paths[idx]).convert("RGB")
        pixel_values = image_processor(images=img, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(device, dtype=dtype)

        with torch.no_grad():
            vision_out = vision_tower(pixel_values, output_hidden_states=True)
            # CLIP patch tokens: hidden_states[-2], skip CLS token
            clip_tokens = vision_out.hidden_states[-2][:, 1:, :]  # (1, 576, 1024)
            proj_tokens = projector(clip_tokens)  # (1, 576, 4096)

        clip_buf.append(clip_tokens[0].cpu().float().numpy())
        proj_buf.append(proj_tokens[0].cpu().float().numpy())

        if (idx + 1) % 100 == 0:
            torch.cuda.empty_cache()

    clip_arr = np.stack(clip_buf, axis=0)  # (n_img, 576, 1024)
    proj_arr = np.stack(proj_buf, axis=0)  # (n_img, 576, 4096)

    logger.info(f"CLIP patches: {clip_arr.shape}, Projector out: {proj_arr.shape}")

    # PCA compress each
    for name, arr in [("clip_patches", clip_arr), ("projector_out", proj_arr)]:
        n_img, n_tok, hd = arr.shape
        if pca_dim > 0 and hd > pca_dim:
            flat = arr.reshape(-1, hd)
            fit_n = min(60000, flat.shape[0])
            rng = np.random.default_rng(42)
            fit_idx = rng.choice(flat.shape[0], size=fit_n, replace=False)
            pca = PCA(n_components=pca_dim, random_state=42, svd_solver="randomized")
            pca.fit(flat[fit_idx])
            reduced = pca.transform(flat).astype(np.float16)
            arr_save = reduced.reshape(n_img, n_tok, pca_dim)
            expl = float(pca.explained_variance_ratio_.sum())
            logger.info(f"  {name}: PCA {hd}->{pca_dim}, explained={expl:.3f}")
        else:
            arr_save = arr.astype(np.float16)
        np.save(out / f"{name}_vis_tokens.npy", arr_save)

    logger.info(f"Saved CLIP baselines to {out}")
    del vision_tower, projector
    torch.cuda.empty_cache()
