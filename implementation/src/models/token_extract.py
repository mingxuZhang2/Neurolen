"""Token-level MLLM activation + logit-lens extraction.

Unlike `extract_activations.py` (which mean-pools the 576 vision tokens into one
vector per layer), this module preserves per-token activations across the
LLaVA-1.5 layer stack and additionally runs *logit lens* — projecting each
layer's hidden state through the final RMSNorm + LM head to recover what word
each token "would predict" at that depth.

Outputs per stimulus (LLaVA-1.5-7B, n_layers=32, n_vis_tokens=576):
  - vision_acts[L][i] : (n_images, 576, hidden_dim) hidden state of vision token i at layer L
  - logit_lens_topk[L][i] : (n_images, 576, K) top-K token ids predicted at (L, i)
  - logit_lens_topp[L][i] : (n_images, 576, K) corresponding probabilities

Storage strategy:
  Raw fp16 hidden states across all 32 layers ≈ 42 GB / 1000 images.
  We PCA-compress *per layer* to 256 dims (shared basis across all 576 tokens),
  bringing total to ~9 GB for n_layers=32, or ~2.6 GB for 9 representative layers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

from src.models.extract_activations import ActivationExtractor

logger = logging.getLogger(__name__)


LLAVA_VIS_TOKENS = 576  # CLIP-ViT-L/14-336 with 24×24 patches


@dataclass
class TokenLevelStore:
    """Per-stimulus accumulator using disk-backed memmaps.

    Holding 32 layers × 1000 imgs × 576 tokens × 4096 dim fp16 in RAM is ~150 GB.
    Instead we open one fp16 memmap per layer on disk and write each image's
    hidden state directly. Logit-lens arrays are tiny (1000 × 576 × 10) so we
    keep them in RAM.

    Saves three arrays per layer:
      layer_{L}_vis_tokens.npy       (n_images, n_tokens, pca_dim) — PCA-compressed
      layer_{L}_logit_topk_ids.npy   (n_images, n_tokens, K) int32
      layer_{L}_logit_topk_probs.npy (n_images, n_tokens, K) fp16
    """
    output_dir: Path
    n_layers: int
    n_images: int
    n_vis_tokens: int = LLAVA_VIS_TOKENS
    hidden_dim: int = 4096
    pca_dim: int = 256
    top_k: int = 10

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Temp dir for raw memmaps (deleted after PCA)
        self.tmp_dir = self.output_dir / "_tmp_raw"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

        # Per-layer memmap for raw activations (disk-backed)
        self._act_mmap: dict[int, np.memmap] = {}
        for L in range(self.n_layers):
            path = self.tmp_dir / f"layer_{L}_raw.dat"
            self._act_mmap[L] = np.memmap(
                path, dtype=np.float16, mode="w+",
                shape=(self.n_images, self.n_vis_tokens, self.hidden_dim),
            )

        # Logit lens stays in RAM (small)
        self._ids_buf: dict[int, np.ndarray] = {
            L: np.zeros((self.n_images, self.n_vis_tokens, self.top_k), dtype=np.int32)
            for L in range(self.n_layers)
        }
        self._prob_buf: dict[int, np.ndarray] = {
            L: np.zeros((self.n_images, self.n_vis_tokens, self.top_k), dtype=np.float16)
            for L in range(self.n_layers)
        }

    def append(
        self,
        image_idx: int,
        layer_idx: int,
        vis_act: np.ndarray,          # (n_tokens, hidden_dim) fp16
        logit_ids: np.ndarray,        # (n_tokens, K) int32
        logit_probs: np.ndarray,      # (n_tokens, K) fp16
    ):
        self._act_mmap[layer_idx][image_idx] = vis_act
        self._ids_buf[layer_idx][image_idx] = logit_ids
        self._prob_buf[layer_idx][image_idx] = logit_probs

    def finalize_and_save(self, save_raw: bool = False):
        """For each layer, load raw memmap, PCA-compress, save final outputs."""
        import shutil

        for L in tqdm(range(self.n_layers), desc="Finalizing per-layer arrays"):
            # Flush memmap to disk
            self._act_mmap[L].flush()
            # Reopen read-only and load into RAM (4.7 GB per layer fp16)
            mmap_path = self.tmp_dir / f"layer_{L}_raw.dat"
            acts = np.array(
                np.memmap(mmap_path, dtype=np.float16, mode="r",
                          shape=(self.n_images, self.n_vis_tokens, self.hidden_dim))
            )
            ids = self._ids_buf[L]
            probs = self._prob_buf[L]

            n_img, n_tok, hd = acts.shape
            logger.info(f"Layer {L}: acts {acts.shape}, ids {ids.shape}")

            if save_raw:
                np.save(self.output_dir / f"layer_{L}_vis_tokens_raw.npy", acts)

            # PCA on subsample then transform all
            if self.pca_dim and self.pca_dim > 0 and hd > self.pca_dim:
                flat = acts.reshape(-1, hd).astype(np.float32)
                n_total = flat.shape[0]
                fit_n = min(60000, n_total)
                if n_total > fit_n:
                    rng = np.random.default_rng(42 + L)
                    fit_idx = rng.choice(n_total, size=fit_n, replace=False)
                    fit_data = flat[fit_idx]
                else:
                    fit_data = flat
                from sklearn.decomposition import PCA as _PCA
                pca = _PCA(n_components=self.pca_dim, random_state=42,
                           svd_solver="randomized")
                pca.fit(fit_data)
                reduced = pca.transform(flat).astype(np.float16)
                acts_to_save = reduced.reshape(n_img, n_tok, self.pca_dim)
                expl = float(pca.explained_variance_ratio_.sum())
                logger.info(f"  PCA {hd}->{self.pca_dim} (fit on {fit_n}), explained={expl:.3f}")
                del flat, fit_data, reduced
            else:
                acts_to_save = acts

            np.save(self.output_dir / f"layer_{L}_vis_tokens.npy", acts_to_save)
            np.save(self.output_dir / f"layer_{L}_logit_topk_ids.npy", ids)
            np.save(self.output_dir / f"layer_{L}_logit_topk_probs.npy", probs)

            # Free per-layer memory
            del acts, acts_to_save
            self._act_mmap[L] = None
            # Delete the raw memmap file
            try:
                mmap_path.unlink()
            except FileNotFoundError:
                pass

        # Clean up tmp dir
        try:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
        except Exception:
            pass

        logger.info(f"Saved all token-level outputs to {self.output_dir}")


class TokenLevelExtractor(ActivationExtractor):
    """Extends ActivationExtractor to preserve per-token activations + logit lens.

    Reuses the existing manual-LLaVA loader from `ActivationExtractor._load_llava`
    (we depend on `self.model`, `self._vision_tower`, `self._mm_projector`,
    `self._image_processor`, `self.processor`).
    """

    def __init__(self, model_config: dict, device: str = "cuda",
                 dtype: str = "float16", top_k: int = 10):
        super().__init__(model_config, device=device, dtype=dtype)
        self.top_k = top_k
        self._token_acts: dict[int, torch.Tensor] = {}
        self._n_vis_tokens_actual: Optional[int] = None

    def _make_token_hook(self, layer_idx: int):
        """Hook that stores the *full sequence* hidden state (not pooled)."""
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # hidden: (1, seq_len, hidden_dim)
            self._token_acts[layer_idx] = hidden.detach()
        return hook_fn

    def _register_token_hooks(self):
        self._remove_hooks()
        self._token_acts.clear()

        if getattr(self, "_is_manual_llava", False):
            layers = self.model.model.layers
        else:
            prefix = self.config["activation_hook_prefix"]
            module = self.model
            for attr in prefix.split("."):
                module = getattr(module, attr)
            layers = module

        for i, layer in enumerate(layers):
            h = layer.register_forward_hook(self._make_token_hook(i))
            self._hooks.append(h)
        logger.info(f"Registered token-level hooks on {len(self._hooks)} layers")

    def _apply_logit_lens(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """LLaMA logit lens: final RMSNorm + LM head, returns top-K ids + probs.

        Args:
            hidden_states: (n_tokens, hidden_dim)
        Returns:
            (top_k_ids, top_k_probs) each (n_tokens, K)
        """
        # LLaMA's final norm is model.model.norm (RMSNorm)
        if hasattr(self.model.model, "norm"):
            normed = self.model.model.norm(hidden_states)
        else:
            normed = hidden_states  # fallback

        # LM head: (vocab_size, hidden_dim)
        logits = self.model.lm_head(normed)  # (n_tokens, vocab)
        probs = torch.softmax(logits.float(), dim=-1)
        topk_p, topk_ids = torch.topk(probs, k=self.top_k, dim=-1)
        return topk_ids, topk_p

    def extract_token_level_for_stimulus(
        self, image: Image.Image, text: str = "Describe this image in detail.",
    ) -> dict:
        """Extract per-token activations + logit lens for one image-text pair.

        Returns dict mapping layer_idx -> {
            "vis_act": (n_vis_tokens, hidden_dim) fp16,
            "logit_ids": (n_vis_tokens, K) int32,
            "logit_probs": (n_vis_tokens, K) fp16,
        }
        """
        assert getattr(self, "_is_manual_llava", False), \
            "TokenLevelExtractor currently only supports manual LLaVA path"

        inputs = self._prepare_input_llava(image, text)
        n_vis = inputs.get("_n_visual_tokens", LLAVA_VIS_TOKENS)
        self._n_vis_tokens_actual = n_vis
        self._token_acts.clear()

        with torch.no_grad():
            self.model(inputs_embeds=inputs["inputs_embeds"])

        result = {}
        for L, hidden in self._token_acts.items():
            if hidden.dim() == 3:
                hidden = hidden[0]  # drop batch
            vis_hidden = hidden[:n_vis]  # (n_vis, hidden_dim)

            with torch.no_grad():
                topk_ids, topk_probs = self._apply_logit_lens(vis_hidden)

            result[L] = {
                "vis_act": vis_hidden.cpu().to(torch.float16).numpy(),
                "logit_ids": topk_ids.cpu().to(torch.int32).numpy(),
                "logit_probs": topk_probs.cpu().to(torch.float16).numpy(),
            }
        return result

    def extract_token_level_all(
        self,
        images: list,
        texts: Optional[list],
        output_dir: str,
        pca_dim: int = 256,
        save_raw: bool = False,
        log_every: int = 50,
    ) -> dict:
        """Run token-level extraction on a list of images. Saves per-layer files.

        Args:
            images: list of PIL Images or file paths
            texts: optional list of text prompts (defaults to neutral prompt)
            output_dir: output directory (model subdir is created)
            pca_dim: PCA reduction; 0 to keep raw 4096-d
            save_raw: also save uncompressed fp16 arrays (large)
            log_every: cleanup CUDA cache every N stimuli
        Returns:
            dict with vocab tokenizer and image ordering for later decoding
        """
        n_images = len(images)
        if texts is None:
            texts = ["Describe this image in detail."] * n_images
        assert len(texts) == n_images

        out = Path(output_dir) / self.config["name"].replace(" ", "_").replace("-", "_")
        out.mkdir(parents=True, exist_ok=True)

        store = TokenLevelStore(
            output_dir=out,
            n_layers=self.config["n_layers"],
            n_images=n_images,
            n_vis_tokens=LLAVA_VIS_TOKENS,
            hidden_dim=self.config["hidden_dim"],
            pca_dim=pca_dim,
            top_k=self.top_k,
        )

        self._register_token_hooks()

        for idx in tqdm(range(n_images), desc="Token-level extraction"):
            img = images[idx]
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
            txt = texts[idx]

            try:
                per_layer = self.extract_token_level_for_stimulus(img, txt)
                for L, d in per_layer.items():
                    store.append(idx, L, d["vis_act"], d["logit_ids"], d["logit_probs"])
            except Exception as e:
                logger.warning(f"Stimulus {idx} failed: {e}")
                # Zeros already initialized; nothing to do

            if (idx + 1) % log_every == 0:
                torch.cuda.empty_cache()

        self._remove_hooks()
        store.finalize_and_save(save_raw=save_raw)

        # Save vocab for later word lookups
        vocab = self.processor.get_vocab()
        inv_vocab = {v: k for k, v in vocab.items()}
        import json
        with open(out / "vocab_inv.json", "w") as f:
            # Save only token id -> string mapping (compact)
            json.dump({str(k): v for k, v in inv_vocab.items()}, f)

        # Save metadata
        meta = {
            "n_images": n_images,
            "n_layers": self.config["n_layers"],
            "n_vis_tokens": LLAVA_VIS_TOKENS,
            "vis_grid_h": 24,
            "vis_grid_w": 24,
            "hidden_dim": self.config["hidden_dim"],
            "pca_dim": pca_dim,
            "top_k": self.top_k,
        }
        with open(out / "token_extract_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Token-level extraction done: {out}")
        return {"output_dir": out, "meta": meta}
