"""Sequence-position dissociation: extract image, prompt, and generated token hidden states.

For each stimulus, runs LLaVA generation and captures hidden states at three
sequence positions:
  1. Image tokens (positions 0:576)
  2. Prompt tokens (positions 576:576+n_prompt)
  3. Generated tokens (new positions produced during generation)

This tests WHERE in the sequence vision-to-language conversion occurs:
  - Image tokens → expected to encode visual cortex (V1-V4)
  - Prompt tokens → expected to carry moderate visual + semantic signal
  - Generated tokens → expected to carry strongest semantic/language signal
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.models.extract_activations import ActivationExtractor

logger = logging.getLogger(__name__)

LLAVA_VIS_TOKENS = 576


class SequencePositionExtractor(ActivationExtractor):
    """Extract hidden states at image, prompt, and generated token positions.

    Uses generation (not just forward pass) to produce actual answer tokens,
    then captures the last-layer hidden states at each position type.
    For brain encoding, we save mean-pooled activations per position stream
    at representative layers.
    """

    def __init__(self, model_config: dict, device: str = "cuda",
                 dtype: str = "float16", max_new_tokens: int = 30):
        super().__init__(model_config, device=device, dtype=dtype)
        self.max_new_tokens = max_new_tokens
        self._layer_acts: dict[int, torch.Tensor] = {}

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._layer_acts[layer_idx] = hidden.detach()
        return hook_fn

    def _register_hooks(self):
        self._remove_hooks()
        self._layer_acts.clear()
        if getattr(self, "_is_manual_llava", False):
            layers = self.model.model.layers
        else:
            prefix = self.config["activation_hook_prefix"]
            module = self.model
            for attr in prefix.split("."):
                module = getattr(module, attr)
            layers = module
        for i, layer in enumerate(layers):
            h = layer.register_forward_hook(self._make_hook(i))
            self._hooks.append(h)

    def extract_sequence_positions(
        self, image: Image.Image, text: str = "Describe this image in detail.",
        rep_layers: list[int] = None,
    ) -> dict:
        """Extract per-position-stream hidden states for one stimulus.

        Returns dict of layer -> {
            "image_mean": (hidden_dim,) mean over 576 image tokens,
            "prompt_mean": (hidden_dim,) mean over prompt tokens,
            "all_mean": (hidden_dim,) mean over image+prompt (prefill),
        }

        Note: generated token states require a separate generation pass.
        For efficiency, we do two passes:
          1. Prefill pass → captures image + prompt token states at all layers
          2. Generation pass → captures final generated token states
        """
        assert getattr(self, "_is_manual_llava", False)

        inputs = self._prepare_input_llava(image, text)
        n_vis = inputs.get("_n_visual_tokens", LLAVA_VIS_TOKENS)
        inputs_embeds = inputs["inputs_embeds"]
        seq_len = inputs_embeds.shape[1]
        n_prompt = seq_len - n_vis

        # Pass 1: prefill (captures image + prompt positions)
        self._layer_acts.clear()
        self._register_hooks()
        with torch.no_grad():
            outputs = self.model(inputs_embeds=inputs_embeds)

        n_layers = self.config["n_layers"]
        if rep_layers is None:
            rep_layers = list(range(n_layers))

        result = {}
        for L in rep_layers:
            if L not in self._layer_acts:
                continue
            h = self._layer_acts[L]
            if h.dim() == 3:
                h = h[0]  # (seq_len, hidden_dim)

            img_h = h[:n_vis]       # image tokens
            prm_h = h[n_vis:]       # prompt tokens

            result[L] = {
                "image_mean": img_h.mean(dim=0).cpu().to(torch.float16).numpy(),
                "prompt_mean": prm_h.mean(dim=0).cpu().to(torch.float16).numpy(),
                "all_mean": h.mean(dim=0).cpu().to(torch.float16).numpy(),
            }

        # Pass 2: generation (captures generated token states)
        self._layer_acts.clear()
        with torch.no_grad():
            gen_out = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        # gen_out.hidden_states is a tuple of (n_gen_steps,) each being
        # a tuple of (n_layers+1,) tensors of shape (batch, 1, hidden_dim).
        # We want the mean hidden state across generated tokens per layer.
        if hasattr(gen_out, "hidden_states") and gen_out.hidden_states is not None:
            n_gen_steps = len(gen_out.hidden_states)
            for L in rep_layers:
                if L not in result:
                    result[L] = {}
                gen_states = []
                for step in range(n_gen_steps):
                    step_hidden = gen_out.hidden_states[step]
                    # step_hidden is tuple of (n_layers+1,) tensors
                    # Index L+1 because index 0 is embedding output
                    if len(step_hidden) > L + 1:
                        h_step = step_hidden[L + 1]  # (1, 1, hidden_dim)
                        gen_states.append(h_step[0, 0])
                if gen_states:
                    gen_h = torch.stack(gen_states, dim=0)  # (n_gen, hidden_dim)
                    result[L]["generated_mean"] = gen_h.mean(dim=0).cpu().to(torch.float16).numpy()
                else:
                    result[L]["generated_mean"] = np.zeros(self.config["hidden_dim"], dtype=np.float16)
        else:
            for L in rep_layers:
                if L in result:
                    result[L]["generated_mean"] = np.zeros(self.config["hidden_dim"], dtype=np.float16)

        self._remove_hooks()
        return result

    def extract_all_sequence_positions(
        self,
        images: list,
        texts: Optional[list],
        output_dir: str,
        rep_layers: list[int] = None,
    ):
        """Extract sequence-position hidden states for all stimuli.

        Saves per-layer:
          layer_{L}_image_mean.npy     (n_img, hidden_dim)
          layer_{L}_prompt_mean.npy    (n_img, hidden_dim)
          layer_{L}_generated_mean.npy (n_img, hidden_dim)
        """
        n_images = len(images)
        if texts is None:
            texts = ["Describe this image in detail."] * n_images

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        n_layers = self.config["n_layers"]
        hidden_dim = self.config["hidden_dim"]
        if rep_layers is None:
            rep_layers = [0, 4, 8, 12, 14, 16, 20, 24, 28, 31]

        # Pre-allocate
        streams = ["image_mean", "prompt_mean", "generated_mean"]
        bufs = {
            (L, s): np.zeros((n_images, hidden_dim), dtype=np.float16)
            for L in rep_layers for s in streams
        }

        self._register_hooks()

        for idx in tqdm(range(n_images), desc="Sequence-position extraction"):
            img = images[idx]
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")

            try:
                per_layer = self.extract_sequence_positions(
                    img, texts[idx], rep_layers=rep_layers)
                for L, d in per_layer.items():
                    for s in streams:
                        if s in d:
                            bufs[(L, s)][idx] = d[s]
            except Exception as e:
                logger.warning(f"Stimulus {idx} failed: {e}")

            if (idx + 1) % 20 == 0:
                torch.cuda.empty_cache()

        self._remove_hooks()

        # Save
        for L in rep_layers:
            for s in streams:
                arr = bufs[(L, s)]
                np.save(out / f"layer_{L}_{s}.npy", arr)

        # Save metadata
        meta = {
            "n_images": n_images,
            "rep_layers": rep_layers,
            "streams": streams,
            "hidden_dim": hidden_dim,
            "max_new_tokens": self.max_new_tokens,
        }
        with open(out / "sequence_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved sequence-position activations to {out}")
