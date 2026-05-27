"""Stage-specific activation patching for causal validation.

Implements mean ablation (primary) and zero ablation (supplementary)
for specific layer ranges. Uses full-sequence patching per Nooralahzadeh (2026).
"""

import logging
from typing import Optional

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ActivationPatcher:
    """Patch activations at specific MLLM processing stages.

    Supports mean ablation (replace with dataset mean), zero ablation,
    and noise ablation. Full-sequence patching (all token positions).
    """

    def __init__(self, model, processor, model_config: dict,
                 device: str = "cuda"):
        self.model = model
        self.processor = processor
        self.config = model_config
        self.device = device
        self._hooks = []
        self._mean_activations = {}  # layer -> mean activation tensor

    def set_vision_components(self, vision_tower, image_processor, mm_projector):
        """Set vision components for manual LLaVA mode."""
        self._vision_tower = vision_tower
        self._image_processor = image_processor
        self._mm_projector = mm_projector

    def _prepare_inputs_manual_llava(self, image: Image.Image, text: str) -> dict:
        """Prepare inputs for manual LLaVA assembly."""
        pixel_values = self._image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"].to(self.model.device, dtype=next(self.model.parameters()).dtype)

        with torch.no_grad():
            vision_out = self._vision_tower(pixel_values, output_hidden_states=True)
            image_features = vision_out.hidden_states[-2][:, 1:, :]
            image_features = self._mm_projector(image_features)

        prompt = f"USER: \n{text}\nASSISTANT:"
        text_ids = self.processor(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        text_embeds = self.model.get_input_embeddings()(text_ids)
        inputs_embeds = torch.cat([image_features, text_embeds], dim=1)
        return {"inputs_embeds": inputs_embeds, "_is_manual": True}

    def _get_layer_prefix(self):
        """Get the correct layer prefix for hook registration."""
        prefix = self.config.get("activation_hook_prefix", "model.layers")
        # Try the configured prefix first; fall back to model.layers for LlamaForCausalLM
        module = self.model
        try:
            for attr in prefix.split("."):
                module = getattr(module, attr)
            return prefix
        except AttributeError:
            return "model.layers"

    def compute_mean_activations(
        self,
        images: list,
        texts: list,
        n_samples: int = 100,
    ):
        """Compute mean activations per layer from a set of inputs.

        These means are used for mean ablation (replacing activations
        with the dataset average).

        Args:
            images: list of PIL Images or paths
            texts: list of text prompts
            n_samples: how many samples to use for computing the mean
        """
        logger.info(f"Computing mean activations from {min(n_samples, len(images))} samples")
        prefix = self._get_layer_prefix()
        module = self.model
        for attr in prefix.split("."):
            module = getattr(module, attr)

        n_layers = len(module)
        accumulators = {i: [] for i in range(n_layers)}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(mod, inp, out):
                hidden = out[0] if isinstance(out, tuple) else out
                accumulators[layer_idx].append(hidden.detach().cpu())
            return hook_fn

        for i, layer in enumerate(module):
            h = layer.register_forward_hook(make_hook(i))
            hooks.append(h)

        n_use = min(n_samples, len(images))
        for idx in range(n_use):
            img = images[idx]
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            text = texts[idx]

            inputs = self._prepare_inputs(img, text)
            with torch.no_grad():
                self._run_forward(inputs)

            if idx % 20 == 0:
                torch.cuda.empty_cache()

        for h in hooks:
            h.remove()

        # Compute mean per layer
        for layer_idx, acts in accumulators.items():
            if acts:
                # Mean across all samples and all token positions
                all_acts = torch.cat(acts, dim=0)  # (total_tokens_across_batches, ...)
                self._mean_activations[layer_idx] = all_acts.mean(dim=(0, 1))
                # Shape: (hidden_dim,) - scalar mean per feature
        logger.info(f"Computed mean activations for {len(self._mean_activations)} layers")

    def patch_and_generate(
        self,
        image,
        text: str,
        layer_range: tuple[int, int],
        patch_type: str = "mean",
        max_new_tokens: int = 128,
    ) -> dict:
        """Run model with activation patching and generate output.

        Args:
            image: PIL Image or path
            text: input text prompt
            layer_range: (start_layer, end_layer) inclusive
            patch_type: 'mean', 'zero', or 'noise'
            max_new_tokens: max tokens to generate

        Returns:
            dict with 'text': generated text, 'logits': output logits
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = self._prepare_inputs(image, text)
        self._install_patching_hooks(layer_range, patch_type)

        try:
            with torch.no_grad():
                output = self._run_generate(inputs, max_new_tokens)
        finally:
            self._remove_hooks()

        return output

    def patch_and_forward(
        self,
        image,
        text: str,
        layer_range: tuple[int, int],
        patch_type: str = "mean",
    ) -> dict:
        """Run model with patching, return logits only (no generation).

        Useful for computing perplexity or next-token predictions.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = self._prepare_inputs(image, text)
        self._install_patching_hooks(layer_range, patch_type)

        try:
            with torch.no_grad():
                outputs = self._run_forward(inputs)
        finally:
            self._remove_hooks()

        return outputs

    def _install_patching_hooks(self, layer_range: tuple[int, int],
                                patch_type: str):
        """Install forward hooks that patch activations in the specified range."""
        self._remove_hooks()
        prefix = self._get_layer_prefix()
        module = self.model
        for attr in prefix.split("."):
            module = getattr(module, attr)

        start, end = layer_range

        for i in range(start, min(end + 1, len(module))):
            hook = module[i].register_forward_hook(
                self._make_patch_hook(i, patch_type)
            )
            self._hooks.append(hook)

    def _make_patch_hook(self, layer_idx: int, patch_type: str):
        """Create a hook function that patches activations.

        Full-sequence patching: all token positions are patched,
        not just the last token (per Nooralahzadeh 2026).
        """
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output

            if patch_type == "zero":
                patched = torch.zeros_like(hidden)
            elif patch_type == "mean":
                if layer_idx in self._mean_activations:
                    mean_act = self._mean_activations[layer_idx].to(hidden.device)
                    # Broadcast mean across batch and sequence dimensions
                    patched = mean_act.unsqueeze(0).unsqueeze(0).expand_as(hidden)
                else:
                    # Fallback: use per-sample mean
                    patched = hidden.mean(dim=1, keepdim=True).expand_as(hidden)
            elif patch_type == "noise":
                noise_scale = hidden.std()
                patched = hidden + torch.randn_like(hidden) * noise_scale
            else:
                raise ValueError(f"Unknown patch type: {patch_type}")

            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched

        return hook_fn

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _prepare_inputs(self, image: Image.Image, text: str) -> dict:
        """Prepare model inputs based on model type."""
        hf_id = self.config["hf_id"].lower()

        if "llava" in hf_id:
            # Check if we have manual LLaVA components (vision tower + projector)
            if hasattr(self, '_vision_tower'):
                return self._prepare_inputs_manual_llava(image, text)
            prompt = f"USER: <image>\n{text}\nASSISTANT:"
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        elif "qwen2" in hf_id:
            from qwen_vl_utils import process_vision_info
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ]}]
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_input], images=image_inputs, videos=video_inputs,
                return_tensors="pt", padding=True,
            )
        elif "internvl" in hf_id:
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
            transform = T.Compose([
                T.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            pixel_values = transform(image).unsqueeze(0)
            inputs = {"pixel_values": pixel_values, "text": text}
        else:
            raise ValueError(f"Unsupported model: {hf_id}")

        # Move tensors to device
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        return inputs

    def _run_forward(self, inputs: dict):
        """Run a forward pass (no generation)."""
        if inputs.get("_is_manual"):
            return self.model(inputs_embeds=inputs["inputs_embeds"])
        hf_id = self.config["hf_id"].lower()
        if "internvl" in hf_id:
            pixel_values = inputs["pixel_values"]
            text = inputs["text"]
            question = f"<image>\n{text}"
            gen_config = dict(max_new_tokens=1, do_sample=False)
            return self.model.chat(self.processor, pixel_values, question, gen_config)
        else:
            forward_inputs = {k: v for k, v in inputs.items()
                              if isinstance(v, torch.Tensor)}
            return self.model(**forward_inputs)

    def _run_generate(self, inputs: dict, max_new_tokens: int = 128) -> dict:
        """Run generation and decode output."""
        if inputs.get("_is_manual"):
            out = self.model.generate(
                inputs_embeds=inputs["inputs_embeds"],
                max_new_tokens=max_new_tokens, do_sample=False,
            )
            text = self.processor.decode(out[0], skip_special_tokens=True)
            return {"text": text.strip(), "output_ids": out}

        hf_id = self.config["hf_id"].lower()

        if "internvl" in hf_id:
            pixel_values = inputs["pixel_values"]
            text = inputs["text"]
            question = f"<image>\n{text}"
            gen_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
            response = self.model.chat(
                self.processor, pixel_values, question, gen_config
            )
            return {"text": response if isinstance(response, str) else str(response)}
        else:
            gen_inputs = {k: v for k, v in inputs.items()
                          if isinstance(v, torch.Tensor)}
            gen_inputs["max_new_tokens"] = max_new_tokens
            gen_inputs["do_sample"] = False

            output_ids = self.model.generate(**gen_inputs)

            # Decode only the newly generated tokens
            if "input_ids" in gen_inputs:
                input_len = gen_inputs["input_ids"].shape[1]
                generated_ids = output_ids[:, input_len:]
            else:
                generated_ids = output_ids

            text = self.processor.batch_decode(generated_ids,
                                               skip_special_tokens=True)[0]
            return {"text": text.strip(), "output_ids": output_ids}

    def get_stage_layer_range(self, stage_name: str) -> tuple[int, int]:
        """Get layer range for a named stage from config."""
        stages = self.config["stages"]
        if stage_name not in stages:
            raise ValueError(f"Unknown stage: {stage_name}. "
                             f"Available: {list(stages.keys())}")
        return tuple(stages[stage_name])

    # ---------- convenience APIs used by single_layer_sweep ----------

    def generate(self, image, text: str, max_new_tokens: int = 30) -> str:
        """Baseline generation with no patching. Returns generated text."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        inputs = self._prepare_inputs(image, text)
        self._remove_hooks()
        with torch.no_grad():
            out = self._run_generate(inputs, max_new_tokens)
        return out.get("text", "")

    def patch_stage(
        self,
        images,
        texts: list,
        layer_start: int,
        layer_end: int,
        method: str = "mean",
        max_new_tokens: int = 30,
    ) -> list[str]:
        """Generate with activations in [layer_start, layer_end-1] patched.

        layer_range semantics: half-open like Python's range(). The underlying
        `_install_patching_hooks` uses inclusive end, so we pass layer_end-1.

        Returns:
            List of generated strings, one per (image, text) pair.
        """
        results: list[str] = []
        for img, text in zip(images, texts):
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            inputs = self._prepare_inputs(img, text)
            inclusive_end = max(layer_start, layer_end - 1)
            self._install_patching_hooks((layer_start, inclusive_end), method)
            try:
                with torch.no_grad():
                    out = self._run_generate(inputs, max_new_tokens)
            finally:
                self._remove_hooks()
            results.append(out.get("text", ""))
        return results
