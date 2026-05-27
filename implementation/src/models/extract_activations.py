"""Hook-based activation extraction for MLLMs.

Extracts per-layer hidden states from LLaVA-1.5-7B, Qwen2-VL-7B, and InternVL2-8B.
Separates visual tokens from text tokens, applies PCA reduction, and saves as numpy arrays.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class ActivationStore:
    """Accumulates activations across batches and saves incrementally."""
    output_dir: Path
    model_name: str
    n_layers: int
    hidden_dim: int
    pca_dim: int = 256
    token_types: list = field(default_factory=lambda: ["vis", "txt", "all"])

    def __post_init__(self):
        self.output_dir = Path(self.output_dir) / self.model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Per-layer, per-token-type accumulator
        self._buffers: dict[str, list[np.ndarray]] = {}
        for l in range(self.n_layers):
            for tt in self.token_types:
                self._buffers[f"layer_{l}_{tt}"] = []

    def append(self, layer_idx: int, token_type: str, activation: np.ndarray):
        key = f"layer_{layer_idx}_{token_type}"
        self._buffers[key].append(activation)

    def finalize_and_save(self):
        """Concatenate all batches, apply PCA, and save to disk."""
        logger.info(f"Finalizing activations for {self.model_name}")
        for key, buffer in tqdm(self._buffers.items(), desc="Saving activations"):
            if not buffer:
                continue
            # Concatenate across stimuli: (n_stimuli, hidden_dim)
            data = np.concatenate(buffer, axis=0)
            # Save raw (pre-PCA)
            raw_path = self.output_dir / f"{key}_raw.npy"
            np.save(raw_path, data)
            # PCA reduction (skip when pca_dim <= 0)
            if (self.pca_dim is not None and self.pca_dim > 0
                    and data.shape[1] > self.pca_dim
                    and data.shape[0] > self.pca_dim):
                pca = PCA(n_components=self.pca_dim, random_state=42)
                data_pca = pca.fit_transform(data)
                explained_var = pca.explained_variance_ratio_.sum()
                logger.info(f"  {key}: PCA {data.shape[1]} -> {self.pca_dim}, "
                            f"explained variance: {explained_var:.3f}")
            else:
                data_pca = data
                logger.info(f"  {key}: kept raw shape {data.shape} (PCA disabled or "
                            f"insufficient data)")
            pca_path = self.output_dir / f"{key}.npy"
            np.save(pca_path, data_pca)
        logger.info(f"Saved all activations to {self.output_dir}")


def _get_visual_token_mask(input_ids: torch.Tensor, model_name: str) -> torch.Tensor:
    """Identify visual token positions in the input sequence.

    Returns a boolean mask of shape (seq_len,) where True = visual token.
    """
    if "llava" in model_name:
        # LLaVA uses a special image token (typically token_id=32000 for <image>)
        # After processing, image tokens are replaced by vision features.
        # The processor inserts image_token_id at image positions.
        # We detect by checking for contiguous blocks of the same embedding pattern.
        # A safer approach: use the processor's image_token_index
        # For LLaVA-1.5, image tokens occupy a contiguous block.
        # We'll pass the mask separately from the caller.
        pass
    return None  # Caller handles mask via processor outputs


class ActivationExtractor:
    """Extracts layer-wise activations from MLLMs using forward hooks."""

    def __init__(self, model_config: dict, device: str = "cuda",
                 dtype: str = "float16"):
        self.config = model_config
        self.device = device
        self.dtype = getattr(torch, dtype)
        self.model = None
        self.processor = None
        self._hooks = []
        self._activations: dict[str, torch.Tensor] = {}

    def load_model(self):
        """Load the MLLM and processor from HuggingFace."""
        model_name = self.config["name"]
        hf_id = self.config["hf_id"]
        logger.info(f"Loading {model_name} from {hf_id}")

        if "llava" in hf_id.lower():
            self._load_llava(hf_id)
        elif "qwen2" in hf_id.lower():
            self._load_qwen2vl(hf_id)
        elif "internvl" in hf_id.lower():
            self._load_internvl2(hf_id)
        else:
            raise ValueError(f"Unsupported model: {hf_id}")

        self.model.eval()
        logger.info(f"Loaded {model_name} successfully")

    def _load_llava(self, hf_id: str):
        import os, json, glob
        from transformers import (LlamaForCausalLM, LlamaConfig, AutoTokenizer,
                                  CLIPVisionModel, CLIPImageProcessor)

        model_dir = hf_id if os.path.isdir(hf_id) else None
        if not model_dir:
            raise ValueError(f"LLaVA must be loaded from local directory: {hf_id}")

        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)

        # Build LlamaConfig directly
        llama_cfg = LlamaConfig(
            vocab_size=cfg.get("vocab_size", 32000),
            hidden_size=cfg.get("hidden_size", 4096),
            intermediate_size=cfg.get("intermediate_size", 11008),
            num_hidden_layers=cfg.get("num_hidden_layers", 32),
            num_attention_heads=cfg.get("num_attention_heads", 32),
            num_key_value_heads=cfg.get("num_key_value_heads", 32),
            max_position_embeddings=cfg.get("max_position_embeddings", 4096),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
            torch_dtype=self.dtype,
        )

        logger.info("Loading LLM backbone with manual weight mapping...")
        self.model = LlamaForCausalLM(llama_cfg)

        # Load weights manually with key remapping
        idx_path = os.path.join(model_dir, "pytorch_model.bin.index.json")
        with open(idx_path) as f:
            idx = json.load(f)
        shard_files = set(idx["weight_map"].values())

        llm_state = {}
        proj_weights = {}
        for shard_file in sorted(shard_files):
            shard_path = os.path.join(model_dir, shard_file)
            logger.info(f"  Loading shard: {shard_file}")
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            for k, v in shard.items():
                # Strip any HF-format prefix back to original
                clean_k = k
                for prefix in ["model.language_model.", "model.language_model.model."]:
                    if clean_k.startswith(prefix):
                        clean_k = clean_k[len(prefix):]
                        if not clean_k.startswith("model."):
                            clean_k = "model." + clean_k
                        break

                if clean_k.startswith("model.mm_projector.") or clean_k.startswith("model.multi_modal_projector."):
                    proj_weights[clean_k] = v
                elif clean_k.startswith("model.vision_tower."):
                    pass  # skip, loaded from CLIP separately
                elif clean_k.startswith("model.") or clean_k == "lm_head.weight":
                    llm_state[clean_k] = v

        # Load into LlamaForCausalLM
        missing, unexpected = self.model.load_state_dict(llm_state, strict=False)
        loaded = len(llm_state) - len(unexpected)
        logger.info(f"  LLM: loaded {loaded} tensors, {len(missing)} missing, {len(unexpected)} unexpected")

        self.model = self.model.to(dtype=self.dtype)
        self.model = self.model.to("cuda")

        # Verify weights loaded
        w = self.model.model.layers[0].self_attn.q_proj.weight
        logger.info(f"  LLM verify: layer0 q_proj std={w.std():.6f}")

        # Load CLIP vision tower
        clip_path = cfg.get("mm_vision_tower", "")
        logger.info(f"Loading CLIP from {clip_path}")
        self._vision_tower = CLIPVisionModel.from_pretrained(
            clip_path, torch_dtype=self.dtype,
        ).to(self.model.device)
        self._vision_tower.eval()
        self._image_processor = CLIPImageProcessor.from_pretrained(clip_path)

        # Load projector
        mm_proj_path = os.path.join(model_dir, "mm_projector.bin")
        if os.path.exists(mm_proj_path) and not proj_weights:
            proj_weights = torch.load(mm_proj_path, map_location="cpu", weights_only=True)

        hidden_size = cfg.get("hidden_size", 4096)
        mm_hidden_size = cfg.get("mm_hidden_size", 1024)
        self._mm_projector = torch.nn.Sequential(
            torch.nn.Linear(mm_hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, hidden_size),
        ).to(self.model.device, dtype=self.dtype)

        proj_state = {}
        for k, v in proj_weights.items():
            clean = k
            for prefix in ["model.mm_projector.", "model.multi_modal_projector."]:
                if clean.startswith(prefix):
                    clean = clean[len(prefix):]
                    break
            if clean.startswith("linear_1"): clean = clean.replace("linear_1", "0")
            if clean.startswith("linear_2"): clean = clean.replace("linear_2", "2")
            proj_state[clean] = v
        self._mm_projector.load_state_dict(proj_state)
        self._mm_projector.eval()
        logger.info(f"  Projector: loaded {len(proj_state)} tensors")

        self.processor = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        self._is_manual_llava = True
        logger.info("LLaVA loaded successfully (manual assembly)")

    def _load_qwen2vl(self, hf_id: str):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            hf_id, torch_dtype=self.dtype, device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(hf_id)

    def _load_internvl2(self, hf_id: str):
        from transformers import AutoModel, AutoTokenizer
        self.model = AutoModel.from_pretrained(
            hf_id, torch_dtype=self.dtype, device_map="auto",
            low_cpu_mem_usage=True, trust_remote_code=True,
        )
        self.processor = AutoTokenizer.from_pretrained(
            hf_id, trust_remote_code=True,
        )

    def _register_hooks(self):
        """Register forward hooks on all LLM decoder layers."""
        self._remove_hooks()
        self._activations.clear()

        if getattr(self, '_is_manual_llava', False):
            prefix = "model.layers"
        else:
            prefix = self.config["activation_hook_prefix"]
        module = self.model
        for attr in prefix.split("."):
            module = getattr(module, attr)

        for i, layer in enumerate(module):
            hook = layer.register_forward_hook(self._make_hook(f"layer_{i}"))
            self._hooks.append(hook)
        logger.info(f"Registered hooks on {len(self._hooks)} layers")

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            # output is typically a tuple; first element is hidden states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self._activations[name] = hidden.detach().cpu().float()
        return hook_fn

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _prepare_input_llava(self, image: Image.Image, text: str) -> dict:
        import torch

        if getattr(self, '_is_manual_llava', False):
            pixel_values = self._image_processor(
                images=image, return_tensors="pt"
            )["pixel_values"].to(self.model.device, dtype=self.dtype)

            with torch.no_grad():
                vision_out = self._vision_tower(pixel_values, output_hidden_states=True)
                image_features = vision_out.hidden_states[-2][:, 1:, :]
                image_features = self._mm_projector(image_features)

            prompt = f"USER: \n{text}\nASSISTANT:"
            text_ids = self.processor(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
            text_embeds = self.model.get_input_embeddings()(text_ids)
            inputs_embeds = torch.cat([image_features, text_embeds], dim=1)

            return {
                "inputs_embeds": inputs_embeds,
                "_n_visual_tokens": image_features.shape[1],
            }
        else:
            prompt = f"USER: <image>\n{text}\nASSISTANT:"
            inputs = self.processor(
                text=prompt, images=image, return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v
                      for k, v in inputs.items()}
            return inputs

    def _prepare_input_qwen2vl(self, image: Image.Image, text: str) -> dict:
        from qwen_vl_utils import process_vision_info
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ]}
        ]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input], images=image_inputs, videos=video_inputs,
            return_tensors="pt", padding=True,
        )
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v
                  for k, v in inputs.items()}
        return inputs

    def _prepare_input_internvl2(self, image: Image.Image, text: str) -> dict:
        # InternVL2 uses its own chat interface
        # We prepare pixel values and input_ids separately
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        def build_transform(input_size=448):
            return T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((input_size, input_size),
                         interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])

        transform = build_transform()
        pixel_values = transform(image).unsqueeze(0).to(
            self.model.device, dtype=self.dtype
        )
        question = f"<image>\n{text}"
        generation_config = dict(max_new_tokens=1, do_sample=False)
        return {
            "pixel_values": pixel_values,
            "question": question,
            "generation_config": generation_config,
        }

    def _get_visual_token_count(self, inputs: dict) -> int:
        """Estimate number of visual tokens in the input sequence."""
        hf_id = self.config["hf_id"].lower()
        if "llava" in hf_id:
            # LLaVA-1.5 with CLIP-ViT-L/14-336px: 576 visual tokens (24x24 patches)
            return 576
        elif "qwen2" in hf_id:
            # Qwen2-VL uses dynamic resolution; estimate from input
            # The image_grid_thw in inputs tells us the grid shape
            if "image_grid_thw" in inputs:
                grid = inputs["image_grid_thw"]
                if hasattr(grid, 'tolist'):
                    grid = grid.tolist()
                    if len(grid) > 0:
                        t, h, w = grid[0] if isinstance(grid[0], (list, tuple)) else grid
                        return int(t * h * w)
            return 256  # fallback
        elif "internvl" in hf_id:
            # InternVL2 with single tile: 256 visual tokens
            return 256
        return 256

    def extract_for_stimulus(self, image: Image.Image, text: str) -> dict:
        """Extract activations for a single image-text pair.

        Returns dict mapping layer_name -> {vis, txt, all} -> np.ndarray of shape (1, hidden_dim).
        """
        hf_id = self.config["hf_id"].lower()

        if "llava" in hf_id:
            inputs = self._prepare_input_llava(image, text)
        elif "qwen2" in hf_id:
            inputs = self._prepare_input_qwen2vl(image, text)
        elif "internvl" in hf_id:
            inputs = self._prepare_input_internvl2(image, text)
        else:
            raise ValueError(f"Unsupported: {hf_id}")

        self._activations.clear()

        with torch.no_grad():
            if getattr(self, '_is_manual_llava', False):
                self.model(inputs_embeds=inputs["inputs_embeds"])
                n_vis = inputs.get("_n_visual_tokens", 576)
            elif "internvl" in hf_id:
                pixel_values = inputs["pixel_values"]
                question = inputs["question"]
                try:
                    self.model.chat(
                        self.processor, pixel_values, question,
                        inputs["generation_config"],
                    )
                except Exception:
                    pass
                n_vis = self._get_visual_token_count(inputs)
            else:
                gen_kwargs = {k: v for k, v in inputs.items()
                              if not k.startswith("_")}
                gen_kwargs["max_new_tokens"] = 1
                gen_kwargs["do_sample"] = False
                self.model.generate(**gen_kwargs)
                n_vis = self._get_visual_token_count(inputs)
        result = {}
        for name, act in self._activations.items():
            # act shape: (1, seq_len, hidden_dim)
            if act.dim() == 3:
                act = act[0]  # (seq_len, hidden_dim)
            seq_len = act.shape[0]
            # Split into visual and text tokens
            vis_end = min(n_vis, seq_len)
            vis_tokens = act[:vis_end]
            txt_tokens = act[vis_end:]

            result[name] = {
                "vis": vis_tokens.mean(dim=0, keepdim=True).numpy(),   # (1, hidden_dim)
                "txt": txt_tokens.mean(dim=0, keepdim=True).numpy() if txt_tokens.shape[0] > 0
                       else np.zeros((1, act.shape[-1])),
                "all": act.mean(dim=0, keepdim=True).numpy(),          # (1, hidden_dim)
            }
        return result

    def extract_all(
        self,
        images: list,
        texts: list,
        output_dir: str,
        batch_size: int = 4,
        pca_dim: int = 256,
    ):
        """Extract activations for all stimuli and save to disk.

        Args:
            images: list of PIL Images or file paths
            texts: list of text prompts (one per image)
            output_dir: directory to save activation arrays
            batch_size: not used for sequential processing but reserved for future
            pca_dim: PCA reduction dimensionality
        """
        assert len(images) == len(texts), "images and texts must be same length"
        n_layers = self.config["n_layers"]
        hidden_dim = self.config["hidden_dim"]

        store = ActivationStore(
            output_dir=output_dir,
            model_name=self.config["name"].replace(" ", "_").replace("-", "_"),
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            pca_dim=pca_dim,
        )

        self._register_hooks()

        for idx in tqdm(range(len(images)), desc=f"Extracting {self.config['name']}"):
            img = images[idx]
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
            text = texts[idx]

            try:
                layer_acts = self.extract_for_stimulus(img, text)
                for layer_name, token_acts in layer_acts.items():
                    layer_idx = int(layer_name.split("_")[1])
                    for token_type in ["vis", "txt", "all"]:
                        store.append(layer_idx, token_type, token_acts[token_type])
            except Exception as e:
                logger.warning(f"Failed on stimulus {idx}: {e}")
                # Append zeros as placeholder to keep alignment
                for l in range(n_layers):
                    for tt in ["vis", "txt", "all"]:
                        store.append(l, tt, np.zeros((1, hidden_dim)))

            # Periodic memory cleanup
            if idx % 50 == 0:
                torch.cuda.empty_cache()

        self._remove_hooks()
        store.finalize_and_save()

    def extract_vision_encoder(
        self,
        images: list,
        output_dir: str,
    ):
        """Extract intermediate vision encoder representations."""
        ve_config = self.config.get("vision_encoder", {})
        extract_layers = ve_config.get("extract_layers", [])
        if not extract_layers:
            logger.warning("No vision encoder layers specified, skipping")
            return

        out_path = Path(output_dir) / self.config["name"].replace(" ", "_").replace("-", "_")
        out_path.mkdir(parents=True, exist_ok=True)

        hf_id = self.config["hf_id"].lower()
        ve_hooks = []
        ve_activations = {}

        # Get the vision encoder module
        if "llava" in hf_id:
            vision_tower = self.model.vision_tower
            if hasattr(vision_tower, 'vision_model'):
                encoder_layers = vision_tower.vision_model.encoder.layers
            else:
                encoder_layers = vision_tower.encoder.layers
        elif "qwen2" in hf_id:
            vision_tower = self.model.visual
            encoder_layers = vision_tower.blocks if hasattr(vision_tower, 'blocks') else []
        elif "internvl" in hf_id:
            vision_tower = self.model.vision_model
            encoder_layers = vision_tower.encoder.layers
        else:
            logger.warning(f"Vision encoder extraction not supported for {hf_id}")
            return

        def make_ve_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    ve_activations[name] = output[0].detach().cpu().float()
                else:
                    ve_activations[name] = output.detach().cpu().float()
            return hook_fn

        for layer_idx in extract_layers:
            if layer_idx < len(encoder_layers):
                h = encoder_layers[layer_idx].register_forward_hook(
                    make_ve_hook(f"ve_layer_{layer_idx}")
                )
                ve_hooks.append(h)

        # Collect activations per layer
        layer_buffers = {f"ve_layer_{l}": [] for l in extract_layers}

        for idx in tqdm(range(len(images)), desc="Extracting vision encoder"):
            img = images[idx]
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")

            ve_activations.clear()
            try:
                # Process image through vision encoder only
                if "llava" in hf_id:
                    inputs = self.processor(images=img, return_tensors="pt")
                    pixel_values = inputs["pixel_values"].to(
                        self.model.device, dtype=self.dtype
                    )
                    with torch.no_grad():
                        vision_tower(pixel_values)
                elif "qwen2" in hf_id:
                    inputs = self.processor(images=[img], return_tensors="pt")
                    if "pixel_values" in inputs:
                        pixel_values = inputs["pixel_values"].to(
                            self.model.device, dtype=self.dtype
                        )
                        with torch.no_grad():
                            self.model.visual(pixel_values)
                elif "internvl" in hf_id:
                    import torchvision.transforms as T
                    from torchvision.transforms.functional import InterpolationMode
                    transform = T.Compose([
                        T.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
                        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                        T.ToTensor(),
                        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ])
                    pixel_values = transform(img).unsqueeze(0).to(
                        self.model.device, dtype=self.dtype
                    )
                    with torch.no_grad():
                        self.model.vision_model(pixel_values)

                for name in layer_buffers:
                    if name in ve_activations:
                        act = ve_activations[name]
                        if act.dim() == 3:
                            act = act[0]
                        # Use CLS token (first position) or mean pool
                        cls_repr = act[0:1]  # (1, hidden_dim)
                        layer_buffers[name].append(cls_repr.numpy())
                    else:
                        layer_buffers[name].append(None)

            except Exception as e:
                logger.warning(f"Vision encoder extraction failed on stimulus {idx}: {e}")
                for name in layer_buffers:
                    layer_buffers[name].append(None)

        # Remove hooks
        for h in ve_hooks:
            h.remove()

        # Save
        for name, buffer in layer_buffers.items():
            valid = [b for b in buffer if b is not None]
            if valid:
                data = np.concatenate(valid, axis=0)
                np.save(out_path / f"{name}.npy", data)
                logger.info(f"Saved {name}: {data.shape}")

    def cleanup(self):
        """Free model memory."""
        self._remove_hooks()
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()


def load_model_config(config_path: str, model_key: str) -> dict:
    """Load model configuration from YAML."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["models"][model_key]


def run_extraction(
    model_key: str,
    images: list,
    texts: list,
    config_path: str = "configs/models.yaml",
    output_dir: str = "activations",
    pca_dim: int = 256,
    extract_vision_encoder: bool = True,
):
    """Run the full extraction pipeline for one model."""
    model_config = load_model_config(config_path, model_key)
    extractor = ActivationExtractor(model_config)
    extractor.load_model()

    extractor.extract_all(images, texts, output_dir, pca_dim=pca_dim)

    if extract_vision_encoder:
        extractor.extract_vision_encoder(images, output_dir)

    extractor.cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract MLLM activations")
    parser.add_argument("--model", required=True, choices=["llava", "qwen2vl", "internvl2"])
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--stimulus-file", required=True,
                        help="JSON file with {image_paths: [...], texts: [...]}")
    parser.add_argument("--output-dir", default="activations")
    parser.add_argument("--pca-dim", type=int, default=256)
    parser.add_argument("--skip-vision-encoder", action="store_true")

    args = parser.parse_args()

    import json
    with open(args.stimulus_file) as f:
        stimuli = json.load(f)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    run_extraction(
        model_key=args.model,
        images=stimuli["image_paths"],
        texts=stimuli["texts"],
        config_path=args.config,
        output_dir=args.output_dir,
        pca_dim=args.pca_dim,
        extract_vision_encoder=not args.skip_vision_encoder,
    )
