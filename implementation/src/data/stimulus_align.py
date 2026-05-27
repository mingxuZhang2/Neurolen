"""Align NSD stimuli with COCO images and captions.

Maps NSD stimulus IDs to COCO image IDs, downloads/loads COCO images,
and retrieves corresponding captions to create paired inputs for MLLMs.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class StimulusAligner:
    """Aligns NSD stimuli with COCO images and captions."""

    def __init__(self, nsd_root: str, coco_root: str):
        self.nsd_root = Path(nsd_root)
        self.coco_root = Path(coco_root)
        self._stim_info = None
        self._coco_captions = None

    @property
    def stim_info(self) -> pd.DataFrame:
        if self._stim_info is None:
            path = (self.nsd_root / "nsddata" / "experiments" / "nsd" /
                    "nsd_stim_info_merged.csv")
            self._stim_info = pd.read_csv(path)
        return self._stim_info

    def load_coco_captions(self) -> dict[int, list[str]]:
        """Load COCO captions, mapping image_id -> list of caption strings."""
        if self._coco_captions is not None:
            return self._coco_captions

        self._coco_captions = {}
        for split in ["train2017", "val2017", "train2014", "val2014"]:
            ann_file = (self.coco_root / "annotations" /
                        f"captions_{split}.json")
            if not ann_file.exists():
                continue
            with open(ann_file) as f:
                data = json.load(f)
            for ann in data["annotations"]:
                img_id = ann["image_id"]
                caption = ann["caption"].strip()
                if img_id not in self._coco_captions:
                    self._coco_captions[img_id] = []
                self._coco_captions[img_id].append(caption)

        logger.info(f"Loaded captions for {len(self._coco_captions)} COCO images")
        return self._coco_captions

    def get_coco_image_path(self, coco_id: int) -> Optional[Path]:
        """Find the path to a COCO image given its ID."""
        # COCO images are in train2017/ or val2017/ (or 2014 versions)
        for split in ["train2017", "val2017", "train2014", "val2014"]:
            # COCO filenames are zero-padded to 12 digits
            fname = f"{coco_id:012d}.jpg"
            path = self.coco_root / split / fname
            if path.exists():
                return path

        # Try images/ directory (some setups use flat structure)
        for split in ["images/train2017", "images/val2017"]:
            fname = f"{coco_id:012d}.jpg"
            path = self.coco_root / split / fname
            if path.exists():
                return path

        return None

    def nsd_stimuli_path(self) -> Path:
        """Path to the NSD stimulus images directory."""
        stim_dir = self.nsd_root / "nsddata_stimuli" / "stimuli" / "nsd"
        if stim_dir.exists():
            return stim_dir
        # Alternative location
        stim_dir = self.nsd_root / "stimuli" / "nsd"
        if stim_dir.exists():
            return stim_dir
        return stim_dir  # return anyway, caller checks existence

    def get_nsd_stimulus_image(self, nsd_id: int) -> Optional[Path]:
        """Get path to the NSD stimulus image file."""
        stim_dir = self.nsd_stimuli_path()
        # NSD stimulus filenames
        # Format varies: nsd-XXXXX.png or similar
        for fmt in [f"nsd{nsd_id:05d}.png", f"nsd-{nsd_id:05d}.png",
                    f"stimuli_{nsd_id:05d}.png", f"{nsd_id}.png"]:
            path = stim_dir / fmt
            if path.exists():
                return path
        return None

    def create_stimulus_pairs(
        self,
        nsd_ids: np.ndarray,
        prompt_template: str = "Describe this image in detail.",
        use_coco_captions: bool = True,
        use_nsd_images: bool = True,
    ) -> dict:
        """Create paired (image_path, text) inputs for MLLMs.

        Args:
            nsd_ids: NSD stimulus IDs
            prompt_template: default text prompt if no caption available
            use_coco_captions: if True, use COCO captions as text input
            use_nsd_images: if True, try NSD stimulus images first, else use COCO

        Returns:
            dict with keys:
                'image_paths': list of valid image paths
                'texts': list of text inputs
                'nsd_ids': list of corresponding NSD IDs (only valid ones)
                'coco_ids': list of corresponding COCO IDs
        """
        coco_ids = self._get_coco_ids(nsd_ids)
        captions = self.load_coco_captions() if use_coco_captions else {}

        image_paths = []
        texts = []
        valid_nsd_ids = []
        valid_coco_ids = []

        for nsd_id, coco_id in tqdm(zip(nsd_ids, coco_ids),
                                     total=len(nsd_ids),
                                     desc="Aligning stimuli"):
            # Find image
            img_path = None
            if use_nsd_images:
                img_path = self.get_nsd_stimulus_image(nsd_id)
            if img_path is None and coco_id > 0:
                img_path = self.get_coco_image_path(coco_id)

            if img_path is None:
                logger.warning(f"Image not found for nsdId={nsd_id}, cocoId={coco_id}")
                continue

            # Find text
            if use_coco_captions and coco_id in captions:
                # Use the first COCO caption
                text = captions[coco_id][0]
            else:
                text = prompt_template

            image_paths.append(str(img_path))
            texts.append(text)
            valid_nsd_ids.append(int(nsd_id))
            valid_coco_ids.append(int(coco_id))

        logger.info(f"Created {len(image_paths)} stimulus pairs "
                    f"({len(nsd_ids) - len(image_paths)} missing)")

        return {
            "image_paths": image_paths,
            "texts": texts,
            "nsd_ids": valid_nsd_ids,
            "coco_ids": valid_coco_ids,
        }

    def _get_coco_ids(self, nsd_ids: np.ndarray) -> np.ndarray:
        """Map NSD IDs to COCO IDs via stim_info."""
        mapping = self.stim_info.set_index('nsdId')['cocoId']
        return np.array([mapping.get(nid, -1) for nid in nsd_ids])

    def create_prompt_only_pairs(
        self,
        nsd_ids: np.ndarray,
        prompt: str = "Describe this image in detail.",
    ) -> dict:
        """Create stimulus pairs using a fixed prompt (no COCO captions).

        Useful for activation extraction where we want consistent text input.
        """
        return self.create_stimulus_pairs(
            nsd_ids, prompt_template=prompt, use_coco_captions=False,
        )

    def save_stimulus_pairs(self, pairs: dict, output_path: str):
        """Save stimulus pairs to JSON for reproducibility."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(pairs, f, indent=2)
        logger.info(f"Saved {len(pairs['image_paths'])} stimulus pairs to {out}")

    def load_stimulus_pairs(self, input_path: str) -> dict:
        """Load previously saved stimulus pairs."""
        with open(input_path) as f:
            return json.load(f)

    def get_multi_object_scenes(
        self,
        nsd_ids: np.ndarray,
        min_objects: int = 3,
    ) -> dict:
        """Find NSD stimuli that contain multiple objects (for binding task).

        Uses COCO instance annotations to identify scenes with multiple
        distinct objects with different attributes.

        Returns:
            dict with scene info including object annotations
        """
        # Load COCO instance annotations
        instances = {}
        for split in ["train2017", "val2017", "train2014", "val2014"]:
            ann_file = (self.coco_root / "annotations" /
                        f"instances_{split}.json")
            if not ann_file.exists():
                continue
            with open(ann_file) as f:
                data = json.load(f)

            # Build category map
            cat_map = {c['id']: c['name'] for c in data['categories']}

            # Group annotations by image
            for ann in data['annotations']:
                img_id = ann['image_id']
                if img_id not in instances:
                    instances[img_id] = []
                instances[img_id].append({
                    'category': cat_map.get(ann['category_id'], 'unknown'),
                    'bbox': ann['bbox'],
                    'area': ann['area'],
                })

        coco_ids = self._get_coco_ids(nsd_ids)
        multi_object_scenes = []

        for nsd_id, coco_id in zip(nsd_ids, coco_ids):
            if coco_id <= 0:
                continue
            anns = instances.get(int(coco_id), [])
            # Count distinct object categories
            categories = list(set(a['category'] for a in anns))
            if len(categories) >= min_objects:
                multi_object_scenes.append({
                    'nsd_id': int(nsd_id),
                    'coco_id': int(coco_id),
                    'n_objects': len(anns),
                    'n_categories': len(categories),
                    'categories': categories,
                    'annotations': anns,
                })

        logger.info(f"Found {len(multi_object_scenes)} multi-object scenes "
                    f"(>= {min_objects} categories)")
        return multi_object_scenes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Align NSD stimuli with COCO")
    parser.add_argument("--nsd-root", required=True)
    parser.add_argument("--coco-root", required=True)
    parser.add_argument("--output", default="stimuli/stimulus_pairs.json")
    parser.add_argument("--prompt", default="Describe this image in detail.")
    parser.add_argument("--use-captions", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    from src.data.nsd_loader import NSDLoader
    nsd = NSDLoader(args.nsd_root)
    shared_ids = nsd.get_shared_image_ids()

    aligner = StimulusAligner(args.nsd_root, args.coco_root)
    pairs = aligner.create_stimulus_pairs(
        shared_ids,
        prompt_template=args.prompt,
        use_coco_captions=args.use_captions,
    )
    aligner.save_stimulus_pairs(pairs, args.output)
