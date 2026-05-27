"""Build task metadata for shared1000 NSD stimuli from COCO annotations.

Outputs per-stimulus metadata used by the three NeuroLens dissociation tasks:
  - Stage 1 (object recognition): main object category from COCO instances
  - Stage 2 (attribute binding): color+object questions derived from instances
  - Stage 3 (caption fluency): reference captions from COCO captions

Stored as a single JSON keyed by NSD ID.
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# COCO supercategories that produce visually-distinctive objects for VQA.
# We avoid 'person' (too generic) and 'accessory' (small) by default.
PREFERRED_SUPERCATEGORIES = {
    "vehicle", "animal", "outdoor", "sports", "kitchen",
    "food", "furniture", "electronic", "indoor", "appliance",
}


def load_coco_annotations(
    coco_annotations_dir: str | Path,
) -> dict:
    """Load and pre-index COCO 2017 annotations for shared1000 lookup."""
    ann_dir = Path(coco_annotations_dir)
    out = {}

    for split in ("train2017", "val2017"):
        inst_path = ann_dir / f"instances_{split}.json"
        cap_path = ann_dir / f"captions_{split}.json"
        if inst_path.exists():
            logger.info(f"Loading instances_{split}.json …")
            with open(inst_path) as f:
                inst = json.load(f)
            out[f"{split}_instances"] = inst
        if cap_path.exists():
            logger.info(f"Loading captions_{split}.json …")
            with open(cap_path) as f:
                cap = json.load(f)
            out[f"{split}_captions"] = cap
    return out


def _index_by_image(coco_data: dict) -> tuple[dict[int, list], dict[int, list], dict[int, str]]:
    """Return (image_id -> list of annotations, image_id -> list of captions,
    category_id -> category_name) maps."""
    img_to_inst: dict[int, list] = {}
    img_to_caps: dict[int, list] = {}
    cat_id_to_name: dict[int, str] = {}
    cat_id_to_super: dict[int, str] = {}

    for split in ("train2017", "val2017"):
        inst = coco_data.get(f"{split}_instances")
        cap = coco_data.get(f"{split}_captions")
        if inst:
            for c in inst["categories"]:
                cat_id_to_name[c["id"]] = c["name"]
                cat_id_to_super[c["id"]] = c["supercategory"]
            for a in inst["annotations"]:
                img_to_inst.setdefault(a["image_id"], []).append(a)
        if cap:
            for a in cap["annotations"]:
                img_to_caps.setdefault(a["image_id"], []).append(a["caption"])

    return img_to_inst, img_to_caps, cat_id_to_name


def _largest_object(
    annotations: list, cat_id_to_name: dict
) -> Optional[dict]:
    """Pick the largest-area annotated object as the 'main object'."""
    if not annotations:
        return None
    main = max(annotations, key=lambda a: a.get("area", 0.0))
    return {
        "category_id": main["category_id"],
        "category_name": cat_id_to_name.get(main["category_id"], "object"),
        "area": float(main.get("area", 0.0)),
        "bbox": main.get("bbox"),
    }


def _generate_color_options(true_color: Optional[str]) -> tuple[list[str], int]:
    """Return (option_list, correct_idx). Used only for placeholder; the actual
    color attribute task needs ground-truth color annotations which COCO doesn't
    provide. For now we fall back to category-based MC."""
    colors = ["red", "green", "blue", "yellow", "white", "black", "gray", "brown"]
    if true_color and true_color in colors:
        return colors, colors.index(true_color)
    # Random placeholder, real annotation needed
    return colors, 0


def _category_mc_options(
    true_cat: str, all_cats: list[str], rng: random.Random
) -> tuple[list[str], int]:
    """Build a 4-option multiple choice where one is the true category."""
    distractors = [c for c in all_cats if c != true_cat]
    distractors = rng.sample(distractors, k=min(3, len(distractors)))
    options = [true_cat] + distractors
    rng.shuffle(options)
    correct_idx = options.index(true_cat)
    return options, correct_idx


def build_shared1000_metadata(
    nsd_stim_info_csv: str | Path,
    coco_annotations_dir: str | Path,
    output_path: str | Path,
    seed: int = 42,
) -> dict:
    """Build per-stimulus task metadata for the 1000 NSD shared stimuli.

    Returns:
        {nsd_id: {
            'coco_id': int,
            'coco_split': str,
            'category': str,
            'category_id': int,
            'caption': str,
            'all_captions': [str, ...],
            'category_mc_options': [str x 4],
            'category_correct_idx': int,
            ...
        }}
    """
    df = pd.read_csv(nsd_stim_info_csv)
    shared = df[df["shared1000"] == True]
    logger.info(f"shared1000: {len(shared)} stimuli")

    coco_data = load_coco_annotations(coco_annotations_dir)
    img_to_inst, img_to_caps, cat_id_to_name = _index_by_image(coco_data)
    all_cat_names = sorted(set(cat_id_to_name.values()))
    rng = random.Random(seed)

    metadata: dict[int, dict] = {}
    missing_inst, missing_cap = 0, 0

    for _, row in shared.iterrows():
        nsd_id = int(row["nsdId"])
        coco_id = int(row["cocoId"])
        coco_split = str(row["cocoSplit"])

        instances = img_to_inst.get(coco_id, [])
        captions = img_to_caps.get(coco_id, [])
        main_obj = _largest_object(instances, cat_id_to_name)

        if main_obj is None:
            missing_inst += 1
            category = "object"
            category_id = -1
        else:
            category = main_obj["category_name"]
            category_id = main_obj["category_id"]

        if not captions:
            missing_cap += 1
            caption = ""
        else:
            caption = captions[0]

        mc_opts, mc_idx = _category_mc_options(category, all_cat_names, rng)

        metadata[nsd_id] = {
            "nsd_id": nsd_id,
            "coco_id": coco_id,
            "coco_split": coco_split,
            "category": category,
            "category_id": category_id,
            "category_mc_options": mc_opts,
            "category_correct_idx": mc_idx,
            "caption": caption,
            "all_captions": captions,
            "n_objects": len(instances),
            "main_object_area": main_obj["area"] if main_obj else 0.0,
        }

    logger.info(
        f"Built metadata for {len(metadata)} stimuli "
        f"(missing instances: {missing_inst}, missing captions: {missing_cap})"
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved task metadata to {out_path}")

    # Summary stats
    category_counts = Counter(m["category"] for m in metadata.values())
    logger.info(f"Top categories: {category_counts.most_common(15)}")
    return metadata


def load_shared1000_metadata(path: str | Path) -> dict[int, dict]:
    """Load previously-generated metadata; keys remain str in JSON, convert to int."""
    with open(path) as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def make_task_batches(
    metadata: dict[int, dict],
    nsd_ids: Iterable[int],
    images: dict[int, "PIL.Image.Image"],
) -> dict[str, dict]:
    """Convert metadata + reconstructed images into task batches for sweeps.

    Returns:
        {
            'object': {images, prompts, ground_truths, ...},
            'category_mc': {images, prompts, option_lists, correct_indices},
            'caption': {images, prompts, ground_truths},
        }
    """
    ids = [int(nid) for nid in nsd_ids]
    ims = [images[nid] for nid in ids]

    # Note: prompts contain NO <image> token. The activation patcher's manual
    # LLaVA path prepends image features as embeddings before the text — the
    # text itself must not reference <image> or embedding lookup blows up
    # (the LLaMA tokenizer doesn't know that special token).
    object_task = {
        "images": ims,
        "prompts": [
            "What is the main object in this image? "
            "Answer with a single word or short phrase."
            for _ in ids
        ],
        "ground_truths": [metadata[nid]["category"] for nid in ids],
        "nsd_ids": ids,
    }

    cat_mc_task = {
        "images": ims,
        "prompts": [
            f"Which of these best describes the main object? "
            f"Choices: {', '.join(metadata[nid]['category_mc_options'])}. "
            f"Answer with one of the choices."
            for nid in ids
        ],
        "ground_truths": [metadata[nid]["category"] for nid in ids],
        "option_lists": [metadata[nid]["category_mc_options"] for nid in ids],
        "correct_indices": [metadata[nid]["category_correct_idx"] for nid in ids],
        "nsd_ids": ids,
    }

    caption_task = {
        "images": ims,
        "prompts": [
            "Describe this image in one sentence."
            for _ in ids
        ],
        "ground_truths": [metadata[nid]["caption"] for nid in ids],
        "all_captions": [metadata[nid]["all_captions"] for nid in ids],
        "nsd_ids": ids,
    }

    return {
        "object": object_task,
        "category_mc": cat_mc_task,
        "caption": caption_task,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsd-stim-info-csv", required=True)
    parser.add_argument("--coco-annotations-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    build_shared1000_metadata(
        args.nsd_stim_info_csv,
        args.coco_annotations_dir,
        args.output,
    )
