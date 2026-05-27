"""Reconstruct NSD stimuli from COCO images using cropBox metadata.

NSD's 73,000 stimuli are derived from COCO by:
  1. Center-cropping each image to a square (based on COCO segmentation masks
     to maximize visible content). The crop is encoded in `cropBox`
     = (top, bottom, left, right) fractions of original image to remove.
  2. Resizing to 425 × 425.

This module replaces the need to download the 39.5 GB nsd_stimuli.hdf5 by
reproducing step 1 + 2 from locally-available COCO images.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

NSD_STIM_SIZE = 425  # NSD canonical stimulus size (pixels)


def parse_crop_box(value) -> tuple[float, float, float, float]:
    """Parse cropBox value from stim_info CSV (stored as tuple-formatted string).

    Format: '(top, bottom, left, right)' — fractions of original image to crop off.
    """
    if isinstance(value, tuple):
        return tuple(float(x) for x in value)  # type: ignore
    if isinstance(value, (list, np.ndarray)):
        return tuple(float(x) for x in value)  # type: ignore
    if isinstance(value, str):
        # Use literal_eval for safety
        parsed = ast.literal_eval(value)
        return tuple(float(x) for x in parsed)  # type: ignore
    raise TypeError(f"Unknown cropBox type: {type(value)}: {value}")


def reconstruct_stimulus(
    coco_path: str | Path,
    crop_box: tuple[float, float, float, float],
    target_size: int = NSD_STIM_SIZE,
    resample: int = Image.BICUBIC,
) -> Image.Image:
    """Reproduce a single NSD stimulus from its source COCO image.

    Args:
        coco_path: path to original COCO image (e.g., train2017/000000262145.jpg)
        crop_box: (top, bottom, left, right) fractions to crop off
        target_size: final size (NSD uses 425)
        resample: PIL resampling filter

    Returns:
        PIL Image of shape (target_size, target_size, 3) in RGB.
    """
    img = Image.open(coco_path).convert("RGB")
    W, H = img.size
    top, bot, left, right = crop_box

    # Crop coordinates: (left, top, right_exclusive, bottom_exclusive)
    left_px = int(round(W * left))
    top_px = int(round(H * top))
    right_px = int(round(W * (1.0 - right)))
    bot_px = int(round(H * (1.0 - bot)))

    if right_px <= left_px or bot_px <= top_px:
        raise ValueError(
            f"Invalid crop box {crop_box} for image {W}x{H} -> "
            f"({left_px}, {top_px}, {right_px}, {bot_px})"
        )

    img = img.crop((left_px, top_px, right_px, bot_px))
    img = img.resize((target_size, target_size), resample)
    return img


class NSDStimulusReconstructor:
    """Build NSD-equivalent PIL images from local COCO images on demand."""

    def __init__(
        self,
        stim_info_csv: str | Path,
        coco_train_dir: Optional[str | Path] = None,
        coco_val_dir: Optional[str | Path] = None,
    ):
        self.stim_info = pd.read_csv(stim_info_csv)
        self.coco_train_dir = Path(coco_train_dir) if coco_train_dir else None
        self.coco_val_dir = Path(coco_val_dir) if coco_val_dir else None
        # Build fast nsdId -> row mapping
        self._nsd_id_to_row = {
            int(row["nsdId"]): i for i, row in self.stim_info.iterrows()
        }

    def coco_path_for_nsd_id(self, nsd_id: int) -> Path:
        """Resolve the local COCO file path for a NSD stimulus."""
        if nsd_id not in self._nsd_id_to_row:
            raise KeyError(f"Unknown NSD ID: {nsd_id}")
        row = self.stim_info.iloc[self._nsd_id_to_row[nsd_id]]
        coco_id = int(row["cocoId"])
        coco_split = str(row["cocoSplit"])  # 'train2017' or 'val2017'

        if coco_split == "train2017":
            if self.coco_train_dir is None:
                raise FileNotFoundError(
                    "COCO train2017 dir not configured but stimulus needs it"
                )
            path = self.coco_train_dir / f"{coco_id:012d}.jpg"
        elif coco_split == "val2017":
            if self.coco_val_dir is None:
                raise FileNotFoundError(
                    "COCO val2017 dir not configured but stimulus needs it"
                )
            path = self.coco_val_dir / f"{coco_id:012d}.jpg"
        else:
            raise ValueError(f"Unknown cocoSplit: {coco_split}")

        if not path.exists():
            raise FileNotFoundError(f"COCO image not found: {path}")
        return path

    def reconstruct(self, nsd_id: int, target_size: int = NSD_STIM_SIZE) -> Image.Image:
        """Reconstruct a single NSD stimulus."""
        row = self.stim_info.iloc[self._nsd_id_to_row[nsd_id]]
        crop_box = parse_crop_box(row["cropBox"])
        return reconstruct_stimulus(
            self.coco_path_for_nsd_id(nsd_id),
            crop_box,
            target_size=target_size,
        )

    def reconstruct_batch(
        self,
        nsd_ids: Sequence[int],
        target_size: int = NSD_STIM_SIZE,
    ) -> list[Image.Image]:
        """Reconstruct a batch of stimuli."""
        return [self.reconstruct(int(nid), target_size=target_size) for nid in nsd_ids]

    def save_batch(
        self,
        nsd_ids: Sequence[int],
        out_dir: str | Path,
        target_size: int = NSD_STIM_SIZE,
        ext: str = "png",
    ) -> list[Path]:
        """Reconstruct and cache stimuli to disk for downstream model extraction."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        paths = []
        for nid in nsd_ids:
            nid_int = int(nid)
            file_path = out_path / f"nsd_{nid_int:05d}.{ext}"
            if not file_path.exists():
                img = self.reconstruct(nid_int, target_size=target_size)
                img.save(file_path)
            paths.append(file_path)
        return paths


def verify_against_hdf5(
    reconstructor: NSDStimulusReconstructor,
    hdf5_path: str | Path,
    nsd_ids_to_check: Sequence[int],
    tolerance_mean: float = 5.0,
    tolerance_max: float = 30.0,
) -> dict:
    """Sanity check: compare reconstructed images against the official NSD HDF5.

    Args:
        reconstructor: configured reconstructor
        hdf5_path: path to nsd_stimuli.hdf5 (only if available)
        nsd_ids_to_check: list of NSD IDs to compare
        tolerance_mean: max acceptable mean absolute pixel diff (out of 255)
        tolerance_max: max acceptable max absolute pixel diff

    Returns:
        Dict with per-ID stats and overall pass/fail.
    """
    import h5py

    results = {"per_id": {}, "all_pass": True, "n_checked": 0}
    with h5py.File(hdf5_path, "r") as f:
        # NSD HDF5 stores images as 'imgBrick' of shape (73000, 425, 425, 3) uint8
        imgs = f["imgBrick"]
        for nid in nsd_ids_to_check:
            nid_int = int(nid)
            # NSD HDF5 is 0-indexed by NSD ID - 1
            official = imgs[nid_int - 1]  # (425, 425, 3) uint8
            reconstructed = np.array(reconstructor.reconstruct(nid_int))

            diff = np.abs(reconstructed.astype(np.int16) - official.astype(np.int16))
            mean_diff = float(diff.mean())
            max_diff = float(diff.max())
            ok = mean_diff < tolerance_mean and max_diff < tolerance_max

            results["per_id"][nid_int] = {
                "mean_diff": mean_diff,
                "max_diff": max_diff,
                "ok": ok,
            }
            if not ok:
                results["all_pass"] = False
                logger.warning(
                    f"NSD {nid_int}: mean_diff={mean_diff:.2f}, max_diff={max_diff}"
                )
            results["n_checked"] += 1
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stim-info-csv", required=True)
    parser.add_argument("--coco-train", required=True)
    parser.add_argument("--coco-val", default=None)
    parser.add_argument("--nsd-id", type=int, default=2950)
    parser.add_argument("--out", default="/tmp/nsd_reconstruct_test.png")
    parser.add_argument("--verify-hdf5", default=None, help="path to nsd_stimuli.hdf5")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    rec = NSDStimulusReconstructor(
        stim_info_csv=args.stim_info_csv,
        coco_train_dir=args.coco_train,
        coco_val_dir=args.coco_val,
    )
    img = rec.reconstruct(args.nsd_id)
    img.save(args.out)
    print(f"Reconstructed NSD ID {args.nsd_id} -> {args.out} ({img.size})")

    if args.verify_hdf5:
        # Sample 10 random shared1000 IDs for verification
        shared = rec.stim_info[rec.stim_info["shared1000"] == True]
        sample_ids = shared.sample(10, random_state=42)["nsdId"].tolist()
        results = verify_against_hdf5(rec, args.verify_hdf5, sample_ids)
        print("Verification:", results["all_pass"])
        for nid, stats in results["per_id"].items():
            print(f"  NSD {nid}: mean_diff={stats['mean_diff']:.2f}, "
                  f"max_diff={stats['max_diff']}, ok={stats['ok']}")
