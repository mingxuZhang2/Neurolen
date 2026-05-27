"""NSD fsaverage surface-space data loader.

Loads NSD GLMsingle betas projected to FreeSurfer fsaverage surface, applies
ROI masks (Kastner2015, HCP-MMP1, streams, floc probmaps), averages across
stimulus repetitions, and returns clean per-stimulus per-ROI voxel patterns.

Used by NeuroLens for real-fMRI alignment analysis on the shared1000 NSD subset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import nibabel as nib
import numpy as np
from scipy.io import loadmat

logger = logging.getLogger(__name__)

# NSD constants
N_VERTICES_PER_HEMI = 163842  # fsaverage standard
N_TRIALS_PER_SESSION = 750
N_SUBJECTS = 8
N_REPS_PER_STIMULUS = 3
SHARED_1000_COUNT = 1000


@dataclass
class NSDExpDesign:
    """NSD experimental design metadata loaded from nsd_expdesign.mat."""
    masterordering: np.ndarray  # (30000,) values 1..10000 = stim slot per trial
    subjectim: np.ndarray  # (8, 10000) NSD IDs per subject
    sharedix: np.ndarray  # (1000,) NSD IDs of shared images

    @classmethod
    def load(cls, path: str | Path) -> "NSDExpDesign":
        """Load and convert to 0-indexed NSD IDs (matching nsd_stim_info CSV).

        Raw NSD expdesign.mat uses Matlab 1-indexing. We subtract 1 from all
        NSD ID values (subjectim, sharedix) so they match the CSV's `nsdId`
        column (0..72999). masterordering values are slot indices (1..10000),
        not NSD IDs — kept as 1-indexed since we use them via subjectim[slot-1].
        """
        mat = loadmat(str(path))
        return cls(
            masterordering=mat["masterordering"].flatten().astype(np.int64),
            subjectim=mat["subjectim"].astype(np.int64) - 1,  # to 0-indexed NSD IDs
            sharedix=mat["sharedix"].flatten().astype(np.int64) - 1,
        )

    def shared_trial_indices(self, subject_idx: int) -> dict[int, list[int]]:
        """Map shared NSD ID -> list of trial indices (0-based) where this subject saw it.

        Args:
            subject_idx: 0..7

        Returns:
            {nsd_id: [trial_indices...]} where len(trial_indices) <= 3.
        """
        # subjectim[subject_idx] gives the 10000 stimulus slots in shown order.
        # masterordering says, for trial t (1..30000), which slot (1..10000) was shown.
        subj_slots = self.subjectim[subject_idx]  # (10000,) NSD IDs
        slot_for_trial = self.masterordering  # (30000,) slot indices 1..10000

        # Build slot -> trial list mapping (which trials show each of the 10000 slots)
        slot_to_trials: dict[int, list[int]] = {}
        for trial_idx, slot in enumerate(slot_for_trial):
            slot_to_trials.setdefault(int(slot), []).append(trial_idx)

        shared_set = set(int(x) for x in self.sharedix)
        result: dict[int, list[int]] = {}
        for slot_idx in range(1, 10001):  # 1-indexed slots
            nsd_id = int(subj_slots[slot_idx - 1])
            if nsd_id in shared_set:
                result[nsd_id] = slot_to_trials.get(slot_idx, [])
        return result


class NSDFsaverageLoader:
    """Load fsaverage-projected NSD betas + ROI masks.

    Layout assumed:
      {nsd_root}/nsddata/experiments/nsd/nsd_expdesign.mat
      {nsd_root}/nsddata/freesurfer/fsaverage/label/{lh,rh}.*.mgz
      {nsd_root}/nsddata_betas/ppdata/subjNN/fsaverage/betas_fithrf_GLMdenoise_RR/{lh,rh}.betas_sessionNN.mgh
      {nsd_root}/nsddata_betas/ppdata/subjNN/fsaverage/betas_fithrf_GLMdenoise_RR/{lh,rh}.ncsnr.mgh
    """

    BETAS_DIR_SUFFIX = "fsaverage/betas_fithrf_GLMdenoise_RR"

    def __init__(self, nsd_root: str | Path):
        self.nsd_root = Path(nsd_root)
        self._expdesign: Optional[NSDExpDesign] = None
        self._roi_cache: dict[str, np.ndarray] = {}

    @property
    def expdesign(self) -> NSDExpDesign:
        if self._expdesign is None:
            self._expdesign = NSDExpDesign.load(
                self.nsd_root / "nsddata/experiments/nsd/nsd_expdesign.mat"
            )
        return self._expdesign

    @property
    def shared_nsd_ids(self) -> np.ndarray:
        """Return the 1000 NSD IDs (1-indexed) shared across all subjects."""
        return self.expdesign.sharedix.copy()

    def betas_dir(self, subject_idx: int) -> Path:
        return self.nsd_root / f"nsddata_betas/ppdata/subj{subject_idx+1:02d}/{self.BETAS_DIR_SUFFIX}"

    def available_sessions(
        self, subject_idx: int, hemisphere: str = "both"
    ) -> list[int]:
        """List sessions (1-indexed) for which the required hemisphere beta
        file(s) exist.

        hemisphere: 'lh', 'rh', or 'both' (default — requires both).
        """
        bdir = self.betas_dir(subject_idx)
        if not bdir.exists():
            return []
        sessions = []
        for s in range(1, 41):
            lh = bdir / f"lh.betas_session{s:02d}.mgh"
            rh = bdir / f"rh.betas_session{s:02d}.mgh"
            if hemisphere == "lh" and lh.exists():
                sessions.append(s)
            elif hemisphere == "rh" and rh.exists():
                sessions.append(s)
            elif hemisphere == "both" and lh.exists() and rh.exists():
                sessions.append(s)
        return sessions

    @staticmethod
    def _load_mgh(path: Path) -> np.ndarray:
        """Load a FreeSurfer .mgh/.mgz file as a numpy array.

        Beta session files have shape (n_vertices, 1, 1, 750).
        ROI label files have shape (n_vertices, 1, 1) or similar.
        Returns array squeezed to its meaningful dimensions.
        """
        img = nib.load(str(path))
        data = np.asarray(img.dataobj).squeeze()
        return data

    def load_session_betas(
        self,
        subject_idx: int,
        session: int,
        hemisphere: str = "both",
    ) -> np.ndarray:
        """Load one session of betas.

        Returns:
            (n_vertices, 750) float array. NSD stores betas as int16 *300 internally
            then converts to float32 in standard pipelines; we cast to float32.
            If hemisphere=='both', concatenates [lh, rh].
        """
        bdir = self.betas_dir(subject_idx)
        if hemisphere in ("lh", "both"):
            lh = self._load_mgh(bdir / f"lh.betas_session{session:02d}.mgh").astype(np.float32)
        if hemisphere in ("rh", "both"):
            rh = self._load_mgh(bdir / f"rh.betas_session{session:02d}.mgh").astype(np.float32)
        if hemisphere == "lh":
            return lh
        if hemisphere == "rh":
            return rh
        return np.concatenate([lh, rh], axis=0)

    def load_ncsnr(self, subject_idx: int, hemisphere: str = "both") -> np.ndarray:
        """Load noise-ceiling SNR per vertex for this subject.

        Returns:
            (n_vertices,) float array. For hemisphere='both' concatenates [lh, rh].
        """
        bdir = self.betas_dir(subject_idx)
        if hemisphere in ("lh", "both"):
            lh = self._load_mgh(bdir / "lh.ncsnr.mgh").astype(np.float32)
        if hemisphere in ("rh", "both"):
            rh = self._load_mgh(bdir / "rh.ncsnr.mgh").astype(np.float32)
        if hemisphere == "lh":
            return lh
        if hemisphere == "rh":
            return rh
        return np.concatenate([lh, rh], axis=0)

    def noise_ceiling(self, subject_idx: int, hemisphere: str = "both") -> np.ndarray:
        """Convert ncsnr to noise ceiling (NC%) per the NSD paper.

        NC = 100 * ncsnr^2 / (ncsnr^2 + 1/n_reps)
        Using n_reps=3 (NSD shared1000 each subject sees 3 times).
        """
        nc_snr = self.load_ncsnr(subject_idx, hemisphere)
        nc = 100.0 * (nc_snr ** 2) / (nc_snr ** 2 + 1.0 / N_REPS_PER_STIMULUS)
        return nc

    def load_shared1000_betas(
        self,
        subject_idx: int,
        average_reps: bool = True,
        hemisphere: str = "both",
        max_sessions: Optional[int] = None,
        verbose: bool = True,
        allow_hemisphere_fallback: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load betas for the shared1000 stimuli, averaging across reps.

        Args:
            subject_idx: 0..7
            average_reps: average across 3 reps per stimulus (NSD standard).
                          If False, returns concatenated trials.
            hemisphere: 'lh', 'rh', or 'both'
            max_sessions: optional cap on number of sessions to load (for quick tests).

        Returns:
            betas: (n_stim, n_vertices) if average_reps else (n_trials, n_vertices)
            nsd_ids: (n_stim,) ordered same as betas first axis
        """
        shared_map = self.expdesign.shared_trial_indices(subject_idx)
        # Sort by NSD ID for reproducibility
        nsd_ids_sorted = np.array(sorted(shared_map.keys()))

        # Determine sessions needed (only count sessions with required hemisphere(s))
        available_sessions = self.available_sessions(subject_idx, hemisphere=hemisphere)
        if not available_sessions and allow_hemisphere_fallback and hemisphere == "both":
            # Fall back to whichever hemisphere has sessions
            lh_sess = self.available_sessions(subject_idx, hemisphere="lh")
            rh_sess = self.available_sessions(subject_idx, hemisphere="rh")
            if lh_sess and not rh_sess:
                logger.warning(
                    f"subj{subject_idx+1}: 'both' requested but only LH available "
                    f"({len(lh_sess)} sessions). Falling back to LH-only."
                )
                hemisphere = "lh"
                available_sessions = lh_sess
            elif rh_sess and not lh_sess:
                logger.warning(
                    f"subj{subject_idx+1}: 'both' requested but only RH available. "
                    f"Falling back to RH-only."
                )
                hemisphere = "rh"
                available_sessions = rh_sess
        if max_sessions is not None:
            available_sessions = available_sessions[:max_sessions]
        if not available_sessions:
            raise FileNotFoundError(
                f"No fsaverage beta sessions found for subject {subject_idx+1} in "
                f"{self.betas_dir(subject_idx)}"
            )
        last_avail_trial = max(available_sessions) * N_TRIALS_PER_SESSION - 1

        # Build mapping: session -> [(trial_in_session, nsd_id, rep_idx)]
        session_to_load: dict[int, list[tuple[int, int, int]]] = {}
        used_trial_counts: dict[int, int] = {}  # nsd_id -> rep counter
        for nsd_id in nsd_ids_sorted:
            for trial_idx in shared_map[int(nsd_id)]:
                if trial_idx > last_avail_trial:
                    continue
                session = trial_idx // N_TRIALS_PER_SESSION + 1  # 1-indexed
                if session not in available_sessions:
                    continue
                trial_in_session = trial_idx % N_TRIALS_PER_SESSION
                rep = used_trial_counts.get(int(nsd_id), 0)
                used_trial_counts[int(nsd_id)] = rep + 1
                session_to_load.setdefault(session, []).append(
                    (trial_in_session, int(nsd_id), rep)
                )

        # Pre-allocate output
        n_vertices_total = (
            N_VERTICES_PER_HEMI * (2 if hemisphere == "both" else 1)
        )
        nsd_id_to_row = {int(nid): r for r, nid in enumerate(nsd_ids_sorted)}

        if average_reps:
            betas = np.zeros((len(nsd_ids_sorted), n_vertices_total), dtype=np.float32)
            counts = np.zeros(len(nsd_ids_sorted), dtype=np.int32)
        else:
            # Allocate enough rows for all trials (up to 3x stimuli)
            betas = np.zeros((len(nsd_ids_sorted) * N_REPS_PER_STIMULUS, n_vertices_total),
                             dtype=np.float32)
            out_nsd_ids_seq: list[int] = []

        # Load session by session to keep memory in check
        for sess in sorted(session_to_load.keys()):
            if verbose:
                logger.info(
                    f"  subj{subject_idx+1} session {sess:02d}: "
                    f"loading {len(session_to_load[sess])} trials"
                )
            sess_betas = self.load_session_betas(subject_idx, sess, hemisphere)
            # fsaverage betas are stored as float32 already in % BOLD units —
            # no rescaling needed (unlike the func1pt8mm int16 variant which
            # is stored as int16 * 300).

            for trial_in_session, nsd_id, rep in session_to_load[sess]:
                trial_vec = sess_betas[:, trial_in_session]
                if average_reps:
                    row = nsd_id_to_row[nsd_id]
                    betas[row] += trial_vec
                    counts[row] += 1
                else:
                    betas[len(out_nsd_ids_seq)] = trial_vec
                    out_nsd_ids_seq.append(nsd_id)

        if average_reps:
            # Average; drop stimuli with no available reps (they would be all-zeros
            # and confuse downstream encoding models).
            missing = counts == 0
            if missing.any():
                logger.warning(
                    f"subj{subject_idx+1}: dropping {missing.sum()}/{len(missing)} "
                    f"shared1000 stimuli with no available reps (sessions missing)"
                )
            counts_safe = np.where(counts == 0, 1, counts)
            betas /= counts_safe[:, None]
            keep_mask = ~missing
            return betas[keep_mask], nsd_ids_sorted[keep_mask]

        # not averaging: trim to actually-loaded trials
        betas = betas[: len(out_nsd_ids_seq)]
        return betas, np.array(out_nsd_ids_seq)

    # ---------- ROI masking ----------

    def fsaverage_label_dir(self) -> Path:
        return self.nsd_root / "nsddata/freesurfer/fsaverage/label"

    def load_label_mgz(self, name: str, hemisphere: str) -> np.ndarray:
        """Load a parcellation .mgz file from the fsaverage label dir.

        Args:
            name: 'HCP_MMP1', 'Kastner2015', 'streams', 'nsdgeneral', or
                  'probmap_FFA-1' etc.
            hemisphere: 'lh' or 'rh'
        """
        cache_key = f"{hemisphere}.{name}"
        if cache_key in self._roi_cache:
            return self._roi_cache[cache_key]
        path = self.fsaverage_label_dir() / f"{hemisphere}.{name}.mgz"
        if not path.exists():
            raise FileNotFoundError(f"ROI label file not found: {path}")
        arr = self._load_mgh(path)
        self._roi_cache[cache_key] = arr
        return arr

    def roi_mask_hcp(
        self,
        parcel_indices: Iterable[int],
        hemisphere: str = "both",
    ) -> np.ndarray:
        """Return boolean mask over (concat-lh-rh) vertices for HCP-MMP parcels.

        Args:
            parcel_indices: integers 1..180 (per-hemisphere parcel indices).
                            Same index used for lh.HCP_MMP1 and rh.HCP_MMP1 since
                            both files use 1-180 labeling internally.
            hemisphere: 'lh', 'rh', or 'both'
        """
        idx = list(parcel_indices)
        if hemisphere in ("lh", "both"):
            lh = self.load_label_mgz("HCP_MMP1", "lh")
            mask_lh = np.isin(lh, idx)
        if hemisphere in ("rh", "both"):
            rh = self.load_label_mgz("HCP_MMP1", "rh")
            mask_rh = np.isin(rh, idx)
        if hemisphere == "lh":
            return mask_lh
        if hemisphere == "rh":
            return mask_rh
        return np.concatenate([mask_lh, mask_rh], axis=0)

    def roi_mask_kastner(
        self,
        roi_indices: Iterable[int],
        hemisphere: str = "both",
    ) -> np.ndarray:
        """Kastner2015 visual ROIs (V1v=1, V1d=2, V2v=3, V2d=4, V3v=5, V3d=6,
        hV4=7, VO1=8, VO2=9, PHC1=10, PHC2=11, TO2=12, TO1=13, LO2=14, LO1=15,
        V3B=16, V3A=17, IPS0=18..IPS5=23, SPL1=24, FEF=25).
        """
        return self._select_label("Kastner2015", roi_indices, hemisphere)

    def roi_mask_streams(
        self,
        stream_indices: Iterable[int],
        hemisphere: str = "both",
    ) -> np.ndarray:
        """NSD visual streams: early=1, midventral=2, midlateral=3, midparietal=4,
        ventral=5, lateral=6, parietal=7.
        """
        return self._select_label("streams", stream_indices, hemisphere)

    def roi_mask_probmap(
        self,
        roi_name: str,
        threshold: float = 0.5,
        hemisphere: str = "both",
    ) -> np.ndarray:
        """Load a probmap_X.mgz and threshold to obtain a boolean mask."""
        if hemisphere in ("lh", "both"):
            lh = self.load_label_mgz(f"probmap_{roi_name}", "lh")
            mask_lh = lh > threshold
        if hemisphere in ("rh", "both"):
            rh = self.load_label_mgz(f"probmap_{roi_name}", "rh")
            mask_rh = rh > threshold
        if hemisphere == "lh":
            return mask_lh
        if hemisphere == "rh":
            return mask_rh
        return np.concatenate([mask_lh, mask_rh], axis=0)

    def roi_mask_nsdgeneral(self, hemisphere: str = "both") -> np.ndarray:
        """The NSD-general high-quality cortical mask (binary)."""
        if hemisphere in ("lh", "both"):
            lh = self.load_label_mgz("nsdgeneral", "lh")
            mask_lh = lh > 0.5
        if hemisphere in ("rh", "both"):
            rh = self.load_label_mgz("nsdgeneral", "rh")
            mask_rh = rh > 0.5
        if hemisphere == "lh":
            return mask_lh
        if hemisphere == "rh":
            return mask_rh
        return np.concatenate([mask_lh, mask_rh], axis=0)

    def _select_label(
        self,
        atlas: str,
        indices: Iterable[int],
        hemisphere: str,
    ) -> np.ndarray:
        idx = list(indices)
        if hemisphere in ("lh", "both"):
            lh = self.load_label_mgz(atlas, "lh")
            mask_lh = np.isin(lh, idx)
        if hemisphere in ("rh", "both"):
            rh = self.load_label_mgz(atlas, "rh")
            mask_rh = np.isin(rh, idx)
        if hemisphere == "lh":
            return mask_lh
        if hemisphere == "rh":
            return mask_rh
        return np.concatenate([mask_lh, mask_rh], axis=0)


# -------- Canonical NeuroLens ROI definitions on fsaverage --------

# HCP-MMP1 parcel name -> integer label (from NSD's HCP_MMP1.mgz.ctab).
# Each hemisphere file uses 1..180. roi_mask_hcp(hemisphere='both') tests
# membership against lh and rh independently using the same indices, so
# this dict is hemisphere-agnostic.
HCP_NAME_TO_IDX: dict[str, int] = {
    "V1": 1, "MST": 2, "V6": 3, "V2": 4, "V3": 5, "V4": 6, "V8": 7,
    "4": 8, "3b": 9, "FEF": 10, "PEF": 11, "55b": 12, "V3A": 13,
    "RSC": 14, "POS2": 15, "V7": 16, "IPS1": 17, "FFC": 18, "V3B": 19,
    "LO1": 20, "LO2": 21, "PIT": 22, "MT": 23, "A1": 24, "PSL": 25,
    "SFL": 26, "PCV": 27, "STV": 28, "7Pm": 29, "7m": 30, "POS1": 31,
    "44": 74, "45": 75, "47l": 76, "a47r": 77, "6r": 78, "IFJa": 79,
    "IFJp": 80, "IFSp": 81, "IFSa": 82, "p9-46v": 83, "46": 84,
    "PFt": 116, "PF": 148, "PFm": 149, "PGi": 150, "PGs": 151, "PFop": 147,
    "TGd": 131, "TE1a": 132, "TE1p": 133, "TGv": 172,
    "STGa": 123, "PBelt": 124, "A5": 125,
    "STSda": 128, "STSdp": 129, "STSvp": 130, "STSva": 176,
    "TPOJ1": 139, "TPOJ2": 140, "TPOJ3": 141,
    "PHT": 137, "PH": 138, "VVC": 163,
    "A4": 175, "MBelt": 173, "LBelt": 174,
    "TE2a": 134, "TE2p": 136, "TE1m": 177,
}


def hcp_indices(names: Iterable[str]) -> list[int]:
    """Convert a list of HCP-MMP1 parcel names to integer indices."""
    out = []
    missing = []
    for n in names:
        if n in HCP_NAME_TO_IDX:
            out.append(HCP_NAME_TO_IDX[n])
        else:
            missing.append(n)
    if missing:
        logger.warning(f"Unknown HCP parcel names: {missing}")
    return out


# Canonical NeuroLens ROI groups (computed on fsaverage)
NEUROLENS_ROIS: dict[str, dict] = {
    # --- Stage 1: Early visual ---
    "V1": {
        "atlas": "kastner",
        "indices": [1, 2],             # V1v + V1d
        "stage": 1,
        "description": "Primary visual cortex",
    },
    "V2": {
        "atlas": "kastner",
        "indices": [3, 4],
        "stage": 1,
        "description": "V2",
    },
    "V3": {
        "atlas": "kastner",
        "indices": [5, 6],
        "stage": 1,
        "description": "V3",
    },
    "V4": {
        "atlas": "kastner",
        "indices": [7],                # hV4
        "stage": 1,
        "description": "V4 (hV4)",
    },
    "early_visual": {
        "atlas": "streams",
        "indices": [1],                # streams.early
        "stage": 1,
        "description": "NSD early stream",
    },
    # --- Stage 1.5: Object-selective higher-level vision ---
    "FFA": {
        "atlas": "probmap",
        "name": "FFA-1",
        "threshold": 0.5,
        "stage": 1.5,
        "description": "Fusiform face area",
    },
    "PPA": {
        "atlas": "probmap",
        "name": "PPA",
        "threshold": 0.5,
        "stage": 1.5,
        "description": "Parahippocampal place area",
    },
    "EBA": {
        "atlas": "probmap",
        "name": "EBA",
        "threshold": 0.5,
        "stage": 1.5,
        "description": "Extrastriate body area",
    },
    "VWFA": {
        "atlas": "probmap",
        "name": "VWFA-1",
        "threshold": 0.5,
        "stage": 1.5,
        "description": "Visual word form area",
    },
    "ventral_stream": {
        "atlas": "streams",
        "indices": [2, 5],             # midventral + ventral
        "stage": 1.5,
        "description": "Ventral visual stream",
    },
    # --- Stage 2: Multimodal integration / lateral / parietal ---
    "STS": {
        "atlas": "hcp",
        "names": ["STSdp", "STSda", "STSvp", "STSva"],
        "stage": 2,
        "description": "Superior temporal sulcus",
    },
    "AG": {
        "atlas": "hcp",
        "names": ["PGi", "PGs", "PFm"],
        "stage": 2,
        "description": "Angular gyrus / IPL",
    },
    "TPOJ": {
        "atlas": "hcp",
        "names": ["TPOJ1", "TPOJ2", "TPOJ3"],
        "stage": 2,
        "description": "Temporo-parietal-occipital junction",
    },
    "lateral_stream": {
        "atlas": "streams",
        "indices": [3, 6],             # midlateral + lateral
        "stage": 2,
        "description": "Lateral visual stream",
    },
    "parietal_stream": {
        "atlas": "streams",
        "indices": [4, 7],
        "stage": 2,
        "description": "Parietal stream",
    },
    # --- Stage 3: Language network (left-lateralized) ---
    "Broca": {
        "atlas": "hcp",
        "names": ["44", "45"],
        "hemisphere": "lh",
        "stage": 3,
        "description": "Broca's area (BA44/45) — left only",
    },
    "IFG_extended": {
        "atlas": "hcp",
        "names": ["IFJa", "IFJp", "IFSa", "IFSp"],
        "hemisphere": "lh",
        "stage": 3,
        "description": "Extended inferior frontal gyrus",
    },
    "auditory_assoc": {
        "atlas": "hcp",
        "names": ["A4", "A5", "PSL", "STGa"],
        "stage": 3,
        "description": "Auditory association + Wernicke",
    },
    "temporal_pole": {
        "atlas": "hcp",
        "names": ["TGd", "TGv", "TE1a"],
        "stage": 3,
        "description": "Anterior temporal pole",
    },
}


def get_roi_mask(
    loader: NSDFsaverageLoader,
    roi_name: str,
    hemisphere: Optional[str] = None,
) -> np.ndarray:
    """Convenience: get fsaverage boolean mask for a NeuroLens ROI.

    Honors per-ROI hemisphere preference unless overridden.
    """
    if roi_name not in NEUROLENS_ROIS:
        raise KeyError(f"Unknown ROI: {roi_name}. Choices: {list(NEUROLENS_ROIS)}")
    spec = NEUROLENS_ROIS[roi_name]
    hemi = hemisphere or spec.get("hemisphere", "both")

    atlas = spec["atlas"]
    if atlas == "hcp":
        return loader.roi_mask_hcp(hcp_indices(spec["names"]), hemisphere=hemi)
    if atlas == "kastner":
        return loader.roi_mask_kastner(spec["indices"], hemisphere=hemi)
    if atlas == "streams":
        return loader.roi_mask_streams(spec["indices"], hemisphere=hemi)
    if atlas == "probmap":
        return loader.roi_mask_probmap(
            spec["name"],
            threshold=spec.get("threshold", 0.5),
            hemisphere=hemi,
        )
    raise ValueError(f"Unknown atlas: {atlas}")


def get_roi_voxels(
    loader: NSDFsaverageLoader,
    subject_idx: int,
    roi_name: str,
    max_sessions: Optional[int] = None,
    allow_hemisphere_fallback: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load shared1000 averaged betas for a single ROI of a single subject.

    Returns:
        voxel_data: (n_stim_with_data, n_vertices_in_roi) float32
        nsd_ids: (n_stim_with_data,) int
        ncsnr: (n_vertices_in_roi,) float — for noise-ceiling estimation
    """
    spec = NEUROLENS_ROIS[roi_name]
    hemi_request = spec.get("hemisphere", "both")

    # Determine what hemisphere(s) are actually available
    hemi_used = hemi_request
    if allow_hemisphere_fallback and hemi_request == "both":
        lh_ok = bool(loader.available_sessions(subject_idx, hemisphere="lh"))
        rh_ok = bool(loader.available_sessions(subject_idx, hemisphere="rh"))
        if lh_ok and not rh_ok:
            hemi_used = "lh"
        elif rh_ok and not lh_ok:
            hemi_used = "rh"

    full_betas, nsd_ids = loader.load_shared1000_betas(
        subject_idx,
        average_reps=True,
        hemisphere=hemi_used,
        max_sessions=max_sessions,
        allow_hemisphere_fallback=allow_hemisphere_fallback,
    )
    mask = get_roi_mask(loader, roi_name, hemisphere=hemi_used)
    if mask.shape[0] != full_betas.shape[1]:
        raise ValueError(
            f"ROI mask size {mask.shape[0]} != beta vertex dim {full_betas.shape[1]} "
            f"for hemisphere={hemi_used} (requested {hemi_request})"
        )
    nc = loader.load_ncsnr(subject_idx, hemisphere=hemi_used)
    return full_betas[:, mask], nsd_ids, nc[mask]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test NSD fsaverage loader")
    parser.add_argument("--nsd-root", required=True)
    parser.add_argument("--subject", type=int, default=1, help="1..8")
    parser.add_argument("--roi", default="V1")
    parser.add_argument("--max-sessions", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    loader = NSDFsaverageLoader(args.nsd_root)
    sessions = loader.available_sessions(args.subject - 1)
    print(f"Available sessions for subj{args.subject:02d}: {len(sessions)} -> {sessions}")

    voxels, ids, nc = get_roi_voxels(
        loader, args.subject - 1, args.roi, max_sessions=args.max_sessions
    )
    print(f"ROI {args.roi}: voxels={voxels.shape}, NSD IDs sample={ids[:5]}, NC mean={nc.mean():.3f}")
