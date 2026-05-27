"""NSD (Natural Scenes Dataset) data loading and ROI extraction.

Loads GLMsingle betas, extracts ROI masks using Glasser/FreeSurfer parcellation,
maps stimuli to COCO image IDs, and computes brain RDMs per ROI.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


class NSDLoader:
    """Load and manage NSD data for brain-model alignment analysis."""

    def __init__(self, nsd_root: str, rois_config: str = "configs/rois.yaml"):
        self.nsd_root = Path(nsd_root)
        with open(rois_config) as f:
            self.roi_config = yaml.safe_load(f)
        self._stim_info = None
        self._expdesign = None

    @property
    def stim_info(self) -> pd.DataFrame:
        """Load NSD stimulus info (maps nsdId to cocoId)."""
        if self._stim_info is None:
            path = self.nsd_root / "nsddata" / "experiments" / "nsd" / "nsd_stim_info_merged.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"NSD stim info not found at {path}. "
                    "Download from https://naturalscenesdataset.org/"
                )
            self._stim_info = pd.read_csv(path)
            logger.info(f"Loaded stim info: {len(self._stim_info)} stimuli")
        return self._stim_info

    @property
    def expdesign(self) -> dict:
        """Load NSD experimental design (which stimuli each subject saw)."""
        if self._expdesign is None:
            path = self.nsd_root / "nsddata" / "experiments" / "nsd" / "nsd_expdesign.mat"
            if path.exists():
                import h5py
                with h5py.File(path, 'r') as f:
                    self._expdesign = {
                        'masterordering': np.array(f['masterordering']).flatten(),
                        'subjectim': np.array(f['subjectim']),
                        'sharedix': np.array(f['sharedix']).flatten(),
                    }
            else:
                # Try .csv fallback
                csv_path = path.with_suffix('.csv')
                if csv_path.exists():
                    self._expdesign = pd.read_csv(csv_path)
                else:
                    logger.warning("NSD expdesign not found; shared image identification "
                                   "will use stim_info only")
                    self._expdesign = {}
        return self._expdesign

    def get_shared_image_ids(self) -> np.ndarray:
        """Get the 1000 NSD stimulus IDs shared across all 8 subjects.

        Returns:
            Array of nsdId values for the shared 1000 images.
        """
        if isinstance(self.expdesign, dict) and 'sharedix' in self.expdesign:
            # sharedix contains 1-indexed NSD stimulus IDs
            return self.expdesign['sharedix'].astype(int)

        # Fallback: use stim_info to find images with subject1000=1 flag
        # (NSD marks the shared 1000 with a flag in stim_info)
        if 'shared1000' in self.stim_info.columns:
            shared = self.stim_info[self.stim_info['shared1000'] == True]
            return shared['nsdId'].values

        # Second fallback: find images seen by all subjects
        # Each column subjectN contains 1 if that subject saw the image
        subj_cols = [c for c in self.stim_info.columns if c.startswith('subject') and
                     c[7:].isdigit()]
        if subj_cols:
            mask = self.stim_info[subj_cols].all(axis=1)
            return self.stim_info[mask]['nsdId'].values

        raise RuntimeError("Cannot determine shared images from available NSD metadata")

    def get_coco_ids(self, nsd_ids: np.ndarray) -> np.ndarray:
        """Map NSD stimulus IDs to COCO image IDs."""
        mapping = self.stim_info.set_index('nsdId')['cocoId']
        return np.array([mapping.get(nid, -1) for nid in nsd_ids])

    def load_betas(
        self,
        subject: str,
        nsd_ids: np.ndarray,
        roi_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Load GLMsingle betas for specific stimuli and optionally mask to an ROI.

        Args:
            subject: subject identifier, e.g. "subj01"
            nsd_ids: array of NSD stimulus IDs to load
            roi_mask: optional boolean mask over voxels to extract

        Returns:
            (n_stimuli, n_voxels) array of beta values
        """
        import nibabel as nib

        beta_dir = (self.nsd_root / "nsddata_betas" / "ppdata" / subject /
                    "func1pt8mm" / "betas_fithrf_GLMdenoise_RR")

        if not beta_dir.exists():
            raise FileNotFoundError(f"Beta directory not found: {beta_dir}")

        # NSD betas are stored per session (750 trials per session)
        # We need to map nsd_ids to (session, trial_within_session)
        # The masterordering tells us which stimulus was shown at each trial
        # For now, load all available betas and index into them

        # Load the full beta volume for this subject
        # Betas may be stored as a single HDF5 or as per-session NIfTI
        h5_path = beta_dir / f"betas_sessions.hdf5"
        if h5_path.exists():
            return self._load_betas_h5(h5_path, nsd_ids, roi_mask)

        # Per-session NIfTI loading
        return self._load_betas_nifti(beta_dir, subject, nsd_ids, roi_mask)

    def _load_betas_h5(self, h5_path: Path, nsd_ids: np.ndarray,
                       roi_mask: Optional[np.ndarray]) -> np.ndarray:
        """Load betas from HDF5 format."""
        import h5py
        with h5py.File(h5_path, 'r') as f:
            # Assume dataset 'betas' with shape (n_trials, x, y, z) or (n_trials, n_voxels)
            betas_dset = f['betas']
            n_total = betas_dset.shape[0]
            # Select trials corresponding to our nsd_ids
            # nsd_ids are 0-indexed or 1-indexed depending on NSD version
            indices = np.clip(nsd_ids - 1, 0, n_total - 1)  # convert to 0-indexed
            betas = betas_dset[indices]

        if roi_mask is not None and betas.ndim == 4:
            # betas: (n_stimuli, x, y, z), roi_mask: (x, y, z) boolean
            betas = betas[:, roi_mask]
        elif roi_mask is not None and betas.ndim == 2:
            betas = betas[:, roi_mask.flatten()]

        return betas.astype(np.float32)

    def _load_betas_nifti(self, beta_dir: Path, subject: str,
                          nsd_ids: np.ndarray,
                          roi_mask: Optional[np.ndarray]) -> np.ndarray:
        """Load betas from per-session NIfTI files."""
        import nibabel as nib

        # Build trial-to-stimulus mapping for this subject
        # Each session has 750 trials; trial order given by masterordering
        trial_to_nsd = self._get_trial_to_nsd_mapping(subject)

        # Find which (session, trial) indices correspond to our desired nsd_ids
        nsd_id_set = set(nsd_ids.tolist())
        needed_trials = {}  # session_idx -> list of (trial_within_session, output_idx)
        nsd_id_to_out_idx = {nid: i for i, nid in enumerate(nsd_ids)}

        for trial_idx, nsd_id in enumerate(trial_to_nsd):
            if nsd_id in nsd_id_set:
                session = trial_idx // 750
                trial_in_session = trial_idx % 750
                if session not in needed_trials:
                    needed_trials[session] = []
                out_idx = nsd_id_to_out_idx[nsd_id]
                needed_trials[session].append((trial_in_session, out_idx))

        # Determine shape from first available session file
        first_session = sorted(needed_trials.keys())[0]
        session_file = beta_dir / f"betas_session{first_session + 1:02d}.nii.gz"
        if not session_file.exists():
            # Try alternative naming
            session_file = beta_dir / f"betas_session{first_session + 1}.nii.gz"

        ref_img = nib.load(session_file)
        vol_shape = ref_img.shape[:3]

        if roi_mask is not None:
            n_voxels = roi_mask.sum()
        else:
            n_voxels = np.prod(vol_shape)

        result = np.zeros((len(nsd_ids), n_voxels), dtype=np.float32)

        for session_idx in tqdm(sorted(needed_trials.keys()),
                                desc=f"Loading betas for {subject}"):
            session_file = beta_dir / f"betas_session{session_idx + 1:02d}.nii.gz"
            if not session_file.exists():
                session_file = beta_dir / f"betas_session{session_idx + 1}.nii.gz"
            if not session_file.exists():
                logger.warning(f"Session file not found: {session_file}")
                continue

            img = nib.load(session_file)
            data = img.get_fdata()  # (x, y, z, n_trials_in_session)

            for trial_in_session, out_idx in needed_trials[session_idx]:
                if trial_in_session < data.shape[-1]:
                    vol = data[..., trial_in_session]
                    if roi_mask is not None:
                        result[out_idx] = vol[roi_mask]
                    else:
                        result[out_idx] = vol.flatten()

        return result

    def _get_trial_to_nsd_mapping(self, subject: str) -> np.ndarray:
        """Get the mapping from trial index to NSD stimulus ID for a subject.

        Returns array where trial_to_nsd[i] = nsdId shown at trial i.
        """
        # subjectim from expdesign: (n_subjects, n_trials)
        # or we reconstruct from session-level design files
        if isinstance(self.expdesign, dict) and 'subjectim' in self.expdesign:
            subj_idx = int(subject.replace("subj", "")) - 1
            subjectim = self.expdesign['subjectim']
            if subjectim.ndim == 2:
                return subjectim[subj_idx].astype(int)

        # Fallback: load from per-session design CSV if available
        design_dir = self.nsd_root / "nsddata" / "experiments" / "nsd"
        subj_num = int(subject.replace("subj", ""))
        ordering_file = design_dir / f"nsd_expdesign_{subject}.csv"
        if ordering_file.exists():
            df = pd.read_csv(ordering_file)
            return df['nsdId'].values

        # Last resort: assume sequential ordering
        logger.warning(f"Could not find trial ordering for {subject}, "
                       "assuming identity mapping")
        return np.arange(1, 10001)

    def load_roi_mask(
        self,
        subject: str,
        roi_name: str,
        atlas: str = "glasser",
    ) -> np.ndarray:
        """Load ROI binary mask for a specific brain region.

        Args:
            subject: e.g. "subj01"
            roi_name: ROI group name from rois.yaml (e.g. "visual_early")
            atlas: "glasser" or "destrieux"

        Returns:
            Boolean 3D array (x, y, z) marking voxels in the ROI.
        """
        import nibabel as nib
        from nilearn import image as nli

        roi_dir = (self.nsd_root / "nsddata" / "ppdata" / subject /
                   "func1pt8mm" / "roi")

        if roi_name in self.roi_config and atlas == "glasser":
            return self._load_glasser_roi(subject, roi_name)
        elif roi_name.startswith("V") and roi_name in ["V1", "V2", "V3", "V4", "hV4"]:
            return self._load_visual_roi(roi_dir, roi_name)
        else:
            return self._load_destrieux_roi(subject, roi_name)

    def _load_visual_roi(self, roi_dir: Path, roi_name: str) -> np.ndarray:
        """Load visual ROI from NSD's prf-based ROI definitions."""
        import nibabel as nib

        visual_rois_path = roi_dir / "prf-visualrois.nii.gz"
        if not visual_rois_path.exists():
            # Try alternative path
            visual_rois_path = roi_dir / "visualrois.nii.gz"

        if not visual_rois_path.exists():
            raise FileNotFoundError(f"Visual ROI file not found in {roi_dir}")

        roi_img = nib.load(visual_rois_path)
        roi_data = roi_img.get_fdata().astype(int)

        # NSD visual ROI labels:
        # 1=V1v, 2=V1d, 3=V2v, 4=V2d, 5=V3v, 6=V3d, 7=hV4
        roi_label_map = {
            "V1": [1, 2],       # V1v + V1d
            "V2": [3, 4],       # V2v + V2d
            "V3": [5, 6],       # V3v + V3d
            "V4": [7],          # hV4
            "hV4": [7],
        }

        labels = roi_label_map.get(roi_name, [])
        mask = np.isin(roi_data, labels)
        logger.info(f"Visual ROI {roi_name}: {mask.sum()} voxels")
        return mask

    def _load_glasser_roi(self, subject: str, roi_group: str) -> np.ndarray:
        """Load ROI from Glasser atlas parcellation.

        The Glasser atlas is in surface space (FreeSurfer label files).
        We need to map it to volumetric space using the subject's
        FreeSurfer registration.
        """
        import nibabel as nib

        group_config = self.roi_config.get(roi_group, {})
        hemisphere = group_config.get("hemisphere", "both")

        # Try to load a pre-computed volumetric Glasser parcellation
        roi_dir = (self.nsd_root / "nsddata" / "ppdata" / subject /
                   "func1pt8mm" / "roi")

        # Check for pre-made Glasser volume
        glasser_vol_path = roi_dir / "HCP_MMP1.nii.gz"
        if glasser_vol_path.exists():
            glasser_img = nib.load(glasser_vol_path)
            glasser_data = glasser_img.get_fdata().astype(int)
            return self._select_glasser_parcels(glasser_data, group_config)

        # If volumetric parcellation not available, try surface-based
        fs_dir = self.nsd_root / "nsddata" / "freesurfer" / subject
        lh_label = fs_dir / "label" / "lh.HCP_MMP1.mgz"
        rh_label = fs_dir / "label" / "rh.HCP_MMP1.mgz"

        if lh_label.exists() or rh_label.exists():
            logger.info(f"Surface-based Glasser labels found for {subject}. "
                        "Converting to volume...")
            return self._surface_to_volume_glasser(subject, group_config)

        # Fallback to NSD general ROI + Destrieux
        logger.warning(f"Glasser parcellation not found for {subject}. "
                       f"Falling back to NSD ROIs / Destrieux for {roi_group}")
        return self._load_fallback_roi(roi_dir, roi_group)

    def _select_glasser_parcels(self, glasser_data: np.ndarray,
                                group_config: dict) -> np.ndarray:
        """Select voxels belonging to specified Glasser parcels."""
        # Glasser atlas has 180 parcels per hemisphere (360 total)
        # Parcel naming convention varies; we use the standard integer labels
        # and map parcel names to integers via a lookup table
        glasser_parcels = group_config.get("glasser_parcels", [])

        # Flatten nested dict structure from rois.yaml
        parcel_names = []
        if isinstance(glasser_parcels, dict):
            for subgroup in glasser_parcels.values():
                if isinstance(subgroup, list):
                    parcel_names.extend(subgroup)
                else:
                    parcel_names.append(subgroup)
        elif isinstance(glasser_parcels, list):
            parcel_names = glasser_parcels

        # Load the Glasser parcel name-to-index mapping
        parcel_indices = self._get_glasser_parcel_indices(parcel_names)

        mask = np.isin(glasser_data, parcel_indices)
        logger.info(f"Glasser ROI with parcels {parcel_names}: {mask.sum()} voxels")
        return mask

    def _get_glasser_parcel_indices(self, parcel_names: list) -> list:
        """Map Glasser parcel names to integer indices.

        Standard Glasser HCP-MMP1.0 mapping. Left hemisphere parcels are 1-180,
        right hemisphere are 181-360.
        """
        # Standard Glasser parcel ordering (abbreviated)
        # Full list has 180 parcels; we include the ones used in our ROI definitions
        name_to_idx = {
            "V1": 1, "V2": 4, "V3": 5, "V4": 6,
            "V3A": 13, "V3B": 19, "V6": 3, "V6A": 14,
            "V7": 16, "V8": 7,
            "FFC": 18, "VVC": 151, "PIT": 22,
            "44": 74, "45": 75,
            "IFJa": 76, "IFJp": 77, "IFSa": 78, "IFSp": 79,
            "A4": 104, "A5": 105, "PSL": 107,
            "STSdp": 128, "STSda": 129, "STSvp": 130, "STSva": 131,
            "PGi": 94, "PGs": 95,
            "TPOJ1": 138, "TPOJ2": 139, "TPOJ3": 140,
        }
        indices = []
        for name in parcel_names:
            if name in name_to_idx:
                idx_lh = name_to_idx[name]
                idx_rh = idx_lh + 180
                indices.append(idx_lh)
                indices.append(idx_rh)
            else:
                logger.warning(f"Unknown Glasser parcel: {name}")
        return indices

    def _surface_to_volume_glasser(self, subject: str,
                                   group_config: dict) -> np.ndarray:
        """Convert surface-based Glasser labels to volumetric mask.

        This is a simplified approach using nilearn's vol_to_surf inverse.
        For production use, consider FreeSurfer's mri_label2vol.
        """
        import nibabel as nib
        from nilearn import surface

        # Load a reference functional volume for the target space
        roi_dir = (self.nsd_root / "nsddata" / "ppdata" / subject /
                   "func1pt8mm" / "roi")
        ref_path = roi_dir / "nsdgeneral.nii.gz"
        if not ref_path.exists():
            logger.warning("Cannot convert surface labels without reference volume")
            return self._load_fallback_roi(roi_dir, "nsdgeneral")

        ref_img = nib.load(ref_path)
        ref_data = ref_img.get_fdata()
        # Return the nsdgeneral mask as a rough approximation
        # until proper surface-to-volume conversion is implemented
        logger.warning("Using nsdgeneral ROI as placeholder for Glasser-based ROI. "
                       "Run FreeSurfer mri_label2vol for precise Glasser mapping.")
        return ref_data > 0

    def _load_fallback_roi(self, roi_dir: Path, roi_group: str) -> np.ndarray:
        """Load a fallback ROI when Glasser is not available."""
        import nibabel as nib

        # Use NSD's built-in ROI definitions
        general_path = roi_dir / "nsdgeneral.nii.gz"
        if general_path.exists():
            img = nib.load(general_path)
            mask = img.get_fdata() > 0
            logger.info(f"Fallback ROI (nsdgeneral): {mask.sum()} voxels")
            return mask

        raise FileNotFoundError(f"No ROI data found in {roi_dir}")

    def _load_destrieux_roi(self, subject: str, roi_name: str) -> np.ndarray:
        """Load ROI from FreeSurfer Destrieux atlas."""
        import nibabel as nib

        aparc_path = (self.nsd_root / "nsddata" / "freesurfer" / subject /
                      "mri" / "aparc.a2009s+aseg.mgz")

        if not aparc_path.exists():
            raise FileNotFoundError(f"Destrieux atlas not found: {aparc_path}")

        img = nib.load(aparc_path)
        data = img.get_fdata().astype(int)

        group_config = self.roi_config.get(roi_name, {})
        destrieux_labels = group_config.get("destrieux_labels", [])

        # Destrieux label name to FreeSurfer integer mapping
        # These are approximate; exact values depend on FreeSurfer version
        label_map = {
            "G_cuneus": [11125, 12125],
            "S_calcarine": [11145, 12145],
            "G_occipital_sup": [11109, 12109],
            "G_occipital_middle": [11111, 12111],
            "S_temporal_sup": [11165, 12165],
            "G_pariet_inf-Angular": [11131, 12131],
            "G_temp_sup-Lateral": [11133, 12133],
            "G_front_inf-Opercular": [11113, 12113],
            "G_front_inf-Triangul": [11115, 12115],
        }

        mask_indices = []
        for label_name in destrieux_labels:
            if label_name in label_map:
                mask_indices.extend(label_map[label_name])
            else:
                logger.warning(f"Unknown Destrieux label: {label_name}")

        mask = np.isin(data, mask_indices)
        logger.info(f"Destrieux ROI {roi_name}: {mask.sum()} voxels")
        return mask

    def extract_roi_data(
        self,
        subject: str,
        roi_name: str,
        nsd_ids: np.ndarray,
        atlas: str = "glasser",
    ) -> np.ndarray:
        """Extract voxel patterns within an ROI for specified stimuli.

        Args:
            subject: e.g. "subj01"
            roi_name: ROI identifier
            nsd_ids: stimulus IDs to extract
            atlas: which atlas to use

        Returns:
            (n_stimuli, n_voxels_in_roi) array
        """
        roi_mask = self.load_roi_mask(subject, roi_name, atlas)
        betas = self.load_betas(subject, nsd_ids, roi_mask)
        return betas

    def compute_brain_rdm(
        self,
        voxel_data: np.ndarray,
        method: str = "correlation",
    ) -> np.ndarray:
        """Compute brain RDM from voxel patterns.

        Args:
            voxel_data: (n_stimuli, n_voxels)
            method: distance metric

        Returns:
            (n_stimuli, n_stimuli) dissimilarity matrix
        """
        import rsatoolbox
        dataset = rsatoolbox.data.Dataset(voxel_data)
        rdm = rsatoolbox.rdm.calc_rdm(dataset, method=method)
        # Convert to full matrix
        from scipy.spatial.distance import squareform
        vec = rdm.get_vectors().flatten()
        return squareform(vec)

    def compute_all_brain_rdms(
        self,
        subjects: list[str],
        roi_names: list[str],
        nsd_ids: np.ndarray,
        output_dir: str,
        method: str = "correlation",
    ):
        """Compute and save brain RDMs for all subjects and ROIs.

        Args:
            subjects: list of subject identifiers
            roi_names: list of ROI group names
            nsd_ids: shared stimulus IDs
            output_dir: directory to save RDM matrices
            method: distance metric
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for subj in subjects:
            for roi in roi_names:
                logger.info(f"Computing brain RDM: {subj} / {roi}")
                try:
                    voxel_data = self.extract_roi_data(subj, roi, nsd_ids)
                    if voxel_data.shape[1] < 2:
                        logger.warning(f"Too few voxels for {subj}/{roi}: "
                                       f"{voxel_data.shape[1]}")
                        continue

                    rdm = self.compute_brain_rdm(voxel_data, method)
                    save_path = out_path / f"brain_rdm_{subj}_{roi}.npy"
                    np.save(save_path, rdm)

                    # Also save voxel data for encoding models
                    voxel_path = out_path / f"voxels_{subj}_{roi}.npy"
                    np.save(voxel_path, voxel_data)

                    logger.info(f"  Saved: {rdm.shape}, {voxel_data.shape[1]} voxels")
                except Exception as e:
                    logger.error(f"Failed for {subj}/{roi}: {e}")

    def get_nsd_functional_rois(self, subject: str) -> dict[str, np.ndarray]:
        """Load NSD functional localizer ROIs (FFA, PPA, EBA, etc.)."""
        import nibabel as nib

        roi_dir = (self.nsd_root / "nsddata" / "ppdata" / subject /
                   "func1pt8mm" / "roi")
        func_rois = {}

        # NSD provides functional localizer ROIs
        for roi_file in ["floc-faces.nii.gz", "floc-places.nii.gz",
                         "floc-bodies.nii.gz", "floc-words.nii.gz"]:
            path = roi_dir / roi_file
            if path.exists():
                img = nib.load(path)
                data = img.get_fdata()
                # These files contain t-stat maps; threshold at t>0
                # Different integer labels for different sub-regions
                name = roi_file.replace(".nii.gz", "").replace("floc-", "")
                func_rois[name] = data
                logger.info(f"Loaded functional ROI: {name}")

        return func_rois


def load_brain_rdm(brain_data_dir: str, subject: str, roi: str) -> np.ndarray:
    """Load a precomputed brain RDM from disk."""
    path = Path(brain_data_dir) / f"brain_rdm_{subject}_{roi}.npy"
    return np.load(path)


def load_voxel_data(brain_data_dir: str, subject: str, roi: str) -> np.ndarray:
    """Load voxel response data from disk."""
    path = Path(brain_data_dir) / f"voxels_{subject}_{roi}.npy"
    return np.load(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process NSD brain data")
    parser.add_argument("--nsd-root", required=True, help="Path to NSD data root")
    parser.add_argument("--rois-config", default="configs/rois.yaml")
    parser.add_argument("--output-dir", default="brain_data")
    parser.add_argument("--subjects", nargs="+",
                        default=[f"subj{i:02d}" for i in range(1, 9)])
    parser.add_argument("--roi-groups", nargs="+",
                        default=["visual_early", "multimodal_integration",
                                 "language_network"])

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    loader = NSDLoader(args.nsd_root, args.rois_config)
    shared_ids = loader.get_shared_image_ids()
    logger.info(f"Found {len(shared_ids)} shared images")

    loader.compute_all_brain_rdms(
        subjects=args.subjects,
        roi_names=args.roi_groups,
        nsd_ids=shared_ids,
        output_dir=args.output_dir,
    )
