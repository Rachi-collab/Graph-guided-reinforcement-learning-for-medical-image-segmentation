"""
Medical image preprocessing utilities for 3D/2D volumetric scans.
Provides intensity normalization, voxel resampling, 3D patch extraction, and slice slicing.
"""

import numpy as np
from scipy.ndimage import zoom
from typing import Tuple, Optional, List, Dict


class MedicalImagePreprocessor:
    """
    Preprocessor for medical scans (MRI/CT).
    Handles intensity normalization, isotropic spatial resampling, and 3D patch extraction.
    """

    def __init__(
        self,
        target_voxel_spacing: Optional[Tuple[float, float, float]] = None,
        normalize_mode: str = "minmax",
        clip_percentiles: Tuple[float, float] = (1.0, 99.0)
    ):
        """
        Args:
            target_voxel_spacing: Target physical voxel dimensions in mm (dz, dy, dx).
            normalize_mode: 'minmax' or 'zscore'.
            clip_percentiles: Lower and upper percentiles for intensity clipping to eliminate extreme outliers.
        """
        self.target_voxel_spacing = target_voxel_spacing
        self.normalize_mode = normalize_mode
        self.clip_percentiles = clip_percentiles

    def normalize_intensity(self, volume: np.ndarray) -> np.ndarray:
        """
        Clips extreme intensity outliers and normalizes volume to zero-mean unit-variance or [0, 1].
        """
        vol = volume.astype(np.float32)

        # 1. Percentile Clipping
        p_low, p_high = np.percentile(vol, self.clip_percentiles)
        vol = np.clip(vol, p_low, p_high)

        # 2. Rescaling
        if self.normalize_mode == "zscore":
            mean = np.mean(vol)
            std = np.std(vol)
            if std > 1e-7:
                vol = (vol - mean) / std
            else:
                vol = vol - mean
        elif self.normalize_mode == "minmax":
            vmin = np.min(vol)
            vmax = np.max(vol)
            if vmax - vmin > 1e-7:
                vol = (vol - vmin) / (vmax - vmin)
            else:
                vol = np.zeros_like(vol)

        return vol

    def resample_volume(
        self,
        volume: np.ndarray,
        current_spacing: Tuple[float, float, float],
        order: int = 1
    ) -> np.ndarray:
        """
        Resamples a 3D medical volume to target isotropic spacing using interpolation.
        
        Args:
            volume: 3D numpy array.
            current_spacing: Tuple (dz, dy, dx) of original spacing.
            order: Interpolation order (1 for trilinear image interpolation, 0 for nearest mask interpolation).
        """
        if self.target_voxel_spacing is None:
            return volume

        zoom_factors = [
            curr / target
            for curr, target in zip(current_spacing, self.target_voxel_spacing)
        ]
        resampled = zoom(volume, zoom_factors, order=order)
        return resampled

    def extract_3d_patches(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None,
        patch_size: Tuple[int, int, int] = (32, 32, 32),
        stride: Tuple[int, int, int] = (16, 16, 16)
    ) -> List[Dict[str, np.ndarray]]:
        """
        Extracts overlapping 3D sliding window patches from a volume for efficient training.
        """
        d, h, w = volume.shape
        pd, ph, pw = patch_size
        sd, sh, sw = stride

        patches = []
        for z in range(0, max(1, d - pd + 1), sd):
            for y in range(0, max(1, h - ph + 1), sh):
                for x in range(0, max(1, w - pw + 1), sw):
                    z_end = min(z + pd, d)
                    y_end = min(y + ph, h)
                    x_end = min(x + pw, w)

                    vol_patch = volume[z:z_end, y:y_end, x:x_end]
                    # Pad if patch is smaller than target patch_size at volume boundaries
                    pad_z = pd - vol_patch.shape[0]
                    pad_y = ph - vol_patch.shape[1]
                    pad_x = pw - vol_patch.shape[2]

                    if pad_z > 0 or pad_y > 0 or pad_x > 0:
                        vol_patch = np.pad(vol_patch, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='edge')

                    patch_item = {
                        "volume_patch": vol_patch,
                        "coords": (z, y, x)
                    }

                    if mask is not None:
                        mask_patch = mask[z:z_end, y:y_end, x:x_end]
                        if pad_z > 0 or pad_y > 0 or pad_x > 0:
                            mask_patch = np.pad(mask_patch, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
                        patch_item["mask_patch"] = mask_patch

                    patches.append(patch_item)

        return patches

    def extract_slice(self, volume: np.ndarray, slice_idx: int, axis: str = "axial") -> np.ndarray:
        """
        Extracts a 2D slice along specified anatomical plane: 'axial' (Z), 'coronal' (Y), 'sagittal' (X).
        """
        if axis == "axial":
            return volume[slice_idx, :, :]
        elif axis == "coronal":
            return volume[:, slice_idx, :]
        elif axis == "sagittal":
            return volume[:, :, slice_idx]
        else:
            raise ValueError(f"Unknown axis plane '{axis}'. Choose from 'axial', 'coronal', 'sagittal'.")
