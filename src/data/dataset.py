"""
Medical Image Dataset handling and synthetic 3D scan generator for GRL segmentation pipeline.
Supports NIfTI (.nii, .nii.gz), DICOM, Numpy arrays, and synthetic medical volumes.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Optional, List, Union


def generate_synthetic_scan(
    shape: Tuple[int, int, int] = (64, 64, 64),
    center: Optional[Tuple[int, int, int]] = None,
    radius: float = 12.0,
    background_intensity: float = 0.2,
    organ_intensity: float = 0.5,
    lesion_intensity: float = 0.85,
    noise_level: float = 0.05,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a realistic synthetic 3D medical volume with background, organ, and lesion regions.
    
    Args:
        shape: 3D volume dimensions (D, H, W).
        center: Center coordinates of the synthetic lesion/tumor.
        radius: Radius of the synthetic lesion.
        background_intensity: Mean intensity for background air/bone.
        organ_intensity: Mean intensity for surrounding healthy organ tissue.
        lesion_intensity: Mean intensity for the target lesion/tumor region.
        noise_level: Standard deviation of Gaussian sensor noise.
        seed: Random seed for reproducibility.

    Returns:
        volume: 3D float32 numpy array normalized to [0, 1].
        mask: 3D uint8 numpy array binary ground-truth segmentation mask (0 or 1).
    """
    if seed is not None:
        np.random.seed(seed)

    depth, height, width = shape
    z_grid, y_grid, x_grid = np.ogrid[:depth, :height, :width]

    # 1. Create healthy organ anatomical sphere/ellipse
    organ_center = (depth // 2, height // 2, width // 2)
    organ_radius = min(shape) // 2.5
    organ_dist = (
        ((z_grid - organ_center[0]) / 1.0)**2 +
        ((y_grid - organ_center[1]) / 1.1)**2 +
        ((x_grid - organ_center[2]) / 0.9)**2
    )
    organ_mask = organ_dist <= organ_radius**2

    # 2. Create target lesion/tumor ellipse inside organ
    if center is None:
        center = (
            depth // 2 + np.random.randint(-3, 4),
            height // 2 + np.random.randint(-3, 4),
            width // 2 + np.random.randint(-3, 4)
        )
    
    lesion_dist = (
        ((z_grid - center[0]) / 1.0)**2 +
        ((y_grid - center[1]) / 0.85)**2 +
        ((x_grid - center[2]) / 1.15)**2
    )
    mask = (lesion_dist <= radius**2).astype(np.uint8)

    # 3. Assemble volume with heterogeneous intensities and texture gradient
    volume = np.full(shape, background_intensity, dtype=np.float32)
    volume[organ_mask] = organ_intensity
    volume[mask == 1] = lesion_intensity

    # Add smooth spatial intensity inhomogeneity (Rician / bias field effect)
    bias_field = 1.0 + 0.15 * np.sin(z_grid / 10.0) * np.cos(y_grid / 10.0)
    volume = volume * bias_field

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, size=shape).astype(np.float32)
    volume += noise

    # Clip to valid range [0, 1]
    volume = np.clip(volume, 0.0, 1.0).astype(np.float32)

    return volume, mask


class MedicalImageDataset(Dataset):
    """
    PyTorch Dataset for 3D/2D Medical Scans and Ground Truth Segmentation Masks.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        num_synthetic_samples: int = 10,
        volume_shape: Tuple[int, int, int] = (64, 64, 64),
        transform = None,
        seed: int = 42
    ):
        """
        Args:
            data_dir: Path to directory containing NIfTI (.nii, .nii.gz) or Numpy files.
            num_synthetic_samples: If data_dir is empty or None, fallback to generating synthetic volumes.
            volume_shape: Dimensions for synthetic volumes.
            transform: Optional transformations.
            seed: Random seed.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.items: List[Dict[str, Union[str, np.ndarray]]] = []

        if data_dir and os.path.exists(data_dir):
            file_paths = sorted(glob.glob(os.path.join(data_dir, "*.nii*")) + 
                                glob.glob(os.path.join(data_dir, "*.npy")))
            for fp in file_paths:
                if "_mask" not in fp:
                    mask_fp = fp.replace(".nii.gz", "_mask.nii.gz").replace(".nii", "_mask.nii").replace(".npy", "_mask.npy")
                    if os.path.exists(mask_fp):
                        self.items.append({"volume_path": fp, "mask_path": mask_fp, "is_synthetic": False})

        # Fallback to synthetic if no valid files found
        if len(self.items) == 0:
            print(f"[MedicalImageDataset] No real data files found in '{data_dir}'. Generating {num_synthetic_samples} synthetic scans...")
            for i in range(num_synthetic_samples):
                vol, mask = generate_synthetic_scan(shape=volume_shape, seed=seed + i)
                self.items.append({
                    "id": f"synthetic_{i:03d}",
                    "volume": vol,
                    "mask": mask,
                    "is_synthetic": True
                })

    def __len__(self) -> int:
        return len(self.items)

    def _load_nifti(self, filepath: str) -> np.ndarray:
        try:
            import nibabel as nib
            img = nib.load(filepath)
            data = img.get_fdata().astype(np.float32)
            return data
        except ImportError:
            raise ImportError("nibabel is required to load NIfTI files. Install via pip install nibabel.")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        if item["is_synthetic"]:
            vol = item["volume"]
            mask = item["mask"]
        else:
            if item["volume_path"].endswith(".npy"):
                vol = np.load(item["volume_path"]).astype(np.float32)
                mask = np.load(item["mask_path"]).astype(np.uint8)
            else:
                vol = self._load_nifti(item["volume_path"])
                mask = self._load_nifti(item["mask_path"]).astype(np.uint8)

        if self.transform is not None:
            vol, mask = self.transform(vol, mask)

        vol_tensor = torch.from_numpy(vol).unsqueeze(0)  # Shape: (1, D, H, W)
        mask_tensor = torch.from_numpy(mask).long()      # Shape: (D, H, W)

        return {
            "volume": vol_tensor,
            "mask": mask_tensor,
            "id": item.get("id", os.path.basename(item.get("volume_path", f"sample_{idx}")))
        }
