"""
Unit tests for data loader, preprocessor, and anatomical graph generator.
"""

import pytest
import numpy as np
import torch
from src.data.dataset import MedicalImageDataset, generate_synthetic_scan
from src.data.preprocess import MedicalImagePreprocessor
from src.data.graph_generator import AnatomicalGraphGenerator


def test_synthetic_scan_generation():
    vol, mask = generate_synthetic_scan(shape=(32, 32, 32), seed=42)
    assert vol.shape == (32, 32, 32)
    assert mask.shape == (32, 32, 32)
    assert vol.dtype == np.float32
    assert mask.dtype == np.uint8
    assert np.min(vol) >= 0.0 and np.max(vol) <= 1.0
    assert set(np.unique(mask)).issubset({0, 1})


def test_medical_image_dataset():
    dataset = MedicalImageDataset(num_synthetic_samples=3, volume_shape=(32, 32, 32))
    assert len(dataset) == 3
    sample = dataset[0]
    assert "volume" in sample and "mask" in sample
    assert sample["volume"].shape == (1, 32, 32, 32)
    assert sample["mask"].shape == (32, 32, 32)


def test_preprocessor():
    preprocessor = MedicalImagePreprocessor(normalize_mode="minmax")
    vol, mask = generate_synthetic_scan(shape=(32, 32, 32))
    
    norm_vol = preprocessor.normalize_intensity(vol)
    assert norm_vol.shape == vol.shape
    assert np.min(norm_vol) >= 0.0 and np.max(norm_vol) <= 1.0

    patches = preprocessor.extract_3d_patches(norm_vol, mask, patch_size=(16, 16, 16), stride=(16, 16, 16))
    assert len(patches) > 0
    assert patches[0]["volume_patch"].shape == (16, 16, 16)


def test_anatomical_graph_generator_2d():
    vol, mask = generate_synthetic_scan(shape=(32, 32, 32))
    slice_2d = vol[16, :, :]
    mask_2d = mask[16, :, :]

    generator = AnatomicalGraphGenerator(n_segments=30)
    graph_data = generator.construct_graph_from_2d_slice(slice_2d, mask_2d)

    assert hasattr(graph_data, "x")
    assert hasattr(graph_data, "edge_index")
    assert hasattr(graph_data, "y")
    assert graph_data.x.dim() == 2
    assert graph_data.edge_index.shape[0] == 2
    assert len(graph_data.y) == graph_data.x.shape[0]


def test_anatomical_graph_generator_3d():
    vol, mask = generate_synthetic_scan(shape=(32, 32, 32))
    generator = AnatomicalGraphGenerator(n_segments=40)
    graph_data = generator.construct_graph_from_3d_volume(vol, mask)

    assert hasattr(graph_data, "x")
    assert hasattr(graph_data, "edge_index")
    assert graph_data.x.shape[0] > 0
    assert graph_data.edge_index.shape[0] == 2
