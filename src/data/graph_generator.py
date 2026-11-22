"""
Anatomical Graph Generator for Medical Image Segmentation.
Converts 2D/3D medical scans into Superpixel/Supervoxel Region Adjacency Graphs (RAG)
represented as PyTorch Geometric `Data` objects.
Includes fallback grid partitioning if skimage is not installed.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional, Union, List

try:
    from skimage.segmentation import slic
    from skimage.graph import region_adjacency
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    class Data:
        """Fallback Data object if torch_geometric is not installed."""
        def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None):
            self.x = x
            self.edge_index = edge_index
            self.edge_attr = edge_attr
            self.y = y
            self.pos = pos


class AnatomicalGraphGenerator:
    """
    Constructs graph representations of medical images using SLIC supervoxels/superpixels.
    Each node represents an anatomical region; edges capture spatial adjacency and feature similarity.
    """

    def __init__(
        self,
        n_segments: int = 100,
        compactness: float = 10.0,
        sigma: float = 1.0,
        start_label: int = 0
    ):
        self.n_segments = n_segments
        self.compactness = compactness
        self.sigma = sigma
        self.start_label = start_label

    def _grid_segmentation_2d(self, shape: Tuple[int, int]) -> np.ndarray:
        """Fallback grid segmentation if skimage is unavailable."""
        H, W = shape
        grid_size = max(1, int(np.sqrt((H * W) / self.n_segments)))
        segments = np.zeros((H, W), dtype=int)
        lbl = 0
        for i in range(0, H, grid_size):
            for j in range(0, W, grid_size):
                segments[i:min(i + grid_size, H), j:min(j + grid_size, W)] = lbl
                lbl += 1
        return segments

    def _grid_segmentation_3d(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Fallback grid segmentation for 3D volumes."""
        D, H, W = shape
        grid_size = max(1, int(np.cbrt((D * H * W) / self.n_segments)))
        segments = np.zeros((D, H, W), dtype=int)
        lbl = 0
        for k in range(0, D, grid_size):
            for i in range(0, H, grid_size):
                for j in range(0, W, grid_size):
                    segments[k:min(k + grid_size, D), i:min(i + grid_size, H), j:min(j + grid_size, W)] = lbl
                    lbl += 1
        return segments

    def construct_graph_from_2d_slice(
        self,
        image_slice: np.ndarray,
        mask_slice: Optional[np.ndarray] = None
    ) -> Data:
        H, W = image_slice.shape

        if HAS_SKIMAGE:
            segments = slic(
                image_slice,
                n_segments=self.n_segments,
                compactness=self.compactness,
                sigma=self.sigma,
                start_label=self.start_label,
                channel_axis=None
            )
        else:
            segments = self._grid_segmentation_2d((H, W))

        unique_labels = np.unique(segments)
        label_to_node_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        N = len(unique_labels)

        node_features = []
        node_positions = []
        node_labels = []

        for lbl in unique_labels:
            region_mask = (segments == lbl)
            intensities = image_slice[region_mask]

            mean_intensity = np.mean(intensities) if len(intensities) > 0 else 0.0
            std_intensity = np.std(intensities) if len(intensities) > 1 else 0.0
            area_ratio = np.sum(region_mask) / (H * W)

            coords = np.argwhere(region_mask)
            if len(coords) > 0:
                cy, cx = np.mean(coords, axis=0)
            else:
                cy, cx = 0.0, 0.0
            norm_cy, norm_cx = cy / H, cx / W

            feat = [mean_intensity, std_intensity, norm_cy, norm_cx, area_ratio]
            node_features.append(feat)
            node_positions.append([norm_cx, norm_cy])

            if mask_slice is not None and len(intensities) > 0:
                mask_pixels = mask_slice[region_mask]
                label_val = 1 if np.mean(mask_pixels) >= 0.3 else 0
            else:
                label_val = 0
            node_labels.append(label_val)

        x = torch.tensor(node_features, dtype=torch.float)
        pos = torch.tensor(node_positions, dtype=torch.float)
        y = torch.tensor(node_labels, dtype=torch.long)

        # Adjacency
        src_list, dst_list, edge_feats = [], [], []

        diff_y = segments[:-1, :] != segments[1:, :]
        diff_x = segments[:, :-1] != segments[:, 1:]

        adjacent_pairs = set()
        for u, v in zip(segments[:-1, :][diff_y], segments[1:, :][diff_y]):
            if u != v: adjacent_pairs.add(tuple(sorted((u, v))))
        for u, v in zip(segments[:, :-1][diff_x], segments[:, 1:][diff_x]):
            if u != v: adjacent_pairs.add(tuple(sorted((u, v))))

        for u, v in adjacent_pairs:
            if u in label_to_node_idx and v in label_to_node_idx:
                u_idx = label_to_node_idx[u]
                v_idx = label_to_node_idx[v]

                dist = float(np.linalg.norm(pos[u_idx].numpy() - pos[v_idx].numpy()))
                intensity_diff = float(abs(x[u_idx, 0].item() - x[v_idx, 0].item()))

                src_list.extend([u_idx, v_idx])
                dst_list.extend([v_idx, u_idx])
                edge_feats.extend([[dist, intensity_diff], [dist, intensity_diff]])

        if len(src_list) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.tensor(edge_feats, dtype=torch.float)

        graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            pos=pos
        )
        graph_data.segments = segments
        return graph_data

    def construct_graph_from_3d_volume(
        self,
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Data:
        D, H, W = volume.shape

        if HAS_SKIMAGE:
            segments = slic(
                volume,
                n_segments=self.n_segments,
                compactness=self.compactness,
                sigma=self.sigma,
                start_label=self.start_label,
                channel_axis=None
            )
        else:
            segments = self._grid_segmentation_3d((D, H, W))

        unique_labels = np.unique(segments)
        label_to_node_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        N = len(unique_labels)

        node_features = []
        node_positions = []
        node_labels = []

        total_voxels = D * H * W

        for lbl in unique_labels:
            region_mask = (segments == lbl)
            intensities = volume[region_mask]

            mean_val = np.mean(intensities) if len(intensities) > 0 else 0.0
            std_val = np.std(intensities) if len(intensities) > 1 else 0.0
            volume_ratio = np.sum(region_mask) / total_voxels

            coords = np.argwhere(region_mask)
            if len(coords) > 0:
                cz, cy, cx = np.mean(coords, axis=0)
            else:
                cz, cy, cx = 0.0, 0.0, 0.0
            norm_cz, norm_cy, norm_cx = cz / D, cy / H, cx / W

            feat = [mean_val, std_val, norm_cz, norm_cy, norm_cx, volume_ratio]
            node_features.append(feat)
            node_positions.append([norm_cx, norm_cy, norm_cz])

            if mask is not None and len(intensities) > 0:
                mask_voxels = mask[region_mask]
                label_val = 1 if np.mean(mask_voxels) >= 0.3 else 0
            else:
                label_val = 0
            node_labels.append(label_val)

        x = torch.tensor(node_features, dtype=torch.float)
        pos = torch.tensor(node_positions, dtype=torch.float)
        y = torch.tensor(node_labels, dtype=torch.long)

        src_list, dst_list, edge_feats = [], [], []

        diff_z = segments[:-1, :, :] != segments[1:, :, :]
        diff_y = segments[:, :-1, :] != segments[:, 1:, :]
        diff_x = segments[:, :, :-1] != segments[:, :, 1:]

        adjacent_pairs = set()
        for u, v in zip(segments[:-1, :, :][diff_z], segments[1:, :, :][diff_z]):
            if u != v: adjacent_pairs.add(tuple(sorted((u, v))))
        for u, v in zip(segments[:, :-1, :][diff_y], segments[:, 1:, :][diff_y]):
            if u != v: adjacent_pairs.add(tuple(sorted((u, v))))
        for u, v in zip(segments[:, :, :-1][diff_x], segments[:, :, 1:][diff_x]):
            if u != v: adjacent_pairs.add(tuple(sorted((u, v))))

        for u, v in adjacent_pairs:
            if u in label_to_node_idx and v in label_to_node_idx:
                u_idx = label_to_node_idx[u]
                v_idx = label_to_node_idx[v]

                dist = float(np.linalg.norm(pos[u_idx].numpy() - pos[v_idx].numpy()))
                intensity_diff = float(abs(x[u_idx, 0].item() - x[v_idx, 0].item()))

                src_list.extend([u_idx, v_idx])
                dst_list.extend([v_idx, u_idx])
                edge_feats.extend([[dist, intensity_diff], [dist, intensity_diff]])

        if len(src_list) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)
        else:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.tensor(edge_feats, dtype=torch.float)

        graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            pos=pos
        )
        graph_data.segments = segments
        return graph_data
