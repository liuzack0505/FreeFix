import os
from typing import Any, Dict, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from recon.datasets.colmap import Parser as ColmapParser


class Parser:
    """Parser for custom layout:
    base_dir/
      colmap/
      model.ply
    """

    def __init__(
        self,
        base_dir: str,
        factor: int = 1,
        normalize: bool = False,
        train_every: int = 8,
    ):
        self.base_dir = base_dir
        self.factor = factor
        self.normalize = normalize
        self.train_every = max(int(train_every), 1)

        base_dir = os.path.abspath(base_dir)
        colmap_dir = os.path.join(base_dir, "colmap_project")
        assert os.path.isdir(
            colmap_dir
        ), f"Missing COLMAP folder: {colmap_dir}"

        colmap_parser = ColmapParser(
            data_dir=colmap_dir,
            factor=factor,
            normalize=normalize,
            test_every=self.train_every,
        )

        self.image_names = colmap_parser.image_names
        self.image_paths = colmap_parser.image_paths
        self.camtoworlds = colmap_parser.camtoworlds
        self.camera_ids = colmap_parser.camera_ids
        self.Ks_dict = colmap_parser.Ks_dict
        self.params_dict = colmap_parser.params_dict
        self.imsize_dict = colmap_parser.imsize_dict
        self.points = colmap_parser.points
        self.points_err = colmap_parser.points_err
        self.points_rgb = colmap_parser.points_rgb
        self.point_indices = colmap_parser.point_indices
        self.transform = colmap_parser.transform
        self.mapx_dict = colmap_parser.mapx_dict
        self.mapy_dict = colmap_parser.mapy_dict
        self.roi_undist_dict = colmap_parser.roi_undist_dict
        self.scene_scale = colmap_parser.scene_scale

        all_indices = np.arange(len(self.image_names))
        self.train_indices = all_indices[:: self.train_every]
        self.test_indices = np.setdiff1d(all_indices, self.train_indices)

        self.train_c2ws = self.camtoworlds[self.train_indices]
        self.test_c2ws = self.camtoworlds[self.test_indices]

        print(
            f"[CustomColmapParser] {len(self.train_indices)} train images "
            f"(1/{self.train_every}), {len(self.test_indices)} test images."
        )


class Dataset:
    """Dataset backed by custom COLMAP parser."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
        partition_file: Optional[str] = None,  # unused, kept for API parity
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size

        if load_depths:
            print(
                "[EngineCameraDataset] Warning: load_depths=True is not "
                "supported for this parser; depths will be omitted."
            )

        if split == "train":
            self.indices = parser.train_indices
        else:
            self.indices = parser.test_indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()
        params = self.parser.params_dict[camera_id]
        camtoworld = self.parser.camtoworlds[index]

        if len(params) > 0:
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y: y + h, x: x + w]

        if self.patch_size is not None:
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y: y + self.patch_size, x: x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        return {
            "K":          torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworld).float(),
            "image":      torch.from_numpy(image).float(),
            "image_id":   item,
        }
