# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import numpy as np
from typing import Optional
from plyfile import PlyData, PlyElement


def save_points_to_ply(filename: str, points: np.ndarray,
                       normals: Optional[np.ndarray] = None,
                       colors: Optional[np.ndarray] = None,
                       labels: Optional[np.ndarray] = None,
                       text: bool = False):

  point_cloud = [points]
  point_cloud_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
  if normals is not None:
    point_cloud.append(normals)
    point_cloud_types += [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
  if colors is not None:
    point_cloud.append(colors)
    point_cloud_types += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
  if labels is not None:
    point_cloud.append(labels)
    point_cloud_types += [('label', 'u1')]
  point_cloud = np.concatenate(point_cloud, axis=1)

  vertices = [tuple(p) for p in point_cloud]
  structured_array = np.array(vertices, dtype=point_cloud_types)
  el = PlyElement.describe(structured_array, 'vertex')

  folder = os.path.dirname(filename)
  if not os.path.exists(folder):
    os.makedirs(folder)
  PlyData([el], text).write(filename)
