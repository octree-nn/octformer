# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import numpy as np

from thsolver import Dataset
from ocnn.octree import Points
from ocnn.dataset import CollateBatch

from .utils import ReadPly, Transform


class ModelNetTransform(Transform):

  def preprocess(self, sample: dict, idx: int):
    points = super().preprocess(sample, idx)

    # Comment out the following lines if the shapes have been normalized
    # in the dataset generation stage.
    #
    # Normalize the points into one unit sphere in [-0.8, 0.8]
    # bbmin, bbmax = points.bbox()
    # points.normalize(bbmin, bbmax, scale=0.8)
    #
    # points.scale(torch.Tensor([0.8, 0.8, 0.8]))

    return points


def read_file(filename: str):
  filename = filename.replace('\\', '/')
  if filename.endswith('.ply'):
    read_ply = ReadPly(has_normal=True)
    return read_ply(filename)
  elif filename.endswith('.npz'):
    raw = np.load(filename)
    output = {'points': raw['points'], 'normals': raw['normals']}
    return output
  else:
    raise ValueError


def get_modelnet40_dataset(flags):
  transform = ModelNetTransform(flags)
  collate_batch = CollateBatch()

  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_file, take=flags.take)
  return dataset, collate_batch
