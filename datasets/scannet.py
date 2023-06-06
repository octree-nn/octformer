# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import torch
import random
import scipy.interpolate
import scipy.ndimage
import numpy as np

from ocnn.octree import Points, Octree
from ocnn.dataset import CollateBatch
from thsolver import Dataset
from typing import List

from .utils import ReadFile, Transform


def color_distort(color, trans_range_ratio, jitter_std):

  def _color_autocontrast(color):
    assert color.shape[1] >= 3
    lo = color[:, :3].min(0, keepdims=True)
    hi = color[:, :3].max(0, keepdims=True)
    assert hi.max() > 1

    scale = 255 / (hi - lo)
    contrast_feats = (color[:, :3] - lo) * scale

    blend_factor = random.random()
    color[:, :3] = (1 - blend_factor) * color + blend_factor * contrast_feats
    return color

  def _color_translation(color, trans_range_ratio=0.1):
    assert color.shape[1] >= 3
    if random.random() < 0.95:
      tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * trans_range_ratio
      color[:, :3] = np.clip(tr + color[:, :3], 0, 255)
    return color

  def _color_jiter(color, std=0.01):
    if random.random() < 0.95:
      noise = np.random.randn(color.shape[0], 3)
      noise *= std * 255
      color[:, :3] = np.clip(noise + color[:, :3], 0, 255)
    return color

  color = color * 255.0
  color = _color_autocontrast(color)
  color = _color_translation(color, trans_range_ratio)
  color = _color_jiter(color, jitter_std)
  color = color / 255.0
  return color


def elastic_distort(points, distortion_params):

  def _elastic_distort(coords, granularity, magnitude):
    blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
    blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
    blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
    coords_min = coords.min(0)

    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    convolve = scipy.ndimage.filters.convolve
    for _ in range(2):
      noise = convolve(noise, blurx, mode='constant', cval=0)
      noise = convolve(noise, blury, mode='constant', cval=0)
      noise = convolve(noise, blurz, mode='constant', cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [np.linspace(d_min, d_max, d)
          for d_min, d_max, d in zip(coords_min - granularity,
                                     coords_min + granularity*(noise_dim - 2),
                                     noise_dim)]

    interp = scipy.interpolate.RegularGridInterpolator(
        ax, noise, bounds_error=0, fill_value=0)
    coords += interp(coords) * magnitude
    return coords

  assert distortion_params.shape[1] == 2
  if random.random() < 0.95:
    for granularity, magnitude in distortion_params:
      points = _elastic_distort(points, granularity, magnitude)
  return points


def align_z(points: Points):
  points.points[:, 2] -= points.points[:, 2].min()
  return points


def rand_crop(points: Points, max_npt: int):
  r''' Keeps `max_npt` pts at most centered by a radomly chosen pts. 
  '''

  pts = points.points
  npt = points.npt
  crop_mask = torch.ones(npt, dtype=torch.bool)
  if npt > max_npt:
    rand_idx = torch.randint(low=0, high=npt, size=(1,))
    sort_idx = torch.argsort(torch.sum((pts - pts[rand_idx])**2, 1))
    crop_idx = sort_idx[max_npt:]
    crop_mask[crop_idx] = False
    points = points[crop_mask]
  return points, crop_mask


class ScanNetTransform(Transform):

  def __init__(self, flags):
    super().__init__(flags)

    # The `self.scale_factor` is used to normalize the input point cloud to the
    # range of [-1, 1]. If this parameter is modified, the `self.elastic_params`
    # and the `jittor` in the data augmentation should be scaled accordingly.
    # self.scale_factor = 5.12    # depth 9: voxel size 2cm
    self.scale_factor = 10.24     # depth 10: voxel size 2cm; depth 11: voxel size 1cm
    self.color_trans_ratio = 0.10
    self.color_jit_std = 0.05
    self.elastic_params = np.array(
        [[0.05, 0.1], [0.1, 0.2], [0.2, 0.4], [0.4, 0.8]], np.float32)

  def __call__(self, sample, idx=None):

    # normalize points
    xyz = sample['points']
    center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2.0
    xyz = (xyz - center) / self.scale_factor  # xyz in [-1, 1]

    # normalize color
    color = sample['colors'] / 255.0

    # data augmentation specific to scannet
    if self.flags.distort:
      color = color_distort(color, self.color_trans_ratio, self.color_jit_std)
      xyz = elastic_distort(xyz, self.elastic_params)

    # construct points
    points = Points(torch.from_numpy(xyz), torch.from_numpy(sample['normals']),
                    torch.from_numpy(color), torch.from_numpy(sample['labels']))

    # transform provided by `ocnn`,
    # including rotatation, translation, scaling, and flipping
    output = self.transform(points, idx)   # points and inbox_mask
    points, inbox_mask = output['points'], output['inbox_mask']

    # random crop
    if self.flags.distort:
      max_npt = self.flags.max_npt if self.flags.max_npt > 0 else points.npt
      max_npt = min(max_npt, int(points.npt * self.flags.crop_ratio))
      points, crop_mask = rand_crop(points, max_npt)
      inbox_mask[inbox_mask.clone()] = crop_mask   # update inbox_mask

    # align z
    points = align_z(points)
    return {'points': points, 'inbox_mask': inbox_mask}


def apply_cutmix(points: List[Points], cutmix: float):
  if cutmix <= 0:
    return points

  batch_size = len(points)
  outputs = [None] * batch_size
  for i in range(batch_size):
    j = (i + 1) % batch_size
    points_a = points[i]
    points_b = points[j]

    npt_a = points_a.points.shape[0]
    npt_b = points_b.points.shape[0]
    na = int(cutmix * npt_a)
    nb = int((1 - cutmix) * npt_b)

    rand_idx = torch.randint(0, npt_a, size=(1,))
    rand_pts = points_a.points[rand_idx]
    dist_a, idx_a = torch.sort(torch.sum((points_a.points - rand_pts)**2, 1))
    cut_a = idx_a[:na]

    dist_b = torch.sum((points_b.points - rand_pts)**2, 1) - dist_a[na]
    mask_b = dist_b < 0
    dist_b[mask_b] += 1.0e3
    dist_b, idx_b = torch.sort(dist_b)
    cut_b = idx_b[:nb]

    outputs[i] = ocnn.octree.merge_points(
        [points_a[cut_a], points_b[cut_b]], update_batch_info=False)
  return outputs


class CollateBatch:

  def __init__(self, cutmix: int = 0.5):
    super().__init__()
    self.cutmix = cutmix

  def __call__(self, batch: list):
    assert type(batch) == list

    # a list of dicts -> a dict of lists
    outputs = {key: [b[key] for b in batch] for key in batch[0].keys()}

    # apply cutmix
    points = outputs['points']
    if self.cutmix > 0:  # and torch.rand(1) > 0.3:
      points = apply_cutmix(points, self.cutmix)
    outputs['points'] = points
    return outputs


def get_scannet_dataset(flags):
  transform = ScanNetTransform(flags)
  read_file = ReadFile(has_normal=True, has_color=True, has_label=True)
  collate_batch = CollateBatch(flags.cutmix)

  dataset = Dataset(flags.location, flags.filelist, transform,
                    read_file=read_file)
  return dataset, collate_batch
