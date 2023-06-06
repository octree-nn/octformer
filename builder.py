# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import ocnn
import torch
import datasets
import models


def octsegformer_large(in_channels, out_channels, **kwargs):
  return models.OctFormerSeg(
      in_channels, out_channels,
      channels=[192, 384, 768, 768],
      num_blocks=[2, 2, 18, 2],
      num_heads=[12, 24, 48, 48],
      patch_size=32, dilation=4,
      drop_path=0.5, nempty=True,
      stem_down=2, head_up=2,
      fpn_channel=168,
      head_drop=[0.5, 0.5])


def octsegformer(in_channels, out_channels, **kwargs):
  return models.OctFormerSeg(
      in_channels, out_channels,
      channels=[96, 192, 384, 384],
      num_blocks=[2, 2, 18, 2],
      num_heads=[6, 12, 24, 24],
      patch_size=32, dilation=4,
      drop_path=0.5, nempty=True,
      stem_down=2, head_up=2,
      fpn_channel=168,
      head_drop=[0.5, 0.5])


def octsegformer_small(in_channels, out_channels, **kwargs):
  return models.OctFormerSeg(
      in_channels, out_channels,
      channels=[96, 192, 384, 384],
      num_blocks=[2, 2, 6, 2],
      num_heads=[6, 12, 24, 24],
      patch_size=32, dilation=4,
      drop_path=0.5, nempty=True,
      stem_down=2, head_up=2,
      fpn_channel=168,
      head_drop=[0.5, 0.5])


def octsegformer_cls(in_channels, out_channels, nemtpy, **kwargs):
  return models.OctFormerCls(
      in_channels, out_channels,
      channels=[96, 192],
      num_blocks=[6, 6],
      num_heads=[6, 12],
      patch_size=32, dilation=2,
      drop_path=0.5, nempty=nemtpy,
      stem_down=2, head_drop=0.5)


def get_segmentation_model(flags):
  params = {
      'in_channels': flags.channel, 'out_channels': flags.nout,
      'interp': flags.interp, 'nempty': flags.nempty,
  }
  networks = {
      'octsegformer': octsegformer,
      'octsegformer_large': octsegformer_large,
      'octsegformer_small': octsegformer_small,
  }

  return networks[flags.name.lower()](**params)


def get_classification_model(flags):
  if flags.name.lower() == 'lenet':
    model = ocnn.models.LeNet(
        flags.channel, flags.nout, flags.stages, flags.nempty)
  elif flags.name.lower() == 'hrnet':
    model = ocnn.models.HRNet(
        flags.channel, flags.nout, flags.stages, nempty=flags.nempty)
  elif flags.name.lower() == 'octformercls':
    model = octsegformer_cls(flags.channel, flags.nout, flags.nempty)
  else:
    raise ValueError
  return model


def get_segmentation_dataset(flags):
  if flags.name.lower() == 'shapenet':
    return datasets.get_shapenet_seg_dataset(flags)
  elif flags.name.lower() == 'scannet':
    return datasets.get_scannet_dataset(flags)
  elif flags.name.lower() == 'kitti':
    return datasets.get_kitti_dataset(flags)
  else:
    raise ValueError
