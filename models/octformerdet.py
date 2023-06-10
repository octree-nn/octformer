# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# ------------------------------------------------------

import torch
import ocnn
import MinkowskiEngine as ME
from ocnn.octree import Octree, Points
from mmdet3d.core import bbox3d2result
from mmdet3d.models import DETECTORS, build_backbone, build_head
from mmdet3d.models.detectors import Base3DDetector
from mmdet3d.models.builder import BACKBONES

from .octformer import OctFormer as _OctFormer


@BACKBONES.register_module()
class OctFormer(_OctFormer):
  pass


@DETECTORS.register_module()
class OctFormerSingleStage3DDetector(Base3DDetector):
  def __init__(
          self, backbone, head, voxel_size, octree_depth, octree_feature='F',
          train_cfg=None, test_cfg=None, init_cfg=None, pretrained=None,):
    super().__init__(init_cfg)
    self.backbone = build_backbone(backbone)
    head.update(train_cfg=train_cfg)
    head.update(test_cfg=test_cfg)
    self.head = build_head(head)
    self.voxel_size = voxel_size
    self.octree_depth = octree_depth
    self.scale_factor = 2 / (2**octree_depth * voxel_size)
    self.octree_feature = octree_feature
    self.init_weights()

  def build_octree(self, raw_points):
    octrees = []
    batch_size = len(raw_points)
    for batch_idx in range(batch_size):
      xyz = raw_points[batch_idx][:, :3]
      color = raw_points[batch_idx][:, 3:]
      xyz = xyz * self.scale_factor  # normalize points to [-1, 1]
      point_cloud = Points(xyz, features=color)
      point_cloud.clip(min=-1.0, max=1.0)

      octree = Octree(self.octree_depth, device=xyz.device)
      octree.build_octree(point_cloud)
      octrees.append(octree)

    octree = ocnn.octree.merge_octrees(octrees)
    octree.construct_all_neigh()
    x = ocnn.modules.InputFeature(self.octree_feature, nempty=True)(octree)
    return x, octree

  def extract_feat(self, points):
    xs, octree = self.build_octree(points)
    xs = self.backbone(xs, octree, self.octree_depth)
    features, depths = list(xs.values()), list(xs.keys())

    outs = []
    coordinate_manager = None
    scale_factor = self.scale_factor * self.voxel_size
    for i in range(len(features)):
      (x, y, z, b) = octree.xyzb(depths[i], nempty=True)
      xyz = torch.stack([x, y, z], dim=-1)
      xyz = (xyz / (2 ** (depths[i] - 1)) - 1) / scale_factor
      bxyz = torch.cat([b.unsqueeze(1), xyz], dim=1).int()
      outs.append(ME.SparseTensor(
          coordinates=bxyz, features=features[i], tensor_stride=2**(i+3),
          coordinate_manager=coordinate_manager))
      if coordinate_manager is None:
        coordinate_manager = outs[0].coordinate_manager
    return outs

  def forward_train(self, points, gt_bboxes_3d, gt_labels_3d, img_metas):
    r'''Forward of training.
    '''
    xs = self.extract_feat(points)
    losses = self.head.forward_train(xs, gt_bboxes_3d, gt_labels_3d, img_metas)
    return losses

  def simple_test(self, points, img_metas, *args, **kwargs):
    r'''Test without augmentations.
    '''
    xs = self.extract_feat(points)
    bbox_list = self.head.forward_test(xs, img_metas)
    bbox_results = [
        bbox3d2result(bboxes, scores, labels)
        for bboxes, scores, labels in bbox_list
    ]
    return bbox_results

  def aug_test(self, points, img_metas, **kwargs):
    r'''Test with augmentations.
    '''
    raise NotImplementedError
