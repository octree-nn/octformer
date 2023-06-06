# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import torch.nn.functional as F
import ocnn

from thsolver import Solver
from datasets import get_modelnet40_dataset
from builder import get_classification_model


class ClsSolver(Solver):

  def get_model(self, flags):
    return get_classification_model(flags)

  def get_dataset(self, flags):
    return get_modelnet40_dataset(flags)

  def get_input_feature(self, octree):
    flags = self.FLAGS.MODEL
    octree_feature = ocnn.modules.InputFeature(flags.feature, flags.nempty)
    data = octree_feature(octree)
    return data

  def forward(self, batch):
    octree, label = batch['octree'].cuda(), batch['label'].cuda()
    data = self.get_input_feature(octree)
    logits = self.model(data, octree, octree.depth)
    log_softmax = F.log_softmax(logits, dim=1)
    loss = F.nll_loss(log_softmax, label)
    pred = torch.argmax(logits, dim=1)
    accu = pred.eq(label).float().mean()
    return loss, accu

  def train_step(self, batch):
    loss, accu = self.forward(batch)
    return {'train/loss': loss, 'train/accu': accu}

  def test_step(self, batch):
    with torch.no_grad():
      loss, accu = self.forward(batch)
    return {'test/loss': loss, 'test/accu': accu}


if __name__ == "__main__":
  ClsSolver.main()
