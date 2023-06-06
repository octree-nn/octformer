# --------------------------------------------------------
# OctFormer: Octree-based Transformers for 3D Point Clouds
# Copyright (c) 2023 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import os
import argparse
import numpy as np
import wget
import zipfile
from tqdm import tqdm
from plyfile import PlyData, PlyElement


parser = argparse.ArgumentParser()
parser.add_argument('--path_in', type=str, default='data/scannet.ply/train')
parser.add_argument('--path_out', type=str, default='data/scannet')
parser.add_argument('--suffix_out', type=str, default='.npz')
# parser.add_argument('--align_axis', action='store_true')
parser.add_argument('--path_pred', type=str, default='logs/scannet/pred_eval')
parser.add_argument('--filelist', type=str, default='scannetv2_test_new.txt')
parser.add_argument('--scannet200', type=str, default='false')
parser.add_argument('--run', type=str, default='process_scannet',  # noqa
    help='Choose from `process_scannet`, `generate_output_seg` and `calc_iou`')
args = parser.parse_args()


suffix = '_vh_clean_2.ply'
class_labels = ('wall', 'floor', 'cabinet', 'bed', 'chair',
                'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'desk', 'curtain',
                'refrigerator', 'shower curtain', 'toilet', 'sink',
                'bathtub', 'otherfurniture')

if args.scannet200.lower() == 'false':
  class_ids = (  # !!! the first element 0 represents unlabeled points  !!!
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
else:
  class_ids = (
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22,
      23, 24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45,
      46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67,
      68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89,
      90, 93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110,
      112, 115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138,
      139, 140, 141, 145, 148, 154, 155, 156, 157, 159, 161, 163, 165, 166, 168,
      169, 170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 221, 229,
      230, 232, 233, 242, 250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325,
      331, 342, 356, 370, 392, 395, 399, 408, 417, 488, 540, 562, 570, 572, 581,
      609, 748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171,
      1172, 1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184,
      1185, 1186, 1187, 1188, 1189, 1190, 1191)

label_dict = np.zeros(1200, dtype=np.int32)
label_dict[np.array(class_ids)] = np.arange(len(class_ids))
ilabel_dict = np.array(class_ids)


def download_filelists():
  path_out = args.path_out
  os.makedirs(path_out, exist_ok=True)

  # download via wget
  zip_file = os.path.join(path_out, 'filelist.zip')
  if not os.path.exists(zip_file):
    url = 'https://www.dropbox.com/s/3b068cw4uofaxtx/scannet_filelist.zip?dl=1'
    print('-> Download the filelist from dropbox: %s' % url)
    wget.download(url, zip_file)

  # unzip
  with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    print('-> Unzip the filelist to %s' % path_out)
    zip_ref.extractall(path_out)


def read_ply(filename):
  plydata = PlyData.read(filename)
  vertex, face = plydata['vertex'].data, plydata['face'].data
  props = [vertex[name].astype(np.float32) for name in vertex.dtype.names]
  vertex = np.stack(props[:3], axis=1)
  props = np.stack(props[3:], axis=1)
  face = np.stack(face['vertex_indices'], axis=0)
  return vertex, face, props


def face_normal(vertex, face):
  v01 = vertex[face[:, 1]] - vertex[face[:, 0]]
  v02 = vertex[face[:, 2]] - vertex[face[:, 0]]
  vec = np.cross(v01, v02)
  length = np.sqrt(np.sum(vec**2, axis=1, keepdims=True)) + 1.0e-8
  nf = vec / length
  area = length * 0.5
  return nf, area


def vertex_normal(vertex, face):
  nf, area = face_normal(vertex, face)
  nf = nf * area

  nv = np.zeros_like(vertex)
  for i in range(face.shape[0]):
    nv[face[i]] += nf[i]

  length = np.sqrt(np.sum(nv**2, axis=1, keepdims=True)) + 1.0e-8
  nv = nv / length
  return nv


def save_ply(point_cloud, filename):
  ncols = point_cloud.shape[1]
  py_types = (float, float, float, float, float, float,
              int, int, int, int)[:ncols]
  npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
               ('label', 'u1')][:ncols]

  # format into NumPy structured array
  vertices = []
  for row_idx in range(point_cloud.shape[0]):
    point = point_cloud[row_idx]
    vertices.append(tuple(dtype(val) for dtype, val in zip(py_types, point)))
  structured_array = np.array(vertices, dtype=npy_types)
  el = PlyElement.describe(structured_array, 'vertex')

  # write ply
  PlyData([el]).write(filename)
  print('Save:', filename)


def save_npz(point_cloud, filename):
  output = {'points': point_cloud[:, :3].astype(np.float32),
            'normals': point_cloud[:, 3:6].astype(np.float32),
            'colors': point_cloud[:, 6:9].astype(np.float32), }
  if point_cloud.shape[1] == 10:
    output['labels'] = point_cloud[:, 9].astype(np.int32)
  np.savez(filename, **output)
  print('Save:', filename)


def align_to_axis(filename, vertex):
  with open(filename) as fid:
    for line in fid:
      (key, val) = line.split(" = ")
      if key == 'axisAlignment':
        rot = np.fromstring(val, sep=' ').reshape(4, 4)
        vertex = np.matmul(vertex, rot[:3, :3].T) + rot[:3, 3]
        break
  return vertex


def process_scannet():
  download_filelists()

  subsets = {'train': 'scans', 'test': 'scans_test'}
  for path_out, path_in in subsets.items():
    curr_path_out = os.path.join(args.path_out, path_out)
    curr_path_in = os.path.join(args.path_in, path_in)
    os.makedirs(curr_path_out, exist_ok=True)

    scene_ids = os.listdir(curr_path_in)
    for scene_id in scene_ids:
      filename_ply = os.path.join(curr_path_in, scene_id, scene_id + suffix)
      filename_out = os.path.join(curr_path_out, scene_id + args.suffix_out)
      if os.path.exists(filename_out):
        print('Skip:', filename_out)
        continue

      # Load pointcloud file
      vertex, face, props = read_ply(filename_ply)
      assert np.unique(props[:, -1]).size == 1
      # info_file = filename_ply.replace(suffix, '.txt')
      # vertex = align_to_axis(info_file, vertex)
      nv = vertex_normal(vertex, face)
      pointcloud = np.concatenate([vertex, nv, props], axis=1)

      # Load label file
      label = np.zeros((pointcloud.shape[0], 1))
      filename_label = filename_ply[:-4] + '.labels.ply'
      if os.path.exists(filename_label):
        _, _, label = read_ply(filename_label)
        # check that the pointcloud and its label has same vertices.
        assert pointcloud.shape[0] == label.shape[0]

        # save the original label
        label = label[:, -1:].astype(np.int32)
        filename_txt = filename_out[:-4] + '.txt'
        np.savetxt(filename_txt, label, fmt='%d')

        # map the original label to the new label
        label_new = label_dict[label]
        pointcloud[:, -1:] = label_new

      # save the file
      save_file = save_ply if args.suffix_out == '.ply' else save_npz
      save_file(pointcloud, filename_out)


def generate_output_seg():
  ''' Converts the predicted probabilities to segmentation labels: merge the
  predictions for each chunk; map the predicted labels to the original labels.
  '''

  # load filelist
  filename_scans = []
  with open(args.filelist, 'r') as fid:
    for line in fid:
      filename_scans.append(line.split()[0][:-4])

  # process
  probs = {}
  for filename_scan in tqdm(filename_scans, ncols=80):
    filename_pred = os.path.join(args.path_pred, filename_scan + '.eval.npz')
    pred = np.load(filename_pred)
    prob0 = pred['prob']

    # merge `chunk_x`
    if 'chunk' in filename_scan:
      filename_mask = filename_scan + '.mask.npy'
      mask = np.load(os.path.join(args.path_in, filename_mask))
      prob1 = np.zeros([mask.shape[0], prob0.shape[1]])
      prob1[mask] = prob0
      prob0 = prob1                       # update prob0
      filename_scan = filename_scan[:-8]  # remove '.chunk_x'

    probs[filename_scan] = probs.get(filename_scan, 0) + prob0

  # output
  os.makedirs(args.path_out, exist_ok=True)
  for filename, prob in tqdm(probs.items(), ncols=80):
    filename_label = filename + '.txt'
    filename_npy = filename + '.npy'
    label = np.argmax(prob, axis=1)
    label = ilabel_dict[label]
    np.savetxt(os.path.join(args.path_out, filename_label), label, fmt='%d')
    # np.save(os.path.join(args.path_out, filename_npy), prob)


def calc_iou():
  # init
  intsc, union, accu = {}, {}, 0
  for k in class_ids[1:]:
    intsc[k] = 0
    union[k] = 0

  # load files
  pred_files = sorted(os.listdir(args.path_pred))
  pred_files = [f for f in pred_files if f.endswith('.txt')]
  for filename in tqdm(pred_files, ncols=80):
    label_pred = np.loadtxt(os.path.join(args.path_pred, filename))
    label_gt = np.loadtxt(os.path.join(args.path_in, filename))

    # omit labels out of class_ids[1:]
    mask = np.zeros_like(label_gt).astype(bool)
    for i in range(label_gt.shape[0]):
      mask[i] = label_gt[i] in class_ids[1:]
    label_pred = label_pred[mask]
    label_gt = label_gt[mask]

    ac = (label_gt == label_pred).mean()
    tqdm.write("Accu: %s, %.4f" % (filename, ac))
    accu += ac

    for k in class_ids[1:]:
      pk, lk = label_pred == k, label_gt == k
      intsc[k] += np.sum(np.logical_and(pk, lk).astype(np.float32))
      union[k] += np.sum(np.logical_or(pk, lk).astype(np.float32))

  # iou
  iou_part = 0
  for k in class_ids[1:]:
    iou_part += intsc[k] / (union[k] + 1.0e-10)
  iou = iou_part / len(class_ids[1:])
  print('Accu: %.6f' % (accu / len(pred_files)))
  print('IoU: %.6f' % iou)


if __name__ == '__main__':
  eval('%s()' % args.run)
