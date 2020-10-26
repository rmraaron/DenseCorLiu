import numpy as np
import re
import tensorflow as tf
import random
from tensorflow.python.framework import ops
import sys
import os
import h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
nn_distance_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_nndistance_so.so'))


def open_face_obj(obj_file):

    with open(obj_file, 'r') as obj:
        data = obj.read()
        lines = data.splitlines()
        faces = np.zeros(dtype=np.int, shape=(58366, 3))
        i = 0
        for line in lines:
            if line:
                if line[0] == 'f':
                    line_f = re.split(' ', line)
                    faces[i] = [int(line_f[1]) - 1, int(line_f[2]) - 1, int(line_f[3]) - 1]
                    i += 1
    return faces


def open_face_file(npy_file):

    # with open(obj_file, 'r') as obj:
    #     data = obj.read()
    # lines = data.splitlines()
    # faces = np.zeros(dtype=np.int, shape=(58366, 3))
    # i = 0
    # for line in lines:
    #     if line:
    #         if line[0] == 'f':
    #             line_f = re.split(' ', line)
    #             faces[i] = [int(line_f[1]) - 1, int(line_f[2]) - 1, int(line_f[3]) - 1]
    #             i += 1

    data_npy = np.load(npy_file, allow_pickle=True)
    faces = data_npy[1]
    # faces = tf.convert_to_tensor(faces, dtype=tf.int32)
    return faces


def normals_cal(points_data, faces):
    # normals = tf.zeros(shape=(58366, 3), dtype=tf.dtypes.float32)
    ver_1 = tf.gather(points_data, faces[:, 0])
    ver_2 = tf.gather(points_data, faces[:, 1])
    ver_3 = tf.gather(points_data, faces[:, 2])
    u = ver_2 - ver_1
    v = ver_3 - ver_1
    normals = tf.linalg.cross(u, v)
    return normals


def edge_cal(points_data, faces):
    v_1 = tf.gather(points_data, faces[:, 0])
    v_2 = tf.gather(points_data, faces[:, 1])
    v_3 = tf.gather(points_data, faces[:, 2])

    def edge_length(point_a, point_b):
        return tf.norm(point_a - point_b, axis=1)

    edge_0 = edge_length(v_1, v_2)
    edge_1 = edge_length(v_2, v_3)
    edge_2 = edge_length(v_1, v_3)

    return edge_0, edge_1, edge_2


def load_npyfile(npy_file):
    points_clouds = np.load(npy_file, allow_pickle=True)
    # return point clouds instead of triangles
    return points_clouds[0]


def load_real_npyfile(npy_file):
    points_clouds = np.load(npy_file, allow_pickle=True)
    points_clouds = points_clouds - np.mean(points_clouds, axis=0)
    points_clouds_norm = points_clouds / np.max(np.linalg.norm(points_clouds, axis=1))
    return points_clouds_norm


def points_sampling(i, npy_file, faces):
    points_data = np.load(npy_file, allow_pickle=True)[0]
    for j in range(10):
        index_random = np.arange(points_data.shape[0])
        np.random.shuffle(index_random)
        face_index_random = np.zeros(index_random.size, dtype=np.int)
        face_index_random[index_random] = np.arange(0, index_random.size)
        shuffled_points_data = points_data[index_random]
        shuffled_faces = face_index_random[faces]
        np.save('./pc_sampling/sub{0}_rand{1}'.format(i, j), (shuffled_points_data, shuffled_faces))


# def shuffle_data(pc_list, label_list):
#     indices = list(zip(pc_list, label_list))
#     random.shuffle(indices)
#     pc_list, label_list = zip(*indices)
#     return pc_list, label_list


def nn_distance(shp_id, label_points):

    # def chamfer_distance(point_set_a, point_set_b, name=None):
    #     with tf.compat.v1.name_scope(name, "chamfer_distance_evaluate",
    #                                  [point_set_a, point_set_b]):
    #         point_set_a = tf.convert_to_tensor(value=point_set_a)
    #         point_set_b = tf.convert_to_tensor(value=point_set_b)
    #
    #         dimension = point_set_a.shape.as_list()[-1]
    #
    #         # Create N x M matrix where the entry i,j corresponds to ai - bj (vector of
    #         # dimension D).
    #         difference = (
    #                 tf.expand_dims(point_set_a, axis=-2) -
    #                 tf.expand_dims(point_set_b, axis=-3))
    #         # Calculate the square distances between each two points: |ai - bj|^2.
    #         square_distances = tf.einsum("...i,...i->...", difference, difference)
    #
    #         minimum_square_distance_a_to_b = tf.reduce_min(
    #             input_tensor=square_distances, axis=-1)
    #         minimum_square_distance_b_to_a = tf.reduce_min(
    #             input_tensor=square_distances, axis=-2)
    #
    #         return (
    #                 tf.reduce_mean(input_tensor=minimum_square_distance_a_to_b, axis=-1) +
    #                 tf.reduce_mean(input_tensor=minimum_square_distance_b_to_a, axis=-1))
    # loss = chamfer_distance(shp_pred, label_points)
    #
    # return loss
    '''
    dist = tf.Variable(1, shape=(N, 1))
    for i in range(N):
        points = tf.gather(shp_id, i)
        points_pre = tf.reshape(points, shape=(1, 3))
        pc_diff = label_points - points_pre
        pc_dist = tf.sqrt(tf.reduce_sum(pc_diff ** 2, axis=1))
        dist1_point = tf.reduce_min(pc_dist)
        idx1_point = tf.argmin(pc_dist)
        dist[i] = dist1_point
        print()
    '''

    shp_pred = tf.expand_dims(shp_id, 0)
    # N and M represent numbers of two point clouds respectively
    # N = shp_pred.get_shape()[1]
    # M = label_points.get_shape()[1]
    # pre_expand_tile = tf.tile(tf.expand_dims(shp_pred, 2), [1, 1, M, 1])
    # label_expand_tile = tf.tile(tf.expand_dims(label_points, 1), [1, N, 1, 1])
    # pc_diff = pre_expand_tile - label_expand_tile
    # pc_dist = tf.sqrt(tf.reduce_sum(pc_diff ** 2, axis=-1))
    # dist1 = tf.reduce_min(pc_dist, axis=2)
    # idx1 = tf.argmin(pc_dist, axis=2)
    # dist2 = tf.reduce_min(pc_dist, axis=1)
    # idx2 = tf.argmin(pc_dist, axis=1)
    # return dist1, idx1, dist2, idx2
    return nn_distance_module.nn_distance(shp_pred, label_points)

@ops.RegisterGradient('NnDistance')
def _nn_distance_grad(op,grad_dist1,grad_idx1,grad_dist2,grad_idx2):
    xyz1=op.inputs[0]
    xyz2=op.inputs[1]
    idx1=op.outputs[1]
    idx2=op.outputs[3]
    return nn_distance_module.nn_distance_grad(xyz1,xyz2,grad_dist1,idx1,grad_dist2,idx2)


def loadh5File(h5file):
    f = h5py.File(h5file)
    data = f['data'][:]
    label = f['data'][:]
    return data, label


def shuffle_data(data, label, n):
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    idx = idx[:n]
    return data[idx, ...], label[idx], idx


# points_random = np.zeros(shape=(29495, 3))
    #
    # points_data = np.load(npy_file, allow_pickle=True)[0]
    # index_list = list(range(29495))
    # for j in range(1):
    #     faces_random = []
    #     index_random = np.random.permutation(index_list)
    #     index_rand_list = index_random.tolist()
    #     for k in range(len(index_random)):
    #         points_random[k] = points_data[index_random[k]]
    #     for face in faces:
    #         face = [face[0] - 1, face[1] - 1, face[2] - 1]
    #         faces_random.append([index_rand_list.index(face[0]),
    #                              index_rand_list.index(face[1]),
    #                              index_rand_list.index(face[2])])
    #
    #     np.save('./pc_sampling/sub{0}_rand{1}'.format(i, j), (points_random, faces_random))



# def points_sampling(i, npy_file):
#     if not os.path.exists('./points_sampling'):
#         os.mkdir('./points_sampling')
#     points_data = np.load(npy_file, allow_pickle=True)[0]
#     for j in range(10):
#         points_random = np.random.permutation(points_data)
#         np.save('./points_sampling/sub{0}_rand{1}'.format(i, j), points_random)


# if __name__ == '__main__':
#     '''
#     for i in range(1500):
#         load_npyfile('./points_sampling/sub{}_rand0.npy'.format(i))
#     '''
#
#     if not os.path.exists('./pc_sampling'):
#         os.mkdir('./pc_sampling')
#
#     faces = []
#     with open('./subjects/sub0_exp0.obj', 'r') as obj:
#         lines = obj.readlines()
#     for line in lines:
#         if line:
#             if line[0] == 'f':
#                 line_f = line.split()
#                 faces.append([int(line_f[1]), int(line_f[2]), int(line_f[3])])
#     faces = np.array(faces) - 1
#
#     for i in range(1500):
#         points_sampling(i, './subject_points/sub{}_exp0.npy'.format(i), faces)

