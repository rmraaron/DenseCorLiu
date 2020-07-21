import numpy as np
import re
import tensorflow as tf


def open_face_file(obj_file):
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
    faces = tf.convert_to_tensor(faces, dtype=tf.int32)
    return faces


def normals_cal(points_data, faces):
    normals = tf.zeros(shape=(58366, 3), dtype=tf.dtypes.float32)
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
    return points_clouds


'''
def points_sampling(i, npy_file):
    points_data = np.load(npy_file, allow_pickle=True)[0]
    for j in range(10):
        points_random = np.random.permutation(points_data)
        np.save('./points_sampling/sub{0}_rand{1}'.format(i, j), points_random)


if __name__ == '__main__':

    for i in range(1500):
        load_npyfile('./points_sampling/sub{}_rand0.npy'.format(i))

'''