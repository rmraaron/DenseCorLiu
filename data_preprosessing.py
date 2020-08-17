import numpy as np
import re
import tensorflow as tf
import os


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

