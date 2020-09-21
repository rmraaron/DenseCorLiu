import numpy as np
import re
import pymesh
import trimesh
from tqdm import tqdm
import os


def open_file_points(wrl_filename):
    with open(wrl_filename, 'r') as vrml:
        i = 0
        # points_list is used to save all points coordinates, each point is represented as one element
        points_list = []
        for lines in vrml:
            if i > 6 and i < 6003:
                line = lines.split('\n')[0]
                points = line.split(',')[0]
                # point_list is used to save one point, each element represent one axis (x, y, z)
                point_list = []
                point_list.append(float(points.split()[0]))
                point_list.append(float(points.split()[1]))
                point_list.append(float(points.split()[2]))
                points_list.append(point_list)
            i += 1
        points_array = np.array(points_list, dtype=np.float)
        return points_array


def open_file_mesh(wrl_filename):
    with open(wrl_filename, 'r') as vrml:
        i = 0
        triangle_list = []
        for lines in vrml:
            if i > 6005 and i < 17759:
                line = lines.split('\n')[0]
                coordinates = line.split('-1')[0]
                coor_list = []
                coor_list.append(int(coordinates.split(', ')[0]))
                coor_list.append(int(coordinates.split(', ')[1]))
                coor_list.append(int(coordinates.split(', ')[2]))
                triangle_list.append(coor_list)
            i += 1
        triangle_array = np.array(triangle_list)
        return triangle_array


def open_vertices_obj(obj_filename, n_vert=29495):
    with open(obj_filename, 'r') as obj:
        data = obj.read()
        lines = data.splitlines()
        vertices = np.zeros(dtype=np.float, shape=(n_vert, 3))
        i = 0
        for line in lines:
            if line:
                if line[0] == 'v':
                    line_f = re.split(' ', line)
                    vertices[i] = [float(line_f[1]), float(line_f[2]), float(line_f[3])]
                    i += 1
    return vertices


def open_faces_obj(obj_filename):
    with open(obj_filename, 'r') as obj:
        data = obj.read()
        lines = data.splitlines()
        faces = np.zeros(dtype=np.int, shape=(58034, 3))
        i = 0
        for line in lines:
            if line:
                if line[0] == 'f':
                    line_f = re.split(' ', line)
                    faces[i] = [int(line_f[1]), int(line_f[2]), int(line_f[3])]
                    i += 1
    return faces


def loop_subdivision(wrl_filename, faces):
    vertices = open_file_points(wrl_filename)
    mesh_inpo = pymesh.form_mesh(vertices, faces)
    mesh_interpolation = pymesh.subdivide(mesh_inpo, order=2, method='loop')
    nodes = mesh_interpolation.nodes
    faces_inpo = mesh_interpolation.faces
    return nodes, faces_inpo


def get_vertices_index(faces):
    nodes, faces_inpo = loop_subdivision('./BU3DFE/F0001_AN01WH_F3D.wrl', faces)

    vertice_simp = open_vertices_obj('./test_subdivision_simp.obj')
    vertices_indexlist = []
    i = 0
    for vertex_simp in tqdm(vertice_simp):
        vertex_diff = np.abs(nodes - vertex_simp)
        vertex_dist = np.sum(vertex_diff, axis=1)
        # min_dist = vertex_dist.min()
        # num_candidates = len(np.argwhere(vertex_dist < min_dist + 0.06))
        # if num_candidates >= 2:
        #     candidates = np.argsort(vertex_dist)[:num_candidates]
        #     candidate_diff = vertex_diff[candidates]
        #     vertex_index = candidates[np.argmin(candidate_diff[:, 2])]
        # else:
        vertex_index = np.argmin(vertex_dist)
        vertices_indexlist.append(vertex_index)
        i += 1
    np.save('./realdata_ver_index', vertices_indexlist)


def save_realpoints(wrl_filename, faces, save_file='./realpoints_bu3dfe/test'):
    nodes_array = np.zeros(shape=(29495, 3), dtype=np.float)
    nodes, faces_inpo = loop_subdivision(wrl_filename, faces)
    points_indice = np.load('realdata_ver_index.npy', allow_pickle=True)
    i = 0
    for point_index in points_indice:
        nodes_array[i] = nodes[point_index]
        i += 1
    np.save(save_file, nodes_array)


if __name__ == '__main__':
    faces = open_file_mesh('./BU3DFE/F0001_AN01WH_F3D.wrl')

    if not os.path.exists('./realpoints_bu3dfe'):
        os.mkdir('./realpoints_bu3dfe')

    for file in tqdm(os.listdir('./BU3DFE')):
        if file.endswith('.wrl'):
            file_path = os.path.join('./BU3DFE', file)
            save_path = os.path.join('./realpoints_bu3dfe', os.path.splitext(file)[0])
            save_realpoints(file_path, faces, save_path)





    # f = open('./test_subdivision.obj', 'w')
    # for node in nodes:
    #     f.write("v {0} {1} {2}\n".format(node[0], node[1], node[2]))
    # for face in faces_inpo:
    #     f.write("f {0} {1} {2}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))
    # f.close()

    # mesh_sample = trimesh.Trimesh(vertices=nodes, faces=faces_inpo)
    # sample_result = trimesh.sample.sample_surface(mesh_sample, 29495)
    # vertices_sample = sample_result[0]

    # f = open('./test.obj', 'w')
    # for vertice in vertices_sample:
    #     f.write("v {0} {1} {2}\n".format(vertice[0], vertice[1], vertice[2]))
    # f.close()

    # get_vertices_index(faces)
