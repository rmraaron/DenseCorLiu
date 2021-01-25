from scipy.io import loadmat
import numpy as np


mat_file = loadmat("/home/yajie/PycharmProjects/DenseCorLiu/3ddfa_data/bfm_subset(29495).mat")
tri_mat = mat_file['subset_info']
mat_faces = tri_mat["face"][0][0]
mat_shapes = tri_mat["shape"][0][0]


def compute_edges_num(faces):
    edge_set = set()
    for face in faces:
        for ids in [[0, 1], [0, 2], [1, 2]]:
            edge = face[ids]
            if edge[0] == edge[1]:
                print("edge same points", face)
                continue
            elif edge[0] > edge[1]:
                edge[1], edge[0] = edge[0], edge[1]
            edge_set.add(",".join([str(i) for i in edge]))
    edge_array = np.array([[int(j)-1 for j in i.split(",")] for i in edge_set])
    # return edge_array
    np.save("./edge_index", edge_array)


def compute_normed_shape(shapes):
    shapes = shapes - np.mean(shapes, axis=0)
    shapes_norm = shapes / np.max(np.linalg.norm(shapes, axis=1))
    return shapes_norm


if __name__ == '__main__':
    # compute_edges_num(mat_faces)
    edge_array = np.load('./edge_index.npy', allow_pickle=True)
    shapes_norm = compute_normed_shape(mat_shapes)
    template_edge_length = np.linalg.norm(shapes_norm[edge_array[:, 0]] - shapes_norm[edge_array[:, 1]], axis=1)
    np.save("./template_edge_length", template_edge_length)