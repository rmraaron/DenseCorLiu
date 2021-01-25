from data_preprosessing import open_vertices_obj
from scipy.io import loadmat
from scipy import spatial
import numpy as np

mouth_vertices = open_vertices_obj('./MouthRegion.obj', n=2067)

mat_file = loadmat("./3ddfa_data/bfm_subset(29495).mat")
tri_mat = mat_file['subset_info']
faces_tri = tri_mat["face"][0][0]
faces_vertices = tri_mat["shape"][0][0]

head_tree = spatial.KDTree(faces_vertices)

face_corres = []
face_distance = []

for i in mouth_vertices:
    distance, index = head_tree.query(i)
    face_corres.append(index)
    face_distance.append(distance)


face_corres = np.array(face_corres) + 1
corres_tri = set()
for index in face_corres:
    found_tri = faces_tri[np.any(index == faces_tri, axis=1)]
    [corres_tri.add(tuple(i)) for i in found_tri]
corres_tri = np.array([list(i) for i in corres_tri])

np.save("mouth_triangles", corres_tri)
