import numpy as np
from scipy.io import loadmat
import pickle


def load_file():
    u_exp = np.load("./3ddfa_data/u_exp.npy")
    w_exp = np.load("./3ddfa_data/w_exp_sim.npy")
    u_shp = np.load("./3ddfa_data/u_shp.npy")
    w_shp = np.load("./3ddfa_data/w_shp_sim.npy")
    # triangle_data = loadmat("./3ddfa_data/template_data.mat")
    # triangle = triangle_data["mean_face"]
    triangle_data = loadmat("./3ddfa_data/tri.mat")
    triangle = triangle_data["tri"].T
    return [u_exp, w_exp, u_shp, w_shp, triangle]


def shp_exp_generate():
    # Gaussian distribution
    mu = 0
    sigma = 1
    vertices = []
    param_list = []
    # vertex_list = np.zeros(shape=(29495, 3))
    # vertex_list = np.zeros(shape=(53215, 3))
    exp_mapping_list = [np.zeros(shape=(53215, 3))]

    [u_exp, w_exp, u_shp, w_shp, triangle] = load_file()
    triangle = triangle.astype(np.int)

    # generate random 6 expressions per subject
    for i in range(6):
        param = np.random.normal(mu, sigma, (62, ))
        param_list.append(param)

    f = open("./3ddfa_data/param_whitening.pkl", 'rb')
    meta = pickle.load(f)
    f.close()


    param_mean = meta['param_mean']
    param_std = meta['param_std']
    for param in param_list:
        param_whitening = param * param_std + param_mean
        alpha_exp = param_whitening[52:].reshape(-1, 1)

        # exp_mapping to control expression variances
        exp_mapping = w_exp @ alpha_exp
        exp_mapping = exp_mapping.reshape(-1, 3)
        exp_mapping_list.append(exp_mapping)

    alpha_shp = param_whitening[12: 52].reshape(-1, 1)
    u = u_shp + u_exp
    u = u.reshape(-1, 3)
    # shp_mapping to control shape variances
    shp_mapping = w_shp @ alpha_shp
    shp_mapping = shp_mapping.reshape(-1, 3)

    for i in range(7):
        exp_mapping = exp_mapping_list[i]
        vertex = u + shp_mapping + exp_mapping

        '''
        # crop redundant vertices (from 50000+ to 29000+)
        j = 0
        for k in range(triangle.max()):
            vertex_list[j] = vertex[k]
            j += 1
        '''

        # Normalisation.
        # Transform to origin.
        vertex = vertex - np.mean(vertex, axis=0)
        # vertex_list = vertex_list - np.mean(vertex_list, axis=0)

        # Find distances to origin and do normalisation.
        # vertex_list_norm = vertex_list / np.max(np.linalg.norm(vertex_list, axis=1))
        vertex_list_norm = vertex / np.max(np.linalg.norm(vertex, axis=1))

        vertices.append(vertex_list_norm)

    return [vertices, triangle, shp_mapping, exp_mapping_list]


def write_obj(subject_num):
    [vertices, triangle, shp_mapping, exp_mapping_list] = shp_exp_generate()
    for i in range(7):
        f = open('./subjects/sub{0}_exp{1}.obj'.format(subject_num, i), 'w')
        for j in range(0, len(vertices[i])):
            f.write("v {0} {1} {2}\n".format(vertices[i][j][0], vertices[i][j][1], vertices[i][j][2]))
        for face in triangle:
            f.write("f {0} {1} {2}\n".format(face[0], face[1], face[2]))
        f.write("shp {0}\n".format(shp_mapping))
        f.write("exp {0}\n".format(exp_mapping_list[i]))
        f.close()
        np.save('./subject_points/sub{0}_exp{1}'.format(subject_num, i), [vertices[i], shp_mapping, exp_mapping_list[i]])


if __name__ == '__main__':

    for i in range(1500):
        write_obj(i)