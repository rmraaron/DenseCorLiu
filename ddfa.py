import numpy as np
from scipy.io import loadmat
import pickle


def load_file():
    u_exp = np.load("./3dffa_data/u_exp.npy")
    w_exp = np.load("./3dffa_data/w_exp_sim.npy")
    u_shp = np.load("./3dffa_data/u_shp.npy")
    w_shp = np.load("./3dffa_data/w_shp_sim.npy")
    triangle_data = loadmat("./3dffa_data/tri.mat")
    triangle = triangle_data["tri"].T
    return [u_exp, w_exp, u_shp, w_shp, triangle]


def shp_exp_generate():
    mu = 0
    sigma = 1
    vertices = []
    param_list = []
    alpha_exp_list = []
    for i in range(6):
        param = np.random.normal(mu, sigma, (62, ))
        param_list.append(param)
    f = open("./3dffa_data/param_whitening.pkl", 'rb')
    meta = pickle.load(f)
    f.close()
    param_mean = meta['param_mean']
    param_std = meta['param_std']
    for param in param_list:
        param_whitening = param * param_std + param_mean
        alpha_exp = param_whitening[52:].reshape(-1, 1)
        alpha_exp_list.append(alpha_exp)
    alpha_shp = param_whitening[12: 52].reshape(-1, 1)
    [u_exp, w_exp, u_shp, w_shp, triangle] = load_file()
    u = u_shp + u_exp
    for i in range(6):
        alpha_exp = alpha_exp_list[i]
        vertex = u + w_shp @ alpha_shp + w_exp @ alpha_exp
        vertices.append(vertex)
    return [vertices, triangle]


def write_obj(subject_num):
    [vertex_list, triangle] = shp_exp_generate()
    for i in range(6):
        f = open('./subjects/sub{0}_exp{1}.obj'.format(subject_num, i), 'w')
        for j in range(0, len(vertex_list[i]) - 1, 3):
            f.write("v {0} {1} {2}\n".format(vertex_list[i][j][0], vertex_list[i][j + 1][0], vertex_list[i][j + 2][0]))
        for face in triangle:
            f.write("f {0} {1} {2}\n".format(face[0], face[1], face[2]))
        f.close()
        np.save('./subject_points/sub{0}_exp{1}'.format(subject_num, i), vertex_list[i])


if __name__ == '__main__':
    for i in range(1500):
        write_obj(i)