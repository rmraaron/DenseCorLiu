from scipy.io import loadmat
import numpy as np


class BFModel(object):

    def openfile(self, filename):
        bfm_data = loadmat(filename)
        shapeEV = bfm_data['shapeEV']
        shapeMU = bfm_data['shapeMU']
        shapePC = bfm_data['shapePC']
        texEV = bfm_data['texEV']
        texMU = bfm_data['texMU']
        texPC = bfm_data['texPC']
        tl = bfm_data['tl']
        return [shapeEV, shapeMU, shapePC, texEV, texMU, texPC, tl]

    def sh_tex_generate(self):
        mu = 0
        sigma = 1
        # alpha = np.full((199, 1), 0.5)
        # beta = np.full((199, 1), 0.5)
        alpha = np.random.normal(mu, sigma, (199, 1))
        beta = np.random.normal(mu, sigma, (199, 1))
        [shapeEV, shapeMU, shapePC, texEV, texMU, texPC, tl] = self.openfile(filename_read)
        diag_shapeEV = np.zeros((199, 199), dtype=float)
        np.fill_diagonal(diag_shapeEV, shapeEV)
        shape = shapeMU + shapePC @ diag_shapeEV @ alpha
        diag_texEV = np.zeros((199, 199), dtype=float)
        np.fill_diagonal(diag_texEV, texEV)
        texture = texMU + texPC @ diag_texEV @ beta
        return [shape, tl, texture]

    def write_obj(self, filename, subject_num):
        [shape, tl, texture] = self.sh_tex_generate()
        shape_list = shape.tolist()
        texture_list = texture.tolist()
        f = open(filename, 'w')
        for i in range(0, len(shape_list)-1, 3):

            # f.write("v {0} {1} {2} {3} {4} {5}\n".format(shape_list[i][0], shape_list[i+1][0], shape_list[i+2][0],
            #                                             texture_list[i][0], texture_list[i+1][0], texture_list[i+2][0]))

            f.write("v {0} {1} {2}\n".format(shape_list[i][0], shape_list[i + 1][0], shape_list[i+2][0]))

        for face in tl:
            f.write("f {0} {1} {2}\n".format(face[0], face[1], face[2]))
        f.close()

        np.save('./subject_points/bfm_{0}'.format(subject_num), shape)


if __name__ == '__main__':
    filename_read = "/home/yajie/Downloads/BaselFaceModel/PublicMM1/01_MorphableModel.mat"
    bfmModel = BFModel()
    for i in range(1500):
        bfmModel.write_obj("./subjects/bfm_{0}.obj".format(i), i)









