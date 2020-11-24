import torch
import model_torch
from tqdm import tqdm
import numpy as np
import data_preprosessing
from chamfer_distance import ChamferDistance

device = torch.device('cuda')

BATCH_SIZE = 1
NUM_POINT = 29495

LAMBDA1 = 1.6e-4
LAMBDA2 = 1.6e-4

BASE_LEARNING_RATE = 1e-4


def eval_id():
    logfile_eval = open('./log_torch/log_eval_id.txt', 'w')
    densecor = model_torch.DenseCor().to(device)
    densecor.load_state_dict(torch.load('./log_torch/fixed/model.pth'))

    faces_triangle = data_preprosessing.open_face_obj('./subjects/sub0_exp0.obj')

    pc_data = data_preprosessing.loadh5File_single('./dataset/subject_points.h5')

    file_size = pc_data.shape[0]
    num_batches = file_size

    for batch_idx in tqdm(range(3)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        point_clouds = pc_data[start_idx:end_idx, :, :]
        point_clouds = torch.from_numpy(point_clouds).to(dtype=torch.float32, device=device)
        # densecor = densecor.eval()
        s_id, s_exp, s_pred = densecor(point_clouds)
        loss = model_torch.get_loss(s_pred, faces_triangle, point_clouds.detach(), LAMBDA1, LAMBDA2)
        model_torch.log_writing(logfile_eval, 'loss_test: %f' % loss)
        print('loss_test: %f' % loss)

        point_clouds_np = point_clouds.to(device='cpu').detach().numpy()
        s_pred_np = s_pred.to(device='cpu').detach().numpy()
        np.save('./sub{}_origin'.format(batch_idx), point_clouds_np.reshape(29495, 3))
        np.save('./sub{}_pred'.format(batch_idx), s_pred_np.reshape(29495, 3))


def eval_exp():
    logfile_eval = open('./log_torch/log_eval_exp.txt', 'w')
    densecor = model_torch.DenseCor().to(device)
    densecor.load_state_dict(torch.load('./log_torch/exp/model.pth'))

    faces_triangle = data_preprosessing.open_face_obj('./subjects/sub0_exp0.obj')

    pc_data = data_preprosessing.loadh5File_single('./dataset/expression_points.h5')

    file_size = pc_data.shape[0]
    num_batches = file_size

    for batch_idx in tqdm(range(6, 9)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        point_clouds = pc_data[start_idx:end_idx, :, :]
        point_clouds = torch.from_numpy(point_clouds).to(dtype=torch.float32, device=device)
        # densecor = densecor.eval()
        with torch.no_grad():
            s_id, s_exp, s_pred = densecor(point_clouds)
            loss = model_torch.get_loss(s_pred, faces_triangle, point_clouds.detach(), LAMBDA1, LAMBDA2)
        model_torch.log_writing(logfile_eval, 'loss_test: %f' % loss)
        print('loss_test: %f' % loss)

        point_clouds_np = point_clouds.to(device='cpu').detach().numpy()
        s_pred_np = s_pred.to(device='cpu').detach().numpy()
        # np.save('./sub0_exp{}_origin'.format(batch_idx), point_clouds_np.reshape(29495, 3))
        np.save('./sub1_exp{}_pred'.format(batch_idx-6), s_pred_np.reshape(29495, 3))


def eval_endtoend():
    logfile_eval = open('./log_torch/log_eval_end.txt', 'w')
    densecor = model_torch.DenseCor().to(device)
    densecor.load_state_dict(torch.load('./log_torch/end_to_end/epoch20_model.pth'))

    faces_triangle_real = data_preprosessing.open_face_obj('./test_subdivision_simp.obj', 58034)
    faces_triangle_syn = data_preprosessing.open_face_obj('./subjects/sub0_exp0.obj')

    pc_data_real = data_preprosessing.loadh5File_single('./dataset/all_points.h5')[:2498, ...]
    pc_data_syn = data_preprosessing.loadh5File_single('./dataset/all_points.h5')[2498:, ...]

    chamfer_dist = ChamferDistance()

    file_size = pc_data_real.shape[0]
    num_batches_real = file_size


    for batch_idx in tqdm(range(3)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        point_clouds = pc_data_real[start_idx:end_idx, :, :]
        point_clouds = torch.from_numpy(point_clouds).to(dtype=torch.float32, device=device)
        # densecor = densecor.eval()
        with torch.no_grad():
            s_id, s_exp, s_pred = densecor(point_clouds)
            loss = model_torch.get_loss_real(s_pred, faces_triangle_real, point_clouds.detach(), chamfer_dist, LAMBDA1, LAMBDA2)
        model_torch.log_writing(logfile_eval, 'loss_test: %f' % loss)
        print('loss_test: %f' % loss)

        point_clouds_np = point_clouds.to(device='cpu').detach().numpy()
        s_pred_np = s_pred.to(device='cpu').detach().numpy()
        np.save('./sub{}_real_origin'.format(batch_idx), point_clouds_np.reshape(29495, 3))
        np.save('./sub{}_real_pred'.format(batch_idx), s_pred_np.reshape(29495, 3))


if __name__ == '__main__':
    # eval_id()
    # eval_exp()
    eval_endtoend()