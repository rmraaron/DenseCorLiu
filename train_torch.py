import torch
import model_torch
from tqdm import tqdm
import os
import sys
import numpy as np
import torch.optim as optim
import data_preprosessing
from chamfer_distance import ChamferDistance

device = torch.device('cuda')

BATCH_SIZE = 1
NUM_POINT = 29495
MAX_EPOCH_ID = 20
MAX_EPOCH_EXP = 20
MAX_EPOCH_END = 20

LAMBDA1 = 1.6e-4
LAMBDA2 = 1.6e-4

BASE_LEARNING_RATE = 1e-4

if not os.path.exists('./log_torch'):
    os.mkdir('./log_torch')


def train_id():
    if not os.path.exists('./log_torch/fixed'):
        os.mkdir('./log_torch/fixed')
    logfile_train = open('./log_torch/log_train.txt', 'w')
    densecor = model_torch.DenseCor().to(device)

    optimizer = optim.Adam(densecor.parameters(), lr=BASE_LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    faces_triangle = data_preprosessing.open_face_obj('./subjects/sub0_exp0.obj')
    pc_data = data_preprosessing.loadh5File_single('./dataset/subject_points.h5')

    for epoch in range(1, MAX_EPOCH_ID + 1):
        model_torch.log_writing(logfile_train, '************************* EPOCH %d *************************' % epoch)
        model_torch.log_writing(logfile_train,
                    '***************** LEARNING RATE: %f *****************' % optimizer.param_groups[0]['lr'])
        sys.stdout.flush()
        print('************************* EPOCH %d *************************' % epoch)

        file_size = pc_data.shape[0]
        num_batches = file_size

        np.random.shuffle(pc_data)
        epoch_loss = 0

        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            point_clouds = pc_data[start_idx:end_idx, :, :]
            point_clouds = torch.from_numpy(point_clouds).to(dtype=torch.float32, device=device)
            optimizer.zero_grad()
            densecor = densecor.train()
            s_id, s_exp, s_pred = densecor(point_clouds)
            loss = model_torch.get_loss(s_pred, faces_triangle, point_clouds.detach(), LAMBDA1, LAMBDA2)
            loss.backward()
            optimizer.step()
            model_torch.log_writing(logfile_train, 'loss_train: %f' % loss)
            epoch_loss += loss

            if epoch == MAX_EPOCH_ID and batch_idx == num_batches - 1:
                point_clouds_np = point_clouds.to(device='cpu').detach().numpy()
                s_pred_np = s_pred.to(device='cpu').detach().numpy()
                np.save('./sub{}_origin'.format(batch_idx), point_clouds_np.reshape(29495, 3))
                np.save('./sub{}_pred'.format(batch_idx), s_pred_np.reshape(29495, 3))

        scheduler.step()
        epoch_loss_ave = epoch_loss / float(num_batches)
        model_torch.log_writing(logfile_train, 'mean_loss: %f' % epoch_loss_ave)
        print('epoch mean loss: %f' % epoch_loss_ave)

    torch.save(densecor.state_dict(), './log_torch/fixed/model.pth')
    model_torch.log_writing(logfile_train, 'model saved in file: %s' % './log_torch/fixed/model.pth')
    print('model saved in file: %s' % './log_torch/fixed/model.pth')


def train_exp():
    if not os.path.exists('./log_torch/exp'):
        os.mkdir('./log_torch/exp')
    logfile_train = open('./log_torch/log_train_exp.txt', 'w')
    densecor = model_torch.DenseCor().to(device)
    densecor.load_state_dict(torch.load('./log_torch/fixed/model.pth'))

    params = list(densecor.fc2_exp.parameters()) + list(densecor.fc3_exp.parameters())
    optimizer = optim.Adam(params, lr=BASE_LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    faces_triangle = data_preprosessing.open_face_obj('./subjects/sub0_exp0.obj')

    pc_data_train = data_preprosessing.loadh5File_single('./dataset/expression_points.h5')
    pc_data_eval = data_preprosessing.loadh5File_single('./dataset/expression_points.h5')[7500:, ...]

    for epoch in range(1, MAX_EPOCH_EXP + 1):
        model_torch.log_writing(logfile_train, '************************* EPOCH %d *************************' % epoch)
        model_torch.log_writing(logfile_train,
                    '***************** LEARNING RATE: %f *****************' % optimizer.param_groups[0]['lr'])
        sys.stdout.flush()
        print('************************* EPOCH %d *************************' % epoch)

        file_size = pc_data_train.shape[0]
        num_batches_train = file_size
        epoch_loss_train = 0
        epoch_loss_eval = 0

        np.random.shuffle(pc_data_train)
        for batch_idx in tqdm(range(num_batches_train)):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            point_clouds = pc_data_train[start_idx:end_idx, :, :]
            point_clouds = torch.from_numpy(point_clouds).to(dtype=torch.float32, device=device)
            optimizer.zero_grad()
            densecor = densecor.train()
            s_id, s_exp, s_pred = densecor(point_clouds)
            loss = model_torch.get_loss(s_pred, faces_triangle, point_clouds.detach(), LAMBDA1, LAMBDA2)
            loss.backward()
            optimizer.step()
            model_torch.log_writing(logfile_train, 'loss_train: %f' % float(loss))
            epoch_loss_train += float(loss)

            if epoch == MAX_EPOCH_EXP and batch_idx == num_batches_train - 1:
                point_clouds_np = point_clouds.to(device='cpu').detach().numpy()
                s_pred_np = s_pred.to(device='cpu').detach().numpy()
                np.save('./sub_exp{}_origin'.format(batch_idx), point_clouds_np.reshape(29495, 3))
                np.save('./sub_exp{}_pred'.format(batch_idx), s_pred_np.reshape(29495, 3))

        scheduler.step()
        epoch_loss_ave = epoch_loss_train / float(num_batches_train)
        model_torch.log_writing(logfile_train, 'train_mean_loss: %f' % epoch_loss_ave)
        print('epoch train mean loss: %f' % epoch_loss_ave)

        file_size = pc_data_eval.shape[0]
        num_batches_eval = file_size

        for batch_idx in tqdm(range(num_batches_eval)):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            point_clouds = pc_data_eval[start_idx:end_idx, :, :]
            point_clouds = torch.from_numpy(point_clouds).to(dtype=torch.float32, device=device)
            densecor = densecor.eval()
            with torch.no_grad():
                s_id, s_exp, s_pred = densecor(point_clouds)
                loss = model_torch.get_loss(s_pred, faces_triangle, point_clouds.detach(), LAMBDA1, LAMBDA2)
            epoch_loss_eval += float(loss)
        epoch_loss_mean = epoch_loss_eval / float(num_batches_eval)
        model_torch.log_writing(logfile_train, 'eval_mean_loss: %f' % epoch_loss_mean)
        print('epoch eval mean loss: %f' % epoch_loss_mean)

    torch.save(densecor.state_dict(), './log_torch/exp/model.pth')
    model_torch.log_writing(logfile_train, 'model saved in file: %s' % './log_torch/exp/model.pth')
    print('model saved in file: %s' % './log_torch/exp/model.pth')


def end_to_end_train():
    if not os.path.exists('./log_torch/end_to_end'):
        os.mkdir('./log_torch/end_to_end')
    logfile_train = open('./log_torch/log_train_endtoend.txt', 'w')
    densecor = model_torch.DenseCor().to(device)
    densecor.load_state_dict(torch.load('./log_torch/exp/model.pth'))
    optimizer = optim.Adam(densecor.parameters(), lr=BASE_LEARNING_RATE * 0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    faces_triangle_real = data_preprosessing.open_face_obj('./test_subdivision_simp.obj', 58034)
    faces_triangle_syn = data_preprosessing.open_face_obj('./subjects/sub0_exp0.obj')

    pc_data = data_preprosessing.loadh5File_single('./dataset/all_points.h5')

    chamfer_dist = ChamferDistance()

    file_size = pc_data.shape[0]
    num_batches = file_size
    idx = np.arange(num_batches)
    np.random.shuffle(idx)
    idx_train = idx[:10000]
    idx_eval = idx[10000:]

    for epoch in range(1, MAX_EPOCH_END + 1):
        model_torch.log_writing(logfile_train, '************************* EPOCH %d *************************' % epoch)
        model_torch.log_writing(logfile_train,
                    '***************** LEARNING RATE: %f *****************' % optimizer.param_groups[0]['lr'])
        sys.stdout.flush()
        print('************************* EPOCH %d *************************' % epoch)

        epoch_loss_train = 0
        epoch_loss_eval = 0

        np.random.shuffle(idx_train)
        for batch_idx in tqdm(idx_train):  # Assume batch size == 1.
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            point_clouds = pc_data[start_idx:end_idx, :, :]
            point_clouds = torch.from_numpy(point_clouds).to(dtype=torch.float32, device=device)
            optimizer.zero_grad()
            densecor = densecor.train()
            s_id, s_exp, s_pred = densecor(point_clouds)
            if batch_idx < 2498:
                loss = model_torch.get_loss_real(
                    s_pred, faces_triangle_real, point_clouds.detach(), chamfer_dist, LAMBDA1, LAMBDA2)
            else:
                loss = model_torch.get_loss_real(
                    s_pred, faces_triangle_syn, point_clouds.detach(), chamfer_dist, LAMBDA1, LAMBDA2)
            loss.backward()
            optimizer.step()
            model_torch.log_writing(logfile_train, 'loss_train: %f' % float(loss))
            epoch_loss_train += float(loss)

            if epoch % 5 == 0 and batch_idx == idx_train[-1]:
                point_clouds_np = point_clouds.to(device='cpu').detach().numpy()
                s_pred_np = s_pred.to(device='cpu').detach().numpy()
                np.save('./sub_endtoend{0}_epoch{1}_origin'.format(batch_idx, epoch), point_clouds_np.reshape(29495, 3))
                np.save('./sub_endtoend{0}_epoch{1}_pred'.format(batch_idx, epoch), s_pred_np.reshape(29495, 3))

        # file_size = pc_data_syn.shape[0]
        # num_batches_syn = file_size
        #
        # np.random.shuffle(pc_data_syn)
        # for batch_idx in tqdm(range(num_batches_syn)):
        #     start_idx = batch_idx * BATCH_SIZE
        #     end_idx = (batch_idx + 1) * BATCH_SIZE
        #     point_clouds = pc_data_syn[start_idx:end_idx, :, :]
        #     point_clouds = torch.from_numpy(point_clouds).to(dtype=torch.float32, device=device)
        #     optimizer.zero_grad()
        #     densecor = densecor.train()
        #     s_id, s_exp, s_pred = densecor(point_clouds)
        #     loss = model_torch.get_loss_real(
        #         s_pred, faces_triangle_syn, point_clouds.detach(), chamfer_dist, LAMBDA1, LAMBDA2)
        #     loss.backward()
        #     optimizer.step()
        #     model_torch.log_writing(logfile_train, 'loss_train: %f' % float(loss))
        #     epoch_loss += float(loss)
        #
        #     if epoch == MAX_EPOCH_END and batch_idx == num_batches_syn - 1:
        #         point_clouds_np = point_clouds.to(device='cpu').detach().numpy()
        #         s_pred_np = s_pred.to(device='cpu').detach().numpy()
        #         np.save('./sub_syn{}_origin'.format(batch_idx), point_clouds_np.reshape(29495, 3))
        #         np.save('./sub_syn{}_pred'.format(batch_idx), s_pred_np.reshape(29495, 3))

        scheduler.step()
        epoch_loss_ave = epoch_loss_train / float(len(idx_train))
        model_torch.log_writing(logfile_train, 'train_mean_loss: %f' % epoch_loss_ave)
        print('epoch train mean loss: %f' % epoch_loss_ave)

        torch.save(densecor.state_dict(), './log_torch/end_to_end/epoch{}_model.pth'.format(epoch))
        model_torch.log_writing(logfile_train, 'model saved in file: %s' % './log_torch/end_to_end/epoch{}_model.pth'.format(epoch))
        print('model saved in file: %s' % './log_torch/end_to_end/epoch{}_model.pth'.format(epoch))

        for batch_idx in tqdm(idx_eval):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            point_clouds = pc_data[start_idx:end_idx, :, :]
            point_clouds = torch.from_numpy(point_clouds).to(dtype=torch.float32, device=device)
            # densecor = densecor.eval()
            with torch.no_grad():
                s_id, s_exp, s_pred = densecor(point_clouds)
                if batch_idx < 2498:
                    loss = model_torch.get_loss_real(
                        s_pred, faces_triangle_real, point_clouds.detach(), chamfer_dist, LAMBDA1, LAMBDA2)
                else:
                    loss = model_torch.get_loss_real(
                        s_pred, faces_triangle_syn, point_clouds.detach(), chamfer_dist, LAMBDA1, LAMBDA2)
            epoch_loss_eval += float(loss)
        epoch_loss_mean = epoch_loss_eval / float(len(idx_eval))
        model_torch.log_writing(logfile_train, 'eval_mean_loss: %f' % epoch_loss_mean)
        print('epoch eval mean loss: %f' % epoch_loss_mean)



if __name__ == '__main__':
    # train_id()
    # train_exp()
    end_to_end_train()


