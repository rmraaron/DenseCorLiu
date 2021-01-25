import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss.chamfer import _handle_pointcloud_input
from pytorch3d.structures import Meshes


class DenseCor(nn.Module):
    def __init__(self, num_points=29495):
        super(DenseCor, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)
        self.maxp = nn.MaxPool1d(num_points)

        self.fc1_id = nn.Linear(1024, 512)
        self.fc2_id = nn.Linear(512, 1024)
        self.fc3_id = nn.Linear(1024, num_points*3)
        self.fc1_exp = nn.Linear(1024, 512)
        self.fc2_exp = nn.Linear(512, 1024)
        self.fc3_exp = nn.Linear(1024, num_points * 3)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        batchsize = x.size()[0]
        x = x.transpose(2, 1).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.maxp(x)
        x = x.view(-1, 1024)

        f_id = self.fc1_id(x)
        f_exp = self.fc1_exp(x)

        f_id1 = F.relu(self.fc2_id(f_id))
        s_id = self.fc3_id(f_id1)
        f_exp1 = F. relu(self.fc2_exp(f_exp))
        s_exp = self.fc3_exp(f_exp1)

        s_id = s_id.view(batchsize, 3, self.num_points).transpose(1, 2).contiguous()
        s_exp = s_exp.view(batchsize, 3, self.num_points).transpose(1, 2).contiguous()

        s_pred = torch.add(s_id, s_exp)
        return s_id, s_exp, s_pred


def normals_cal(points_data, faces):
    ver_1 = points_data[faces[:, 0]]
    ver_2 = points_data[faces[:, 1]]
    ver_3 = points_data[faces[:, 2]]
    u = ver_2 - ver_1
    v = ver_3 - ver_1
    normals = torch.cross(u, v)
    normal_norms = torch.norm(normals, dim=1).reshape(-1, 1)
    normals = normals / torch.where(normal_norms != 0, normal_norms, torch.tensor([1.]).cuda())
    return normals


def edge_cal(points_data, faces):
    v_1 = points_data[faces[:, 0]]
    v_2 = points_data[faces[:, 1]]
    v_3 = points_data[faces[:, 2]]

    def edge_length(point_a, point_b):
        return torch.norm(point_a - point_b, dim=1)

    edge_0 = edge_length(v_1, v_2)
    edge_1 = edge_length(v_2, v_3)
    edge_2 = edge_length(v_1, v_3)

    return edge_0, edge_1, edge_2


def log_writing(logfile, str_written):
    logfile.write(str_written + '\n')
    logfile.flush()


def get_loss(s_pred, faces, label_points, lambda1, lambda2):
    label_points = torch.squeeze(label_points)
    shp_pred = s_pred.squeeze(0)
    s_target = label_points.unsqueeze(0)

    normals_pred = normals_cal(shp_pred, faces)
    normals_target = normals_cal(label_points, faces)

    edge_0_pred, edge_1_pred, edge_2_pred = edge_cal(shp_pred, faces)
    edge_0_target, edge_1_target, edge_2_target = edge_cal(label_points, faces)

    # L1 loss for vertices
    l_vt = torch.sum(torch.abs(s_target - s_pred))

    # l_normal is the loss for surface normals
    l_normal = torch.sum(1 - torch.sum(normals_target * normals_pred, dim=1)) / normals_target.shape[0]

    # l_edge is the loss for edge length
    l_edge = torch.mean(torch.sum(torch.abs(edge_0_pred / edge_0_target - 1)) +
                            torch.sum(torch.abs(edge_1_pred / edge_1_target - 1)) +
                            torch.sum(torch.abs(edge_2_pred / edge_2_target - 1))) / 58366

    loss_supervised = 0.001 * l_vt + lambda1 * l_normal + lambda2 * l_edge

    return loss_supervised


edge_array = torch.from_numpy(np.load("./edge_index.npy")).to(torch.long).cuda()
template_edge_lengths = torch.from_numpy(np.load("./template_edge_length.npy")).cuda()


def get_loss_real(s_pred, faces_real, faces_syn, label_points, chamfer_dist, laplacian_loss, lambda1, lambda2, lambda3, epsilon=0.001):

    dist1, dist2, idx1, idx2 = chamfer_dist(s_pred, label_points)

    # Chamfer distance as unsupervised vertices loss
    vertices_unsupervised = torch.sum(torch.where(dist1 <= epsilon, dist1, torch.tensor([0.]).cuda())) + \
        torch.sum(torch.where(dist2 <= epsilon, dist2, torch.tensor([0.]).cuda()))

    # vertices_unsupervised = torch.sum(dist1) + torch.sum(dist2)
    # vertices_unsupervised = chamfer_distance(s_pred, label_points, point_reduction="sum")[0]

    # x_nn_idx = knn_points(s_pred, label_points, K=1).idx
    # x_nn_idx = torch.squeeze(x_nn_idx, dim=2)

    closest_points = label_points[0, idx1[0].to(torch.long)]  # Assume batch size == 1.

    # normals_pred = normals_cal(s_pred[0], faces)  # Assume batch size == 1.
    # normals_target = normals_cal(closest_points, faces)

    vertex_normals_pred = Meshes([s_pred[0]], [torch.from_numpy(faces_syn).cuda()]).verts_normals_list()[0]
    vertex_normals_target = Meshes([closest_points], [torch.from_numpy(faces_real).cuda()]).verts_normals_list()[0]

    normal_unsupervised = torch.sum(-torch.sum(vertex_normals_target * vertex_normals_pred, dim=1) + 1) / \
        vertex_normals_target.shape[0]  # Mean normal loss.

    # edge_0_pred, edge_1_pred, edge_2_pred = edge_cal(s_pred[0], faces_syn)  # Assume batch size == 1.
    # edge_0_target, edge_1_target, edge_2_target = edge_cal(label_points[0], faces_real)  # Assume batch size == 1.
    # edge_0_target, edge_1_target, edge_2_target = edge_cal(template, faces_syn)

    pred_edge_lengths = torch.norm(s_pred[0][edge_array[:, 0]] - s_pred[0][edge_array[:, 1]], dim=1)

    edges_unsupervised = torch.sum(torch.abs(pred_edge_lengths / template_edge_lengths - 1)) / template_edge_lengths.size(0)

    # def calc_edge_loss(pred_length, target_length):
    #     inf_count = torch.sum(target_length == 0)  # Remove zero edges into average counting.+
    #
    #     target_length = torch.where(target_length == 0, pred_length, target_length)
    #     div = pred_length / target_length
    #     res = torch.abs(div - 1)
    #     res = torch.sum(res) / (res.shape[0] - inf_count)
    #     return res
    #
    # # edges_unsupervised = (calc_edge_loss(edge_0_pred, edge_0_target) +
    # #                       calc_edge_loss(edge_1_pred, edge_1_target) +
    # #                       calc_edge_loss(edge_2_pred, edge_2_target)) / 3
    #
    # edges_unsupervised = torch.mean(torch.sum(torch.abs(edge_0_pred / edge_0_target - 1)) +
    #                                 torch.sum(torch.abs(edge_1_pred / edge_1_target - 1)) +
    #                                 torch.sum(torch.abs(edge_2_pred / edge_2_target - 1))) / 58366

    lapla_loss = laplacian_loss(s_pred)

    print(vertices_unsupervised, lambda1 * normal_unsupervised, lambda2 * edges_unsupervised, lambda3 * lapla_loss)
    loss_unsupervised = vertices_unsupervised + lambda1 * normal_unsupervised + lambda2 * edges_unsupervised + lambda3 * lapla_loss
    return loss_unsupervised


# if __name__ == '__main__':
#     input_variables = torch.rand((5, 29495, 3), device=device)
#     shape = DenseCor().to(device)
#     esti_shape = shape(input_variables)
#     print(esti_shape)

