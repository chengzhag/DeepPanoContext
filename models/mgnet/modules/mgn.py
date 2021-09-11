# Definition of PoseNet
# author: ynie
# date: March, 2020

import pickle
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
import trimesh

from configs.data_config import IG56CLASSES
from models.registers import MODULES
from models.modules import resnet
from models.modules.resnet import model_urls
from utils.mesh_utils import MeshExtractor
from utils.net_utils import weights_init


number_pnts_on_template = 2562

# load sphere faces and points
def load_template(number):
    file_name = './models/mgnet/modules/sphere%d.pkl' % (number)

    with open(file_name, 'rb') as file:
        sphere_obj = pickle.load(file)
        sphere_points_normals = torch.from_numpy(sphere_obj['v']).float()
        sphere_faces = torch.from_numpy(sphere_obj['f']).long()
        sphere_adjacency = torch.from_numpy(sphere_obj['adjacency'].todense()).long()
        sphere_edges = torch.from_numpy(sphere_obj['edges']).long()
        sphere_edge2face = torch.from_numpy(sphere_obj['edge2face'].todense()).type(torch.uint8)
    return sphere_points_normals, sphere_faces, sphere_adjacency, sphere_edges, sphere_edge2face

sphere_points_normals, sphere_faces, sphere_adjacency, sphere_edges, sphere_edge2face = load_template(
        number_pnts_on_template)


def sample_points_on_edges(points, edges, quantity = 1, mode = 'train'):
    n_batch = edges.shape[0]
    n_edges = edges.shape[1]

    if mode == 'train':
        # if the sampling rate is larger than 1, we randomly pick points on faces.
        weights = np.diff(np.sort(np.vstack(
            [np.zeros((1, n_edges * quantity)), np.random.uniform(0, 1, size=(1, n_edges * quantity)),
             np.ones((1, n_edges * quantity))]), axis=0), axis=0)
    else:
        # if in test mode, we pick the central point on faces.
        weights = 0.5 * np.ones((2, n_edges * quantity))

    weights = weights.reshape([2, quantity, n_edges])
    weights = torch.from_numpy(weights).float().to(points.device)
    weights = weights.transpose(1, 2)
    weights = weights.transpose(0, 1).contiguous()
    weights = weights.expand(n_batch, n_edges, 2, quantity).contiguous()
    weights = weights.view(n_batch * n_edges, 2, quantity)

    left_nodes = torch.gather(points.transpose(1, 2), 1,
                              (edges[:, :, 0] - 1).unsqueeze(-1).expand(edges.size(0), edges.size(1), 3))
    right_nodes = torch.gather(points.transpose(1, 2), 1,
                              (edges[:, :, 1] - 1).unsqueeze(-1).expand(edges.size(0), edges.size(1), 3))

    edge_points = torch.cat([left_nodes.unsqueeze(-1), right_nodes.unsqueeze(-1)], -1).view(n_batch*n_edges, 3, 2)

    new_point_set = torch.bmm(edge_points, weights).contiguous()
    new_point_set = new_point_set.view(n_batch, n_edges, 3, quantity)
    new_point_set = new_point_set.transpose(2, 3).contiguous()
    new_point_set = new_point_set.view(n_batch, n_edges * quantity, 3)
    new_point_set = new_point_set.transpose(1, 2).contiguous()
    return new_point_set


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size = 2500, output_dim = 3):
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(bottleneck_size, bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(bottleneck_size, bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(bottleneck_size//2, bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(bottleneck_size//4, output_dim, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size//4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class EREstimate(nn.Module):
    def __init__(self, bottleneck_size=2500, output_dim = 3):
        super(EREstimate, self).__init__()
        self.conv1 = torch.nn.Conv1d(bottleneck_size, bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(bottleneck_size, bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(bottleneck_size//2, bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(bottleneck_size//4, output_dim, 1)

        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size//4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x


@MODULES.register_module
class DensTMNet(nn.Module):

    def __init__(self, cfg, optim_spec=None, bottleneck_size=1024,
                 pretrained_encoder=True, threshold=0.2, factor=1):
        super(DensTMNet, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''Module parameters'''
        self.threshold = threshold
        self.factor = factor
        self.num_points = number_pnts_on_template
        model_config = cfg.config['model']['mesh_reconstruction']
        self.subnetworks = model_config['tmn_subnetworks']
        self.train_e_e = model_config['with_edge_classifier']
        self.face_samples = model_config['face_samples']

        '''Modules'''
        self.encoder = resnet.resnet18_full(pretrained=False, num_classes=1024,
                                            input_channels=4 if cfg.config['data'].get('mask', False) else 3)
        self.decoders = nn.ModuleList(
            [PointGenCon(bottleneck_size=3 + bottleneck_size + len(IG56CLASSES)) for i in range(0, self.subnetworks)])

        if self.train_e_e:
            self.error_estimators = nn.ModuleList(
                [EREstimate(bottleneck_size=3 + bottleneck_size + len(IG56CLASSES), output_dim=1) for i in range(0, max(self.subnetworks-1, 1))])

        # initialize weight
        self.apply(weights_init)

        # initialize resnet
        if pretrained_encoder:
            pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
            model_dict = self.encoder.state_dict()
            if pretrained_dict['conv1.weight'].shape != model_dict['conv1.weight'].shape:
                model_dict['conv1.weight'][:,:3,...] = pretrained_dict['conv1.weight']
                pretrained_dict.pop('conv1.weight')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and not k.startswith('fc.')}
            model_dict.update(pretrained_dict)
            self.encoder.load_state_dict(model_dict)

    def forward(self, image, size_cls):
        mode = 'train' if self.training else 'test'
        threshold = self.threshold
        factor = self.factor
        device = image.device

        n_batch = image.size(0)
        n_edges = sphere_edges.shape[0]

        # image encoding
        image = image.contiguous()
        afeature = self.encoder(image)
        code = torch.cat([afeature, size_cls], 1)

        if not self.training:
            current_faces = sphere_faces.clone().unsqueeze(0).to(device)
            current_faces = current_faces.repeat(n_batch, 1, 1)
        else:
            current_faces = None

        current_edges = sphere_edges.clone().unsqueeze(0).to(device)
        current_edges = current_edges.repeat(n_batch, 1, 1)

        current_shape_grid = sphere_points_normals[:, :3].t().expand(n_batch, 3, self.num_points).to(device)

        # outputs for saving
        out_shape_points = []
        out_sampled_mesh_points = []
        out_indicators = []

        # boundary faces for boundary refinement
        boundary_point_ids = torch.zeros(size=(n_batch, self.num_points), dtype=torch.bool).to(device)
        remove_edges_list = []

        output = {
            'mesh_coordinates_results': out_shape_points, 'points_from_edges': out_sampled_mesh_points,
            'point_indicators': out_indicators, 'output_edges': current_edges,
            'boundary_point_ids': boundary_point_ids, 'faces': current_faces
        }

        # AtlasNet deformation + topoly modification
        for i in range(self.subnetworks):
            current_image_grid = code.unsqueeze(2).expand(code.size(0), code.size(1),
                                                           current_shape_grid.size(2)).contiguous()
            current_image_grid = torch.cat((current_shape_grid, current_image_grid), 1).contiguous()
            current_shape_grid = current_shape_grid + self.decoders[i](current_image_grid)

            # save deformed point cloud
            out_shape_points.append(current_shape_grid)

            if i == self.subnetworks - 1 and self.subnetworks > 1:
                remove_edges_list = [item for item in remove_edges_list if len(item)]
                if remove_edges_list:
                    remove_edges_list = torch.unique(torch.cat(remove_edges_list), dim=0)
                    for batch_id in range(n_batch):
                        rm_edges = remove_edges_list[remove_edges_list[:, 0] == batch_id, 1]
                        if len(rm_edges) > 0:
                            rm_candidates, counts = torch.unique(sphere_edges[rm_edges], return_counts=True)
                            boundary_ids = counts < sphere_adjacency[rm_candidates - 1].sum(1)
                            boundary_point_ids[batch_id][rm_candidates[boundary_ids] - 1] = 1

                return output

            if self.train_e_e:
                # sampling from deformed mesh
                sampled_points = sample_points_on_edges(current_shape_grid, current_edges, quantity=self.face_samples, mode=mode)

                # save sampled points from deformed mesh
                out_sampled_mesh_points.append(sampled_points)

                # preprare for face error estimation
                current_image_grid = code.unsqueeze(2).expand(code.size(0), code.size(1), sampled_points.size(2)).contiguous()
                current_image_grid = torch.cat((sampled_points, current_image_grid), 1).contiguous()

                # estimate the distance from deformed points to gt mesh.
                indicators = self.error_estimators[i](current_image_grid)
                indicators = indicators.view(n_batch, 1, n_edges, self.face_samples)
                indicators = indicators.squeeze(1)
                indicators = torch.mean(indicators, dim=2)

                # save estimated distance values from deformed points to gt mesh.
                out_indicators.append(indicators)
                # remove faces and modify the topology
                remove_edges = torch.nonzero(torch.sigmoid(indicators) < threshold)
                remove_edges_list.append(remove_edges)

                for batch_id in range(n_batch):
                    rm_edges = remove_edges[remove_edges[:, 0] == batch_id, 1]
                    if len(rm_edges)>0:
                        # cutting edges in training
                        current_edges[batch_id][rm_edges, :] = 1
                        if not self.training:
                            current_faces[batch_id][sphere_edge2face[rm_edges].sum(0).type(torch.bool), :] = 1

                threshold *= factor

        return output


class MGNetMeshExtractor(MeshExtractor):
    def __init__(self, data):
        self.data = data

    def extract_mesh(self):
        vertices_meshes = self.data['mesh_coordinates_results'][-1].transpose(1, 2).detach().cpu().numpy()
        faces_meshes = self.data['faces'].detach().cpu().numpy() - 1
        meshes = [trimesh.Trimesh(vertices, faces) for vertices, faces in zip(vertices_meshes, faces_meshes)]
        return meshes
