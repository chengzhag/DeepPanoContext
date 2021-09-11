import os
import torch
from .training import Trainer
import numpy as np
import tempfile
import shutil
import subprocess
from collections import defaultdict
import trimesh

from models.testing import BaseTester
from external.ldif.inference.metrics import mesh_chamfer_via_points
from external.ldif.util.file_util import read_mesh
from external.pyTorchChamferDistance.chamfer_distance import ChamferDistance
dist_chamfer = ChamferDistance()


class Tester(BaseTester, Trainer):
    '''
    Tester object for SCNet.
    '''
    def __init__(self, cfg, net, device=None):
        super(Tester, self).__init__(cfg, net, device)
        self._temp_folder = None

    def get_metric_values(self, est_data, gt_data):
        losses = defaultdict(list)

        est_data['mesh'] = est_data['mesh_extractor'].extract_mesh()
        est_vertices = [torch.from_numpy(m.vertices).type(torch.float32).to(self.device) for m in est_data['mesh']]
        gt_vertices = np.stack([trimesh.sample.sample_surface(m, 10000)[0] for m in gt_data['mesh']])
        gt_vertices = torch.from_numpy(gt_vertices).type(torch.float32).to(self.device)  # [10000, 3]

        for index in range(len(est_vertices)):
            class_name = gt_data['class_name'][index]
            dist1, dist2 = dist_chamfer(gt_vertices[index].unsqueeze(0), est_vertices[index].unsqueeze(0))[:2]
            losses['Avg_Chamfer'].append((((torch.mean(dist1)) + (torch.mean(dist2))).item(), class_name))

            if self.cfg.config['full']:
                mesh = est_data['mesh'][index]
                if self._temp_folder is None:
                    self._temp_folder = tempfile.mktemp(dir='/dev/shm')
                    os.makedirs(self._temp_folder)
                    shutil.copy('./external/ldif/gaps/bin/x86_64/mshalign', self._temp_folder)

                # ICP mesh alignment
                output_file = os.path.join(self._temp_folder, 'output.ply')
                mesh.export(output_file)
                align_file = os.path.join(self._temp_folder, 'align.ply')
                gt_file = gt_data['mesh_path'][index]
                cmd = f"{os.path.join(self._temp_folder, 'mshalign')} {output_file} {gt_file} {align_file}"
                subprocess.check_output(cmd, shell=True)

                # Chamfer distance
                align_mesh = read_mesh(align_file)
                gt = gt_vertices[index].cpu().numpy()
                for ext, output in zip(('woICP', 'wICP'), (mesh, align_mesh)):
                    chamfer = mesh_chamfer_via_points(points1=output.sample(10000), points2=gt)
                    losses[f'chamfer_{ext}'].append((chamfer, class_name))

        return losses, (est_data, gt_data)

    def test_step(self, data):
        '''
        test by epoch
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)

        '''network forwarding'''
        est_data = self.net(data)

        loss, est_data = self.get_metric_values(est_data, data)
        return loss, est_data

    def visualize_step(self, data):
        ''' Performs a visualization step.
        '''
        est_data, gt_data = data
        for img_path, mesh in zip(gt_data['img_path'], est_data['mesh']):
            rel_dir = os.path.join(*img_path.split('/')[-3:])
            rel_dir = os.path.splitext(rel_dir)[0] + '.ply'
            rel_dir = f"{self.cfg.config['log']['vis_path']}/{rel_dir}"
            if not os.path.isdir(os.path.dirname(rel_dir)):
                os.makedirs(os.path.dirname(rel_dir))
            mesh.export(rel_dir)
        pass

    def __del__(self):
        if self._temp_folder is not None:
            shutil.rmtree(self._temp_folder)
