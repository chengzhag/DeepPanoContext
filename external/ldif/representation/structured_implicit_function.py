import torch
import torch.nn.functional as F
import numpy as np
import tempfile
import subprocess
import os
import shutil

import trimesh

from external.ldif.util import camera_util
from external.ldif.util import file_util
from external.PIFu.lib import mesh_util
from external.ldif.inference import extract_mesh
from utils.mesh_utils import MeshExtractor


class StructuredImplicit(MeshExtractor):
    def __init__(self, config, constant, center, radius, iparam, ldif=None, occnet2gaps=None):
        # (ldif.representation.structured_implicit_function.StructuredImplicit.from_activation)
        self.config = config
        self.implicit_parameter_length = config['implicit_parameter_length']
        self.element_count = config['element_count']
        self.sym_element_count = config['sym_element_count']
        self.sym_face = config.get('sym_face', 'yz')
        self.sym_sign = {'xy': [1, 1, -1], 'yz': [-1, 1, 1], 'xz': [1, -1, 1]}[self.sym_face]
        self.device = constant.device
        self.batch_size = constant.size(0)
        self.ldif = ldif
        self.occnet2gaps = occnet2gaps

        self.constants = constant
        self.radii = radius
        self.centers = center
        self.iparams = iparam

        self._temp_folder = None

    @classmethod
    def from_packed_vector(cls, config, packed_vector, ldif=None, occnet2gaps=None):
        """Parse an already packed vector (NOT a network activation)."""
        constant, center, radius, iparam = cls._unflatten(packed_vector)
        return cls(config, constant, center, radius, iparam, ldif=ldif, occnet2gaps=occnet2gaps)

    @classmethod
    def from_activation(cls, config, activation, ldif=None, occnet2gaps=None):
        activation = torch.reshape(activation, [activation.size(0), config['element_count'], -1])
        constant, center, radius, iparam = cls._unflatten(activation)
        constant = -torch.abs(constant)
        radius_var = torch.sigmoid(radius[..., :3])
        radius_var = 0.15 * radius_var
        radius_var = radius_var * radius_var
        max_euler_angle = np.pi / 4.0
        radius_rot = torch.clamp(radius[..., 3:], -max_euler_angle, max_euler_angle)
        radius = torch.cat([radius_var, radius_rot], -1)
        center = center / 2
        return cls(config, constant, center, radius, iparam, ldif=ldif, occnet2gaps=occnet2gaps)

    @classmethod
    def cat(cls, l):
        packed_vector = torch.cat([s.packed_vector for s in l])
        sample = l[0]
        s = StructuredImplicit.from_packed_vector(
            sample.config, packed_vector, sample.ldif, sample.occnet2gaps)
        return s

    def dict(self):
        return {'constant': self.constants, 'radius': self.radii, 'center': self.centers, 'iparam': self.iparams}

    @property
    def packed_vector(self):
        packed_vector = torch.cat([self.constants, self.centers, self.radii, self.iparams], -1)
        return packed_vector

    @property
    def all_centers(self):
        sym_centers = self.centers[:, :self.sym_element_count].clone()
        sym_centers[:, :, 0] *= -1  # reflect across the YZ plane
        all_centers = torch.cat([self.centers, sym_centers], 1)
        return all_centers

    @property
    def analytic_code(self):
        analytic_code = torch.cat([self.constants, self.centers, self.radii], -1)
        return analytic_code

    @property
    def world2local(self):
        tx = torch.eye(3, device=self.device).expand(self.batch_size, self.element_count, -1, -1)
        centers = self.centers.unsqueeze(-1)
        tx = torch.cat([tx, -centers], -1)
        lower_row = torch.tensor([0., 0., 0., 1.], device=self.device).expand(self.batch_size, self.element_count, 1, -1)
        tx = torch.cat([tx, lower_row], -2)

        # Compute a rotation transformation
        rotation = camera_util.roll_pitch_yaw_to_rotation_matrices(self.radii[..., 3:6]).inverse()
        diag = 1.0 / (torch.sqrt(self.radii[..., :3] + 1e-8) + 1e-8)
        scale = torch.diag_embed(diag)

        # Apply both transformations and return the transformed points.
        tx3x3 = torch.matmul(scale, rotation)
        tx3x3 = F.pad(tx3x3, [0, 1, 0, 1], "constant", 0)
        tx3x3[..., -1, -1] = 1
        return self.tile_for_symgroups(torch.matmul(tx3x3, tx))

    def savetxt(self, path):
        assert self.packed_vector.shape[0] == 1
        sif_vector = self.packed_vector.squeeze().cpu().numpy()
        sif_vector[:, 4:7] = np.sqrt(np.maximum(sif_vector[:, 4:7], 0))
        out = 'SIF\n%i %i %i\n' % (self.element_count, 0, self.implicit_parameter_length)
        for row_idx in range(self.element_count):
            row = ' '.join(10 * ['%.9g']) % tuple(sif_vector[row_idx, :10].tolist())
            symmetry = int(row_idx < self.sym_element_count)
            row += ' %i' % symmetry
            implicit_params = ' '.join(self.implicit_parameter_length * ['%.9g']) % (
                tuple(sif_vector[row_idx, 10:].tolist()))
            row += ' ' + implicit_params
            row += '\n'
            out += row
        file_util.writetxt(path, out)

    def unbind(self):
        return [StructuredImplicit.from_packed_vector(
            self.config, self.packed_vector[i:i + 1], self.ldif, self.occnet2gaps)
            for i in range(self.packed_vector.size(0))]

    def __getitem__(self, item):
        return StructuredImplicit.from_packed_vector(
            self.config, self.packed_vector[item], self.ldif, self.occnet2gaps)

    def tile_for_symgroups(self, elements):
        # Tiles an input tensor along its element dimension based on symmetry
        # (ldif.representation.structured_implicit_function._tile_for_symgroups)
        sym_elements = elements[:, :self.sym_element_count, ...]
        elements = torch.cat([elements, sym_elements], 1)
        return elements

    def generate_symgroup_samples(self, samples):
        samples = samples.unsqueeze(1).expand(-1, self.element_count, -1, -1)
        sym_samples = samples[:, :self.sym_element_count].clone()
        sym_samples *= torch.tensor(self.sym_sign, dtype=torch.float32, device=self.device)
        effective_samples = torch.cat([samples, sym_samples], 1)
        return effective_samples

    def sdf_at_samples(self, samples):
        return self.ldif(self.packed_vector, samples)

    def extract_mesh(self, resolution=None, extent=None, num_samples=None, cuda_kernel=None):
        if resolution is None:
            resolution = self.config.get('marching_cube_resolution', 64)
        if extent is None:
            extent = self.config.get('bounding_box', 0.7) + 0.5
        if num_samples is None:
            num_samples = self.config.get('num_samples', 10000)
        if cuda_kernel is None:
            cuda_kernel = self.config.get('cuda_kernel', True)

        if cuda_kernel:
            mesh = []
            for s in self.unbind():
                if self._temp_folder is None:
                    self._temp_folder = tempfile.mktemp(dir='/dev/shm')
                    os.makedirs(self._temp_folder)
                    self.ldif.module.decoder.write_occnet_file(os.path.join(self._temp_folder, 'serialized.occnet'))
                    shutil.copy('./external/ldif/ldif2mesh/ldif2mesh', self._temp_folder)
                si_path = os.path.join(self._temp_folder, 'ldif.txt')
                grd_path = os.path.join(self._temp_folder, 'grid.grd')

                s.savetxt(si_path)
                cmd = (f"{os.path.join(self._temp_folder, 'ldif2mesh')} {si_path}"
                       f" {os.path.join(self._temp_folder, 'serialized.occnet')}"
                       f' {grd_path} -resolution {resolution} -extent {extent}')
                subprocess.check_output(cmd, shell=True)
                _, volume = file_util.read_grd(grd_path)
                _, m = extract_mesh.marching_cubes(volume, extent)
                mesh.append(m)
        else:
            mesh = mesh_util.reconstruction(
                structured_implicit=self, resolution=resolution,
                b_min=np.array([-extent] * 3), b_max=np.array([extent] * 3),
                num_samples=num_samples
            )

        if self.occnet2gaps is not None:
            mesh = [m.apply_transform(t.inverse().cpu().numpy())
                    if not isinstance(m, trimesh.primitives.Sphere) else m
                    for m, t in zip(mesh, self.occnet2gaps)]
        return mesh

    @staticmethod
    def _unflatten(vector):
        return torch.split(vector, [1, 3, 6, vector.size(-1) - 10], -1)

