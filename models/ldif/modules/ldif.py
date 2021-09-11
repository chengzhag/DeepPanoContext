import torch
import torch.nn as nn
import torch.nn.functional as F
import struct
import shutil

from external.ldif.representation.structured_implicit_function import StructuredImplicit
from external.ldif.util import file_util
from external.ldif.representation import quadrics
from models.registers import MODULES
from utils.net_utils import weights_init


class BatchedCBatchNormLayer(nn.Module):
    def __init__(self, f_dim=32):
        super(BatchedCBatchNormLayer, self).__init__()
        self.fc_beta = nn.Linear(f_dim, f_dim)
        self.fc_gamma = nn.Linear(f_dim, f_dim)
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))

    def forward(self, shape_embedding, sample_embeddings):
        beta = self.fc_beta(shape_embedding)
        gamma = self.fc_gamma(shape_embedding)
        if self.training:
            batch_mean, batch_variance = sample_embeddings.mean().detach(), sample_embeddings.var().detach()
            self.running_mean = 0.995 * self.running_mean + 0.005 * batch_mean
            self.running_var = 0.995 * self.running_var + 0.005 * batch_variance
        sample_embeddings = (sample_embeddings - self.running_mean) / torch.sqrt(self.running_var + 1e-5)

        out = gamma.unsqueeze(1) * sample_embeddings + beta.unsqueeze(1)

        return out


class BatchedOccNetResnetLayer(nn.Module):
    def __init__(self, f_dim=32):
        super(BatchedOccNetResnetLayer, self).__init__()
        self.bn1 = BatchedCBatchNormLayer(f_dim=f_dim)
        self.fc1 = nn.Linear(f_dim, f_dim)
        self.bn2 = BatchedCBatchNormLayer(f_dim=f_dim)
        self.fc2 = nn.Linear(f_dim, f_dim)

    def forward(self, shape_embedding, sample_embeddings):
        sample_embeddings = self.bn1(shape_embedding, sample_embeddings)
        init_sample_embeddings = sample_embeddings

        sample_embeddings = torch.relu(sample_embeddings)
        sample_embeddings = self.fc1(sample_embeddings)
        sample_embeddings = self.bn2(shape_embedding, sample_embeddings)

        sample_embeddings = torch.relu(sample_embeddings)
        sample_embeddings = self.fc2(sample_embeddings)

        return init_sample_embeddings + sample_embeddings


class OccNetDecoder(nn.Module):
    def __init__(self, f_dim=32):
        super(OccNetDecoder, self).__init__()
        self.fc1 = nn.Linear(3, f_dim)
        self.resnet = BatchedOccNetResnetLayer(f_dim=f_dim)
        self.bn = BatchedCBatchNormLayer(f_dim=f_dim)
        self.fc2 = nn.Linear(f_dim, 1)

    def write_occnet_file(self, path):
        """Serializes an occnet network and writes it to disk."""
        f = file_util.open_file(path, 'wb')

        def write_fc_layer(layer):
            weights = layer.weight.t().cpu().numpy()
            biases = layer.bias.cpu().numpy()
            f.write(weights.astype('f').tostring())
            f.write(biases.astype('f').tostring())

        def write_cbn_layer(layer):
            write_fc_layer(layer.fc_beta)
            write_fc_layer(layer.fc_gamma)
            running_mean = layer.running_mean.item()
            running_var = layer.running_var.item()
            f.write(struct.pack('ff', running_mean, running_var))

        # write_header
        f.write(struct.pack('ii', 1, self.fc1.out_features))
        # write_input_layer
        write_fc_layer(self.fc1)
        # write_resnet
        write_cbn_layer(self.resnet.bn1)
        write_fc_layer(self.resnet.fc1)
        write_cbn_layer(self.resnet.bn2)
        write_fc_layer(self.resnet.fc2)
        # write_cbn_layer
        write_cbn_layer(self.bn)
        # write_activation_layer
        weights = self.fc2.weight.t().cpu().numpy()
        bias = self.fc2.bias.data.item()
        f.write(weights.astype('f').tostring())
        f.write(struct.pack('f', bias))
        f.close()

    def forward(self, embedding, samples):
        sample_embeddings = self.fc1(samples)
        sample_embeddings = self.resnet(embedding, sample_embeddings)
        sample_embeddings = self.bn(embedding, sample_embeddings)
        vals = self.fc2(sample_embeddings)
        return vals


@MODULES.register_module
class LDIF(nn.Module):
    def __init__(self, cfg, optim_spec=None):
        super(LDIF, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''Module parameters'''
        self.cfg = cfg
        self.config = cfg.config['model']['shape_decoder']
        self.element_count = self.config['element_count']
        self.sym_element_count = self.config['sym_element_count']
        self.implicit_parameter_length = self.config['implicit_parameter_length']
        # self.element_embedding_length = 10 + self.implicit_parameter_length
        # self.config['analytic_code_len'] = 10 * self.element_count
        # self.config['structured_implicit_vector_len'] = self.element_embedding_length * self.element_count
        self._temp_folder = None

        '''Modules'''
        self.decoder = OccNetDecoder(f_dim=self.implicit_parameter_length)

        # initialize weight
        self.apply(weights_init)

    def forward(self, packed_vector, samples):
        structured_implicit = StructuredImplicit.from_packed_vector(self.config, packed_vector)

        # (ldif.representation.structured_implicit_function.StructuredImplicit.class_at_samples)
        effective_constants = structured_implicit.tile_for_symgroups(structured_implicit.constants)
        effective_centers = structured_implicit.tile_for_symgroups(structured_implicit.centers)
        effective_radii = structured_implicit.tile_for_symgroups(structured_implicit.radii)

        effective_samples = structured_implicit.generate_symgroup_samples(samples)
        constants_quadrics = torch.zeros(
            effective_constants.size(0), self.element_count + self.sym_element_count, 4, 4, device=samples.device)
        constants_quadrics[:, :, -1:, -1] = effective_constants

        per_element_constants, per_element_weights = quadrics.compute_shape_element_influences(
            constants_quadrics, effective_centers, effective_radii, effective_samples)

        # We currently have constants, weights with shape:
        # [batch_size, element_count, sample_count, 1].
        # We need to use the net to get a same-size grid of offsets.
        # The input samples to the net must have shape
        # [batch_size, element_count, sample_count, 3], while the current samples
        # have shape [batch_size, sample_count, 3]. This is because each sample
        # should be evaluated in the relative coordinate system of the
        # The world2local transformations for each element. Shape [B, EC, 4, 4].
        local_samples = torch.matmul(F.pad(effective_samples, [0, 1], "constant", 1),
                                     structured_implicit.world2local.transpose(-1, -2))[..., :3]

        implicit_parameters = structured_implicit.tile_for_symgroups(structured_implicit.iparams)
        batch_size, element_count, element_embedding_length = list(implicit_parameters.shape)
        sample_count = local_samples.shape[-2]
        batched_parameters = torch.reshape(implicit_parameters, [batch_size * element_count, element_embedding_length])
        batched_samples = torch.reshape(local_samples, [batch_size * element_count, sample_count, -1])
        batched_vals = self.decoder(batched_parameters, batched_samples)
        implicit_values = torch.reshape(batched_vals, [batch_size, element_count, sample_count, 1])

        residuals = 1 + implicit_values
        local_decisions = per_element_constants * per_element_weights * residuals
        local_weights = per_element_weights
        sdf = torch.sum(local_decisions, 1)

        return sdf

    def __del__(self):
        if self._temp_folder is not None:
            shutil.rmtree(self._temp_folder)

