import torch
from torch import nn
import trimesh

from external.ldif.representation.structured_implicit_function import StructuredImplicit
from models.registers import METHODS
from models.method import BaseMethod


@METHODS.register_module
class LIENLDIF(BaseMethod):

    def __init__(self, cfg):
        super(LIENLDIF, self).__init__(cfg, ['shape_encoder', 'shape_decoder'])

        # Multi-GPU setting
        for model_name in ['shape_encoder', 'shape_decoder']:
            if hasattr(self, model_name):
                setattr(self, model_name, nn.DataParallel(self.__getattr__(model_name)))

    def forward(self, data):
        lien_activation = self.shape_encoder(data['img'], data['cls'])['lien_activation']
        structured_implicit = StructuredImplicit.from_activation(
            self.cfg.config['model']['shape_decoder'], lien_activation,
            self.shape_decoder, occnet2gaps=data.get('occnet2gaps')
        )

        if 'uniform_samples' in data.keys():
            samples = torch.cat([data['near_surface_samples'], data['uniform_samples']], 1)
            len_near_surface = data['near_surface_class'].shape[1]
            sdf = structured_implicit.sdf_at_samples(samples)
            samples_class = torch.sigmoid(100 * (sdf + 0.07))
            est_data = {
                'element_centers': structured_implicit.centers,
                'near_surface_class': samples_class[:, :len_near_surface, ...],
                'uniform_class': samples_class[:, len_near_surface:, ...]
            }
        else:
            est_data = {'mesh_extractor': structured_implicit}

        return est_data

    def load_weight(self, pretrained_model):
        # for compatibility with Total3D
        compatible_model = {}
        for k, v in pretrained_model.items():
            k = k.replace('mesh_reconstruction.module.encoder', 'shape_encoder.module.encoder')
            k = k.replace('mesh_reconstruction.module.mlp', 'shape_encoder.module.mlp')
            k = k.replace('mesh_reconstruction.module.decoder', 'shape_decoder.module.decoder')
            compatible_model[k] = v
        super(LIENLDIF, self).load_weight(compatible_model)
