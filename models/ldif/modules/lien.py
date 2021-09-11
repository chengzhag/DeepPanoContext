import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from models.modules.resnet import model_urls
from models.registers import MODULES
from configs.data_config import IG56CLASSES
from models.modules import resnet


@MODULES.register_module
class LIEN(nn.Module):
    def __init__(self, cfg, optim_spec=None):
        super(LIEN, self).__init__()

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''Module parameters'''
        self.cfg = cfg
        model_config = cfg.config['model']['shape_encoder']
        bottleneck_size = model_config['bottleneck_size']

        '''Modules'''
        self.encoder = resnet.resnet18_full(pretrained=False, num_classes=bottleneck_size, input_channels=3)
        self.mlp = nn.Sequential(
            nn.Linear(bottleneck_size + len(IG56CLASSES), bottleneck_size), nn.LeakyReLU(0.2, True),
            nn.Linear(bottleneck_size, bottleneck_size), nn.LeakyReLU(0.2, True),
            nn.Linear(bottleneck_size, model_config['shape_code_length'])
        )

        # initialize resnet
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = self.encoder.state_dict()
        if pretrained_dict['conv1.weight'].shape != model_dict['conv1.weight'].shape:
            model_dict['conv1.weight'][:, :3, ...] = pretrained_dict['conv1.weight']
            pretrained_dict.pop('conv1.weight')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and not k.startswith('fc.')}
        model_dict.update(pretrained_dict)
        self.encoder.load_state_dict(model_dict)

    def forward(self, image, cls_code):
        lien_afeature = self.encoder(image)
        embedding = torch.cat([lien_afeature, cls_code], 1)
        lien_activation = self.mlp(embedding)
        return {'lien_afeature': lien_afeature, 'lien_activation': lien_activation}
