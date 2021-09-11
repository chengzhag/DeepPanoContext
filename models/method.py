from models.registers import MODULES, LOSSES
import torch.nn as nn

class BaseMethod(nn.Module):
    '''
    Base Network Module for other networks to be registered as method
    '''
    def __init__(self, cfg, model_order):
        super(BaseMethod, self).__init__()
        self.cfg = cfg
        self.loss_fcn = {}
        self.model_names = list(cfg.config['model'].keys())

        # load models
        self.loss_fcn = {}
        for model_name in model_order:
            if model_name not in self.model_names:
                continue
            net_spec = cfg.config['model'][model_name]
            # load specific optimizer parameters
            optim_spec = self.load_optim_spec(cfg.config, net_spec)
            # load model
            subnet = MODULES.get(net_spec['module'])(cfg, optim_spec)
            if isinstance(subnet, nn.Module):
                self.add_module(model_name, subnet)
            else:
                setattr(self, model_name, subnet)
            # load corresponding loss functions
            self.loss_fcn[model_name] = LOSSES.get(
                self.cfg.config['model'][model_name].get('loss', 'BaseLoss')
            )(config=cfg.config)

        # freeze submodules
        self.freeze_modules(cfg)

    def freeze_modules(self, cfg):
        if cfg.config['mode'] == 'train':
            freeze_layers = cfg.config['train']['freeze']
            for layer in freeze_layers:
                if not hasattr(self, layer):
                    continue
                for param in getattr(self, layer).parameters():
                    param.requires_grad = False
                cfg.log_string('The module: %s is fixed.' % (layer))

    def train(self, mode=True):
        super(BaseMethod, self).train(mode)

        # freeze
        freeze_layers = self.cfg.config['train']['freeze']
        for name, child in self.named_children():
            if name in freeze_layers:
                child.train(False)

        # turn off BatchNorm if batch_size == 1.
        if self.cfg.config[self.cfg.config['mode']]['batch_size'] == 1:
            for m in self.modules():
                if m._get_name().find('BatchNorm') != -1:
                    m.eval()

        return self

    def load_weight(self, pretrained_model):
        model_dict = self.state_dict()
        pretrained_dict = {}
        for k, v in pretrained_model.items():
            if k not in model_dict:
                print(f"{k} will not be loaded because not exist in target model")
            elif model_dict[k].shape != v.shape:
                print(f"{k} will not be loaded because source shape {model_dict[k].shape} != taget shape {v.shape}")
            else:
                pretrained_dict[k] = v

        self.cfg.log_string(
            str(set([key.split('.')[0] for key in model_dict if key not in pretrained_dict])) + ' subnet missed.')
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    @staticmethod
    def load_optim_spec(config, net_spec):
        if config['mode'] == 'train':
            if 'optimizer' in net_spec.keys():
                optim_spec = net_spec['optimizer']
            else:
                optim_spec = config['optimizer']  # else load default optimizer
        else:
            optim_spec = None

        return optim_spec

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def loss(self, est_data, gt_data):
        loss_weights = self.cfg.config.get('loss_weights', {})
        loss_raw = {}
        for loss_fcn in self.loss_fcn.values():
            loss_raw.update(loss_fcn(est_data, gt_data))
        loss = {k: v * loss_weights.get(k, 1.0) for k, v in loss_raw.items()}
        total_loss = sum(loss.values())
        total_loss_unweighted = sum(loss_raw.values()).item()
        loss = {k: v.item() for k, v in loss.items()}
        return {'total': total_loss, 'total_unweighted': total_loss_unweighted, **loss}
