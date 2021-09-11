import torch
from utils.basic_utils import recursively_to
from torch.cuda.amp import autocast, GradScaler


class BaseTrainer(object):
    '''
    Base trainer for all networks.
    '''
    def __init__(self, cfg, net, optimizer, device=None):
        self.cfg = cfg
        self.net = net
        self.optimizer = optimizer
        self.device = device
        self.scaler = GradScaler()

    def show_lr(self):
        '''
        display current learning rates
        :return:
        '''
        lrs = [self.optimizer.param_groups[i]['lr'] for i in range(len(self.optimizer.param_groups))]
        self.cfg.log_string('Current learning rates are: ' + str(lrs) + '.')

    def train_step(self, data):
        '''
        performs a step training
        :param data (dict): data dictionary
        :return:
        '''
        self.optimizer.zero_grad()
        # with autocast():
        loss = self.compute_loss(data)
        if loss['total'].requires_grad:
            self.scaler.scale(loss['total']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        loss['total'] = loss['total'].item()
        return loss

    def compute_loss(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        ''' Performs an evaluation step.
        '''
        raise NotImplementedError

    def to_device(self, data):
        return recursively_to(data, device=self.device)
