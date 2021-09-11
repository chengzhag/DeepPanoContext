# Utility functions during training and testing.
# author: ynie
# date: Feb, 2020

import sys
import os
import urllib
import numpy as np
import random
import torch
from torch.utils import model_zoo
from datetime import datetime, timedelta
import traceback

from models import method_paths
from models.registers import METHODS


class CheckpointIO(object):
    '''
    load, save, resume network weights.
    '''
    def __init__(self, cfg, **kwargs):
        '''
        initialize model and optimizer.
        :param cfg: configuration file
        :param kwargs: model, optimizer and other specs.
        '''
        self.cfg = cfg
        self._module_dict = kwargs
        self._module_dict.update({'epoch': 0, 'min_loss': 1e8, 'step': 0})
        self._saved_filename = 'model_last.pth'

    @property
    def module_dict(self):
        return self._module_dict

    @property
    def saved_filename(self):
        return self._saved_filename

    @staticmethod
    def is_url(url):
        scheme = urllib.parse.urlparse(url).scheme
        return scheme in ('http', 'https')

    def get(self, key):
        return self._module_dict.get(key, None)

    def register_modules(self, **kwargs):
        ''' Registers modules in current module dictionary.
        '''
        self._module_dict.update(kwargs)

    def save(self, suffix=None, **kwargs):
        '''
        save the current module dictionary.
        :param kwargs:
        :return:
        '''
        outdict = kwargs
        for k, v in self._module_dict.items():
            if hasattr(v, 'state_dict'):
                outdict[k] = v.state_dict()
            else:
                outdict[k] = v

        if not suffix:
            filename = self.saved_filename
        else:
            filename = self.saved_filename.replace('last', suffix)

        torch.save(outdict, os.path.join(self.cfg.config['log']['path'], filename))

    def load(self, filename, *domain):
        '''
        load a module dictionary from local file or url.
        :param filename (str): name of saved module dictionary
        :return:
        '''

        if self.is_url(filename):
            return self.load_url(filename, *domain)
        else:
            return self.load_file(filename, *domain)

    def parse_checkpoint(self):
        '''
        check if resume or finetune from existing checkpoint.
        :return:
        '''
        if self.cfg.config['resume']:
            # resume everything including net weights, optimizer, last epoch, last loss.
            self.cfg.log_string('Begin to resume from the last checkpoint.')
            self.resume()
        elif self.cfg.config['finetune']:
            # only load net weights.
            self.cfg.log_string('Begin to finetune from the existing weight.')
            self.finetune()
        else:
            self.cfg.log_string('Begin to train from scratch.')

    def finetune(self):
        '''
        finetune fron existing checkpoint
        :return:
        '''
        if isinstance(self.cfg.config['weight'], str):
            weight_paths = [self.cfg.config['weight']]
        else:
            weight_paths = self.cfg.config['weight']

        for weight_path in weight_paths:
            if not os.path.exists(weight_path):
                self.cfg.log_string('Warning: finetune failed: the weight path %s is invalid. Begin to train from scratch.' % (weight_path))
            else:
                self.load(weight_path, 'net')
                self.cfg.log_string('Weights for finetuning loaded.')

    def resume(self):
        '''
        resume the lastest checkpoint
        :return:
        '''
        weight_path = self.cfg.config['weight']

        if not os.path.exists(weight_path):
            self.cfg.log_string('Warning: resume failed: No checkpoint available. Begin to train from scratch.')
        else:
            self.load(weight_path)
            self.cfg.log_string(f'Checkpoint {weight_path} resumed.')

    def load_file(self, filename, *domain):
        '''
        load a module dictionary from file.
        :param filename: name of saved module dictionary
        :return:
        '''

        if os.path.exists(filename):
            self.cfg.log_string('Loading checkpoint from %s.' % (filename))
            checkpoint = torch.load(filename)
            scalars = self.parse_state_dict(checkpoint, *domain)
            return scalars
        else:
            raise FileExistsError

    def load_url(self, url, *domain):
        '''
        load a module dictionary from url.
        :param url: url to a saved model
        :return:
        '''
        self.cfg.log_string('Loading checkpoint from %s.' % (url))
        state_dict = model_zoo.load_url(url, progress=True)
        scalars = self.parse_state_dict(state_dict, domain)
        return scalars

    def parse_state_dict(self, checkpoint, *domain):
        '''
        parse state_dict of model and return scalars
        :param checkpoint: state_dict of model
        :return:
        '''

        # if not Total3D format, throw to method to handle
        if 'net' not in checkpoint.keys():
            self.cfg.log_string('Warning: Not Total3D format, throw to method to handle')
            self._module_dict['net'].load_weight(checkpoint)

        for key, value in self._module_dict.items():

            # only load specific key names.
            if domain and (key not in domain):
                continue

            if key in checkpoint:
                if hasattr(value, 'load_state_dict'):
                    if key != 'net':
                        try:
                            value.load_state_dict(checkpoint[key])
                        except Exception as err:
                            traceback.print_exc()
                            self.cfg.log_string(f'Warining: Could not load {key}')
                            print(err)
                    else:
                        '''load weights module by module'''
                        value.load_weight(checkpoint[key])
                else:
                    self._module_dict.update({key: checkpoint[key]})
            else:
                self.cfg.log_string('Warning: Could not find %s in checkpoint!' % key)

        if not domain:
            # remaining weights in state_dict that not found in our models.
            scalars = {k:v for k,v in checkpoint.items() if k not in self._module_dict}
            if scalars:
                self.cfg.log_string('Warning: the remaining modules %s in checkpoint are not found in our current setting.' % (scalars.keys()))
        else:
            scalars = {}

        return scalars

def initiate_environment(config):
    '''
    initiate randomness.
    :param config:
    :return:
    '''
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

def load_device(cfg):
    '''
    load device settings
    :param config:
    :return:
    '''
    if cfg.config['device']['use_gpu'] and torch.cuda.is_available():
        cfg.log_string('GPU mode is on.')
        cfg.log_string('GPU Ids: %s used.' % (cfg.config['device']['gpu_ids']))
        return torch.device("cuda")
    else:
        cfg.log_string('CPU mode is on.')
        return torch.device("cpu")

def load_model(cfg, device):
    '''
    load specific network from configuration file
    :param config: configuration file
    :param device: torch.device
    :return:
    '''
    if cfg.config['method'] not in METHODS.module_dict:
        cfg.log_string('The method %s is not defined, please check the correct name.' % (cfg.config['method']))
        cfg.log_string('Exit now.')
        sys.exit(0)

    return METHODS.get(cfg.config['method'])(cfg).to(device)

def load_trainer(cfg, net, optimizer, device):
    '''
    load trainer for training and validation
    :param cfg: configuration file
    :param net: nn.Module network
    :param optimizer: torch.optim
    :param device: torch.device
    :return:
    '''
    trainer = method_paths[cfg.config['method']].config.get_trainer(cfg=cfg,
                                                                    net=net,
                                                                    optimizer=optimizer,
                                                                    device=device)
    return trainer

def load_tester(cfg, net, device):
    '''
    load tester for testing
    :param cfg: configuration file
    :param net: nn.Module network
    :param device: torch.device
    :return:
    '''
    tester = method_paths[cfg.config['method']].config.get_tester(cfg=cfg,
                                                                  net=net,
                                                                  device=device)
    return tester

def load_dataloader(config, mode):
    '''
    load dataloader
    :param config: configuration file.
    :param mode: 'train', 'val' or 'test'.
    :return:
    '''
    dataloader = method_paths[config['method']].config.get_dataloader(config=config,
                                                                      mode=mode)
    return dataloader


class ETA:
    def __init__(self, smooth=0.99, ignore_first=False):
        self.tic = datetime.now()
        self.smooth = smooth
        self.ignore_first=ignore_first
        self.speed = 0
        self.eta = None

    def __call__(self, left_steps):
        toc = datetime.now()
        if self.ignore_first:
            self.ignore_first = False
            self.tic = toc
            return None
        else:
            if self.speed > 0:
                speed = self.smooth * timedelta(seconds=self.speed) + (1 - self.smooth) * (toc - self.tic)
            else:
                speed = toc - self.tic
            self.tic = toc
            eta = speed * left_steps
            self.eta = timedelta(seconds=round(eta.total_seconds()))
            self.speed = speed.total_seconds()
            return self.eta
