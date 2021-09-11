# Configure trainer and tester
# author: ynie
# date: Feb, 2020
from ..ldif.training import Trainer
from .dataloader import MGNet_dataloader
from ..ldif.testing import Tester as LDIF_Tester

def get_trainer(cfg, net, optimizer, device=None):
    return Trainer(cfg=cfg, net=net, optimizer=optimizer, device=device)

def get_tester(cfg, net, device=None):
    return LDIF_Tester(cfg=cfg, net=net, device=device)

def get_dataloader(config, mode):
    return MGNet_dataloader(config=config, mode=mode)