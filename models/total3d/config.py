from ..pano3d.config import *
from .testing import Tester
from .dataloader import perspective_igibson_dataloader

def get_dataloader(config, mode):
    return perspective_igibson_dataloader(config=config, mode=mode)

def get_tester(cfg, net, device=None):
    return Tester(cfg=cfg, net=net, device=device)
