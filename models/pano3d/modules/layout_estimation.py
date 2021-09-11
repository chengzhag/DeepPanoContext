import numpy as np
from shapely.geometry import Polygon
import sys
import torch.nn as nn

from models.registers import MODULES
from external.HorizonNet.model import HorizonNet as BaseHorizon
from external.HorizonNet.misc import post_proc
from external.HorizonNet.inference import find_N_peaks
from utils.basic_utils import dict_of_array_to_list_of_dict, list_of_dict_to_dict_of_array


class HorizonNetNoPostProcess(BaseHorizon):
    def __init__(self, backbone, use_rnn):
        super(HorizonNetNoPostProcess, self).__init__(backbone, use_rnn)

    def forward(self, x):
        if x.shape[2] != 512 or x.shape[3] != 1024:
            raise NotImplementedError()

        conv_list = self.feature_extractor(x)
        feature = self.reduce_height_module(conv_list, x.shape[3] // self.step_cols)

        # rnn
        if self.use_rnn:
            feature = feature.permute(2, 0, 1)  # [w, b, c*h]
            output, hidden = self.bi_rnn(feature)  # [seq_len, b, num_directions * hidden_size]
            output = self.drop_out(output)
            output = self.linear(output)  # [seq_len, b, 3 * step_cols]
            output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)  # [seq_len, b, 3, step_cols]
            output = output.permute(1, 2, 0, 3)  # [b, 3, seq_len, step_cols]
            output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, seq_len*step_cols]
        else:
            feature = feature.permute(0, 2, 1)  # [b, w, c*h]
            output = self.linear(feature)  # [b, w, 3 * step_cols]
            output = output.view(output.shape[0], output.shape[1], 3, self.step_cols)  # [b, w, 3, step_cols]
            output = output.permute(0, 2, 1, 3)  # [b, 3, w, step_cols]
            output = output.contiguous().view(output.shape[0], 3, -1)  # [b, 3, w*step_cols]

        # output.shape => B x 3 x W
        cor = output[:, :1]  # B x 1 x W
        bon = output[:, 1:]  # B x 2 x W

        return bon, cor


@MODULES.register_module
class HorizonNet(nn.Module):
    def __init__(self, cfg, optim_spec=None):
        super(HorizonNet, self).__init__()
        self.cfg = cfg
        model_config = cfg.config['model']['layout_estimation']
        self.horizon_net = HorizonNetNoPostProcess(model_config['backbone'], model_config['use_rnn'])
        self.horizon_net = nn.DataParallel(self.horizon_net)

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

    def forward(self, x):
        bon, cor = self.horizon_net(x)
        horizon_layout = {'bon': bon, 'cor': cor}

        if self.training:
            return horizon_layout
        else:
            height, width = x.shape[2:]
            # transform pixel layout estimation to pixel manhattan world layout
            manhattan_pix = []
            layout_scenes = dict_of_array_to_list_of_dict(horizon_layout)
            for layout_scene in layout_scenes:
                try:
                    dt_cor_id, z0, z1 = horizon_to_manhattan_layout(
                        layout_scene, height, width, force_cuboid=False)
                except:
                    dt_cor_id = np.array([
                        [k // 2 * 1024, 256 - ((k % 2) * 2 - 1) * 120]
                        for k in range(8)
                    ])
                manhattan_pix.append(dt_cor_id)
            return horizon_layout, manhattan_pix


def horizon_to_manhattan_layout(horizon_layout, H, W, force_cuboid=True, min_v=None, r=0.05):
    y_bon_, y_cor_  = horizon_layout['bon'], horizon_layout['cor']

    y_bon_ = (y_bon_ / np.pi + 0.5) * H - 0.5
    y_cor_ = y_cor_[0]

    # Init floor/ceil plane
    z0 = 50
    _, z1 = post_proc.np_refine_by_fix_z(*y_bon_, z0)

    # Detech wall-wall peaks
    if min_v is None:
        min_v = 0 if force_cuboid else 0.05
    r = int(round(W * r / 2))
    N = 4 if force_cuboid else None
    xs_ = find_N_peaks(y_cor_, r=r, min_v=min_v, N=N)[0]

    # Generate wall-walls
    cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=force_cuboid)
    if not force_cuboid:
        # Check valid (for fear self-intersection)
        xy2d = np.zeros((len(xy_cor), 2), np.float32)
        for i in range(len(xy_cor)):
            xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
            xy2d[i, xy_cor[i - 1]['type']] = xy_cor[i - 1]['val']
        if not Polygon(xy2d).is_valid:
            print(
                'Fail to generate valid general layout!! '
                'Generate cuboid as fallback.',
                file=sys.stderr)
            xs_ = find_N_peaks(y_cor_, r=r, min_v=0, N=4)[0]
            cor, xy_cor = post_proc.gen_ww(xs_, y_bon_[0], z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=True)

    # Expand with btn coory
    cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])

    # Collect corner position in equirectangular
    cor_id = np.zeros((len(cor) * 2, 2), np.float32)
    for j in range(len(cor)):
        cor_id[j * 2] = cor[j, 0], cor[j, 1]
        cor_id[j * 2 + 1] = cor[j, 0], cor[j, 2]

    # # Normalized to [0, 1]
    # cor_id[:, 0] /= W
    # cor_id[:, 1] /= H

    return cor_id, z0, z1
