import numpy as np

from models.pano3d.modules.method import Pano3D
from models.registers import METHODS
from utils.image_utils import show_image


@METHODS.register_module
class MultiView(Pano3D):

    def __init__(self, cfg):
        self._model_order.extend(['embedding_net', 'stitching_stage', 'merging_stage'])
        super(MultiView, self).__init__(cfg)

    def forward(self, data):
        # show_image(data['affinity'].cpu().numpy())

        est_data = {}
        views = []
        for view in data['views']:
            view = super(MultiView, self).forward(view)
            views.append(view)
        est_data['views'] = views

        est_data['affinity'] = self.embedding_net(views)

        if not self.training:
            est_data['affinity'] = self.stitching_stage(est_data)
            # est_data['affinity'] = data['affinity']
            est_data['stitched'] = self.merging_stage(est_data)

        return est_data


