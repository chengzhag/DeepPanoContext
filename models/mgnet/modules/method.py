from models.mgnet.modules.mgn import MGNetMeshExtractor
from models.registers import METHODS
from models.method import BaseMethod
from torch import nn

@METHODS.register_module
class MGNet(BaseMethod):

    def __init__(self, cfg):
        super(MGNet, self).__init__(cfg, ['mesh_reconstruction'])

        # Multi-GPU setting
        self.mesh_reconstruction = nn.DataParallel(self.mesh_reconstruction)

    def forward(self, data):
        est_data = self.mesh_reconstruction(data['img'], data['cls'])

        if not self.training:
            est_data['mesh_extractor'] = MGNetMeshExtractor(est_data)

        return est_data
