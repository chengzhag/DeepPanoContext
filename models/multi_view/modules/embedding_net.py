import torch
from torch import nn as nn
from torch.nn import functional as F

from models.registers import MODULES
from models.pano3d.modules.scene_gcn import FeatureExtractor


def get_affinity_matrix(embedding1, embedding2):
    embedding1 = embedding1.unsqueeze(1).expand(-1, embedding2.shape[0], -1)
    embedding2 = embedding2.expand(embedding1.shape[0], -1, -1)
    affinity = F.cosine_similarity(embedding1, embedding2, dim=-1)
    affinity = torch.sigmoid(affinity * 5)
    return affinity


class IDPredictor(nn.Module):

    def __init__(self, nz_feat):
        super(IDPredictor, self).__init__()
        self.pred_layer = nn.Linear(nz_feat, 256)
        self.sc_layer = nn.Linear(256, 128)
        self.sc_layer2 = nn.Linear(128, 64)

    def forward(self, feat):
        pred = self.pred_layer(feat)
        pred = F.relu(pred)
        pred = self.sc_layer(pred)
        pred = F.relu(pred)
        pred = self.sc_layer2(pred)
        return pred


@MODULES.register_module
class Associative3DEmbedding(FeatureExtractor, nn.Module):

    def __init__(self, cfg, optim_spec=None):
        FeatureExtractor.__init__(self, cfg, 'embedding_net')
        nn.Module.__init__(self)

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        self.id_predictor = nn.DataParallel(IDPredictor(self.obj_features_len))

    def forward(self, views):
        features = []
        for view in views:
            features.append(self._get_object_features(view, self.obj_features))
        split = [len(f) for f in features]
        features = torch.cat(features, dim=0)
        embeddings = self.id_predictor(features)
        embeddings = torch.split(embeddings, split, dim=0)
        affinity = get_affinity_matrix(embeddings[0], embeddings[1])
        return affinity
