import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import random
import collections

from models.registers import MODULES
from .merging_stage import Feedforward
from utils.basic_utils import list_of_dict_to_dict_of_array
from utils.image_utils import show_image
from utils.transform_utils import IGTransform
from utils.igibson_utils import IGScene
from external.pyTorchChamferDistance.chamfer_distance import ChamferDistance
dist_chamfer = ChamferDistance()


@MODULES.register_module
class RawAffinity:

    def __init__(self, cfg, _):
        self.cfg = cfg

    def __call__(self, est_data):
        # stitch_objects_affinity

        # calculate affinity matrix and matching
        affinity_pred = est_data['affinity'].detach().cpu().numpy()
        # affinity_pred[affinity_pred < 0.5] = 0
        # show_image(affinity_pred)
        has_matching = np.max(affinity_pred, axis=1) > 0.5
        matching = np.argmax(affinity_pred, axis=1)

        # adjust matching
        matching = list(matching)
        for i, j in enumerate(matching):
            if not has_matching[i]:
                continue
            if j == -1:
                continue
            boys = [i]
            for m, n in enumerate(matching):
                if i == m:
                    continue
                if j == n:
                    boys.append(m)
            boy_scores = affinity_pred[:, j][boys]
            boy_idx = boys[np.argmax(boy_scores)]

            for boy in boys:
                if boy != boy_idx:
                    matching[boy] = -1
                    has_matching[boy] = False

        # stitch accordingly
        merged = np.zeros(affinity_pred.shape[-1], dtype=np.bool)
        corespond_matrix = torch.zeros_like(est_data['affinity'], device=est_data['affinity'].device)
        for i in range(len(matching)):
            if has_matching[i]:
                j = matching[i]
                if merged[j]:
                    continue
                merged[j] = True
                corespond_matrix[i, j] = 1

        return corespond_matrix


@MODULES.register_module
class EdgeWpose:

    def __init__(self, cfg, _):
        self.cfg = cfg
        model_config = cfg.config['model']['stitching_stage']
        self.lambda_nomatch = model_config['lambda_nomatch']
        self.lambda_rots = model_config['lambda_rots']
        self.lambda_trans = model_config['lambda_trans']
        self.lambda_af = model_config['lambda_af']

    def __call__(self, est_data):
        # stitch_objects_edge_wpose
        # The same as stitch_objects_edge but use input relative camera pose

        # lambdas
        lambda_nomatch = self.lambda_nomatch
        lambda_rots = self.lambda_rots
        lambda_trans = self.lambda_trans
        lambda_af = self.lambda_af

        # TOP-k sample
        topk_match = 3
        upperbound_match = 128
        thres = 0.5

        # calculate affinity matrix and matching
        affinity_m = est_data['affinity'].detach().cpu().numpy()
        affinity_idx = np.argsort(affinity_m, axis=1)[:, ::-1][:, :topk_match]

        af_options = []
        for i in range(len(affinity_idx)):
            options = []
            for j in affinity_idx[i]:
                if affinity_m[i][j] <= thres:
                    continue
                options.append(j)
            af_options.append(options)

        # top k
        matching_proposals = []
        for _ in range(upperbound_match):
            matching = []
            num_nomatch = 0
            for i in range(len(affinity_idx)):
                options = []
                for op in af_options[i]:
                    if op not in matching:
                        options.append(int(op))
                options.append(-1)
                m = random.choice(options)
                if m == -1:
                    num_nomatch += 1
                matching.append(m)

            # check matching is valid
            if not is_valid_matching(matching):
                raise RuntimeError('invalid matching')

            # compute scores
            scores = []
            for i, j in enumerate(matching):
                if j == -1:
                    continue
                scores.append(affinity_m[i, j])
            scores = np.array(scores)

            matching_proposals.append([matching, num_nomatch, scores])

        # extract mesh
        views_vertices = []
        for view_data in est_data['views']:
            objs = view_data['objs']
            if 'mesh' not in objs:
                objs['mesh'] = objs['mesh_extractor'].extract_mesh()
            view_scene = IGScene.from_batch(view_data)[0]
            view_trans = IGTransform(view_scene)
            view_vertices = []
            for m, obj in zip(objs['mesh'], view_scene['objs']):
                m = view_trans.obj2frame(m, obj['bdb3d'])
                view_vertices.append(torch.from_numpy(m.vertices).type(torch.float32).cuda())
            views_vertices.append(view_vertices)

        # pick up the best combination of (rot, tran, matching)
        min_loss = float('inf')
        best_comb = None
        for matching, num_nomatch, scores in (matching_proposals):
            # for each matching
            left_objs = []
            corresponding_right_objs = []
            for i, j in enumerate(matching):
                if j == -1:
                    continue
                left_objs.append(views_vertices[0][i])
                corresponding_right_objs.append(views_vertices[1][j])
            assert (len(left_objs) == len(corresponding_right_objs))

            cd_pairs = []
            for pcd1, pcd2 in zip(left_objs, corresponding_right_objs):
                dist1, dist2 = dist_chamfer(pcd1.unsqueeze(0), pcd2.unsqueeze(0))[:2]
                cd_pairs.append(((torch.mean(dist1)) + (torch.mean(dist2))).item())
            if len(cd_pairs) == 0:  # no matching
                continue
            cd_cost = np.average(cd_pairs)

            loss = cd_cost + lambda_nomatch * num_nomatch \
                     + lambda_af * (1 - scores).mean()
            if loss < min_loss:
                min_loss = loss
                best_comb = matching

        if best_comb is None:
            # if there is no matching at all
            matching = [-1 for _ in range(len(affinity_idx))]
        else:
            matching = best_comb
        affinity = torch.zeros_like(est_data['affinity'], device=est_data['affinity'].device)
        for i, j in enumerate(matching):
            if j >= 0:
                affinity[i][j] = 1

        return affinity


def is_valid_matching(matching):
    """
    Matching proposal should not contain duplicate values except -1.
    """
    cnt = collections.Counter(matching)
    for k in cnt:
        if k != -1 and cnt[k] > 1:
            return False
    return True
