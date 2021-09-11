import os
import math
import numpy as np
import collections
from tqdm import tqdm
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
import torch.utils.data
from torchvision import transforms

from ..pano3d.dataloader import IGSceneDataset, collate_fn
from models.datasets import Pano3DDataset
from configs.data_config import IG56CLASSES
from utils.igibson_utils import IGScene
from utils.basic_utils import list_of_dict_to_dict_of_array, read_json, write_json, recursively_to, get_any_array
from configs import data_config


class MultiViewDataset(IGSceneDataset):

    def __init__(self, config, mode=None):
        super(MultiViewDataset, self).__init__(config, mode)

        # read matching pairs and affinity
        matching_path = os.path.join(self.root, f"{mode}_pairs.json")
        if os.path.exists(matching_path):
            pairs = read_json(matching_path)
            self.pairs = [{'pair': p['pair'], 'affinity': np.float32(p['affinity'])} for p in pairs]
        else:
            # group cameras by rooms
            houses_rooms_scenes = collections.defaultdict(lambda :collections.defaultdict(list))
            for i in tqdm(range(len(self.split)), desc='Reading houses and rooms...'):
                gt_scene = self.get_igibson_scene(i, stype='gt')
                houses_rooms_scenes[gt_scene['scene']][gt_scene['room']].append(i)

            # match cameras in the same room
            self.pairs = []
            for rooms_scenes in tqdm(houses_rooms_scenes.values(), desc='Generating matching pairs...'):
                for scenes in rooms_scenes.values():
                    if len(scenes) < 2:
                        continue
                    scenes = {i_s: self.get_igibson_scene(i_s, stype=('est', 'gt')) for i_s in scenes}
                    for i_s1, (est_s1, gt_s1) in scenes.items():
                        for i_s2, (est_s2, gt_s2) in scenes.items():
                            if i_s1 == i_s2:
                                continue
                            if not est_s1['objs'] or not est_s2['objs']:
                                continue
                            id_obj_s1 = [gt_s1['objs'][obj['gt']]['id'] if obj['gt'] >= 0 else -1 for obj in est_s1['objs']]
                            id_obj_s2 = [gt_s2['objs'][obj['gt']]['id'] if obj['gt'] >= 0 else -1 for obj in est_s2['objs']]
                            id_obj_s1 = np.tile(np.array(id_obj_s1), (len(id_obj_s2), 1)).T
                            id_obj_s2 = np.tile(np.array(id_obj_s2), (len(id_obj_s1), 1))
                            affinity = (id_obj_s1 == id_obj_s2) & (id_obj_s1 >= 0) & (id_obj_s2 >= 0)
                            if not affinity.any():
                                # no corresponding objects between views
                                continue
                            if affinity.any(axis=0).all() or affinity.any(axis=1).all():
                                # all the objects in one view is in another view
                                continue
                            self.pairs.append({'pair': [i_s1, i_s2], 'affinity': np.float32(affinity)})
            pairs = [{'pair': p['pair'], 'affinity': p['affinity'].astype(np.int).tolist()} for p in self.pairs]
            write_json(pairs, matching_path)

    def __getitem__(self, index):
        pair_info = self.pairs[index]
        pair_info = deepcopy(pair_info)
        pair_info['views'] = [super(MultiViewDataset, self).__getitem__(i) for i in pair_info['pair']]
        return pair_info

    def __len__(self):
        return len(self.pairs)


def collate_fn_multi_view(batch):
    assert len(batch) == 1
    batch = batch[0]
    batch['affinity'] = torch.from_numpy(batch['affinity'])
    batch['views'] = [collate_fn([v]) for v in batch['views']]
    return batch


def multiview_dataloader(config, mode='train'):
    dataloader = DataLoader(dataset=MultiViewDataset(config, mode),
                            num_workers=config['device']['num_workers'],
                            batch_size=config[mode]['batch_size'],
                            shuffle=(mode == 'train'),
                            collate_fn=collate_fn_multi_view,
                            pin_memory=True,
                            worker_init_fn=lambda x: np.random.seed())
    return dataloader

