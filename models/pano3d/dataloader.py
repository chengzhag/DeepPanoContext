import os
import math
import numpy as np
import collections
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.utils.data
from torchvision import transforms

from models.datasets import Pano3DDataset
from configs.data_config import IG56CLASSES
from utils.igibson_utils import IGScene
from utils.basic_utils import list_of_dict_to_dict_of_array, read_json, write_json, recursively_to, get_any_array
from configs import data_config


_force_list = {
    'layout': {'manhattan_pix', 'manhattan_world'},
    'mesh_path': True,
    'objs': {'contour': {'x': True, 'y': True}},
    'walls': {'contour': {'x': True, 'y': True}},
}
_force_cat = {'objs', 'walls'}
_stop_recursion = {'relation'}
_to_tensor = {
    'camera': True,
    'layout': {'horizon', 'manhattan_world', 'total3d'},
    'image_tensor': True,
    'objs': True,
    'walls': True,
    'relation': True
}


class IGSceneDataset(Pano3DDataset):

    _basic_info = ('name', 'scene', 'camera', 'image_path')

    def __init__(self, config, mode=None):
        super(IGSceneDataset, self).__init__(config, mode)
        self.igibson_obj_dataset = config['data'].get('igibson_obj_dataset', None)

        # full image argumentation
        self.image_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        # read metadata
        metadata_path = os.path.join(self.root, 'metadata.json')

        def read_metadata():
            metadata = read_json(metadata_path)
            metadata = {k: np.array(v) for k, v in metadata.items()}
            metadata.update(data_config.metadata)
            data_config.metadata = metadata

        if os.path.exists(metadata_path):
            read_metadata()
        else:
            if mode != 'train':
                type(self)(self.config, 'train')
                read_metadata()
            else:
                self.update_metadata()
                write_json(recursively_to(data_config.metadata, 'list'), metadata_path)

    def update_metadata(self):
        dis_max = 0.
        size_avg = collections.defaultdict(list)
        has_bdb3d = True
        for i in tqdm(range(len(self.split)), desc='Generating metadata of size_avg and dis_max...'):
            scene = self.get_igibson_scene(i)
            for obj in scene['objs']:
                if 'bdb3d' not in obj:
                    has_bdb3d = False
                    break
                bdb3d = obj['bdb3d']
                dis = scene.transform.world2campix(bdb3d)['dis']
                if dis > dis_max:
                    dis_max = dis
                size_avg[obj['classname']].append(bdb3d['size'])
        if has_bdb3d and len(self.split):
            size_avg = {k: np.mean(np.stack(v), axis=0) for k, v in size_avg.items()}
            default_size = np.mean(np.stack(size_avg.values()), axis=0)
            size_avg = np.stack([size_avg.get(k, default_size.copy()) for k in IG56CLASSES])
            data_config.metadata.update({
                'size_avg': size_avg,
                'dis_max': float(dis_max)
            })

    def __getitem__(self, index):
        est_scene, gt_scene = self.get_igibson_scene(index, ('est', 'gt'))

        gt_data = {k: gt_scene[k] for k in self._basic_info}
        if est_scene is None:
            est_data = gt_data.copy()
            est_scene = IGScene(est_data)
        else:
            est_data = est_scene.data

        if 'objs' in est_data and len(est_data['objs']) == 0:
            return self.__getitem__((index + 1) % len(self))

        if 'detector' in self.config['model']:
            # Detectron predictor API needs numpy image
            gt_data['image_np'] = {'rgb': gt_scene.image_io['rgb']}
            if 'objs' in gt_scene.data:
                gt_data['objs'] = gt_scene['objs']

        if 'layout_estimation' in self.config['model']:
            # image normalization, data argumentation
            gt_data['image_tensor'] = {'rgb': self.image_transforms(gt_scene.image_io['rgb'])}
            if 'layout' in gt_scene.data:
                gt_data['layout'] = {k: gt_scene['layout'][k] for k in ('horizon', 'manhattan_pix')}

        if 'bdb3d_estimation' in self.config['model']:
            if 'objs' in gt_scene.data:
                gt_data['objs'] = gt_scene['objs']

        if 'scene_gcn' in self.config['model']:
            if 'objs' in gt_scene.data:
                gt_data['objs'] = gt_scene['objs']

        if 'relation' in gt_scene.data and any(k in self.config['model'] for k in ('scene_gcn', 'bdb3d_estimation')):
            if 'relation' in gt_scene.data:
                gt_data['relation'] = gt_scene['relation']

        return est_data, gt_data, est_scene, gt_scene

    def get_igibson_scene(self, item, stype: (str, tuple, list)='gt'):
        pkl = self.split[item]
        if pkl.lower().endswith(('png', 'jpg')):
            gt_scene = IGScene.from_image(pkl)
            est_scene = None
        else:
            if isinstance(stype, str):
                stype = (stype, )

            gt_pkl = os.path.join(os.path.dirname(pkl), 'gt.pkl')
            if os.path.exists(gt_pkl):
                est_scene = IGScene.from_pickle(pkl) if 'est' in stype else None
                gt_scene = IGScene.from_pickle(gt_pkl, self.igibson_obj_dataset) if 'gt' in stype else None
            else:
                est_scene = None
                gt_scene = IGScene.from_pickle(pkl, self.igibson_obj_dataset) if 'gt' in stype else None

        scenes = {'est': est_scene, 'gt': gt_scene}
        if len(stype) == 1:
            return scenes[stype[0]]
        else:
            return [scenes[k] for k in stype]


def collate_fn(batch):
    if isinstance(batch, dict):
        batch = [(batch, )]
    if isinstance(batch[0], (tuple, list)):
        batch = list(map(list, zip(*batch)))
    else:
        batch = [batch]
    data_batch_scene = {k: v for k, v in zip(['est_batch', 'gt_batch', 'est_scenes', 'gt_scenes'], batch)}

    data_batch_scene.update({k: list_of_dict_to_dict_of_array(
        b,
        force_list=_force_list, force_cat=_force_cat, stop_recursion=_stop_recursion,
        to_tensor=_to_tensor,
    ) for k, b in zip(['est_data', 'gt_data'], batch[:2])})

    # add split indexes for objs and gt_objs
    for d_key, b_key in [('est_data', 'est_batch'), ('gt_data', 'gt_batch')]:
        if d_key not in data_batch_scene or b_key not in data_batch_scene:
            continue
        d, b = data_batch_scene[d_key], data_batch_scene[b_key]
        for objs_key in ('objs', 'walls'):
            if objs_key in d:
                interval = torch.tensor([len(
                    elem[objs_key] if isinstance(elem[objs_key], list)
                    else get_any_array(elem[objs_key])
                ) for elem in b]).cumsum(dim=0)
                split = torch.stack([
                    torch.cat([torch.zeros(1, dtype=interval.dtype), interval[:-1]]), interval], -1)
                d[objs_key]['split'] = split

    if 'gt_data' in data_batch_scene:
        est_data = data_batch_scene['est_data']
        # update ground truth ids of estimated objs
        if 'objs' in est_data and 'gt' in est_data['objs']:
            gt_data = data_batch_scene['gt_data']
            id_gt = est_data['objs']['gt']
            for (start, end), (start_gt, end_gt) in zip(
                    est_data['objs']['split'], gt_data['objs']['split']):
                id_gt[start:end][id_gt[start:end] >= 0] += start_gt
                assert (id_gt[start:end] < end_gt).all() \
                       and (id_gt[start:end][id_gt[start:end] >= 0] >= start_gt).all() \
                       and (id_gt[start:end] >= -1).all()

    return_list = []
    for k in ['est_data', 'gt_data', 'est_scenes', 'gt_scenes']:
        if k in data_batch_scene:
            return_list.append(data_batch_scene[k])
        else:
            break
    if len(return_list) <= 1:
        return return_list[0]
    return return_list


def Total3D_dataloader(config, mode='train'):
    dataloader = DataLoader(dataset=IGSceneDataset(config, mode),
                            num_workers=config['device']['num_workers'],
                            batch_size=config[mode]['batch_size'],
                            shuffle=(mode == 'train'),
                            collate_fn=collate_fn,
                            pin_memory=True,
                            worker_init_fn=lambda x: np.random.seed())
    return dataloader

