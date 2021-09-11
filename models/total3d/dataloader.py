import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import transforms

from models.pano3d.dataloader import IGSceneDataset, collate_fn
from utils.igibson_utils import IGScene
from configs import data_config


class PersIGSceneDataset(IGSceneDataset):

    def __init__(self, config, mode=None):
        super(PersIGSceneDataset, self).__init__(config, mode)

        # full image argumentation
        self.image_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def update_metadata(self):
        super(PersIGSceneDataset, self).update_metadata()
        layout_centroid_avg = []
        layout_size_avg = []
        for i in tqdm(range(len(self)), desc='Generating metadata of layout_centroid_avg and layout_size_avg...'):
            scene = self.get_igibson_scene(i)
            layout_bdb3d = scene['layout']['total3d']
            layout_centroid_avg.append(layout_bdb3d['centroid_total3d'])
            layout_size_avg.append(layout_bdb3d['size'])
        layout_centroid_avg = np.mean(layout_centroid_avg, axis=0)
        layout_size_avg = np.mean(layout_size_avg, axis=0)
        data_config.metadata.update({
            'layout_centroid_avg': layout_centroid_avg,
            'layout_size_avg': layout_size_avg
        })

    def __getitem__(self, index):
        est_scene, gt_scene = self.get_igibson_scene(index, ('est', 'gt'))

        gt_data = {k: gt_scene[k] for k in self._basic_info}
        if est_scene is None:
            est_data = gt_data.copy()
            est_scene = IGScene(est_data)
        else:
            est_data = est_scene.data

        if 'detector' in self.config['model']:
            # Detectron predictor API needs numpy image
            gt_data['image_np'] = {'rgb': gt_scene.image_io['rgb']}
            if 'objs' in gt_scene.data:
                gt_data['objs'] = gt_scene['objs']

        if 'layout_estimation' in self.config['model']:
            # image normalization, data argumentation
            gt_data['image_tensor'] = {'rgb': self.image_transforms(gt_scene.image_io['rgb'])}
            if 'layout' in gt_scene.data:
                gt_data['layout'] = {k: gt_scene['layout'][k] for k in ('cuboid_world', 'total3d')}

        if 'bdb3d_estimation' in self.config['model']:
            if 'objs' in gt_scene.data:
                gt_data['objs'] = gt_scene['objs']

        if 'scene_gcn' in self.config['model']:
            if 'objs' in gt_scene.data:
                gt_data['objs'] = gt_scene['objs']

        return est_data, gt_data, est_scene, gt_scene


def perspective_igibson_dataloader(config, mode='train'):
    dataloader = DataLoader(dataset=PersIGSceneDataset(config, mode),
                            num_workers=config['device']['num_workers'],
                            batch_size=config[mode]['batch_size'],
                            shuffle=(mode == 'train'),
                            collate_fn=collate_fn,
                            pin_memory=True,
                            worker_init_fn=lambda x: np.random.seed())
    return dataloader

