import os
from torch.utils.data import DataLoader
import numpy as np

from ..datasets import IGObjRecDataset, collate_fn


class IGMGNDataset(IGObjRecDataset):
    def __init__(self, config, mode):
        super(IGMGNDataset, self).__init__(config, mode)

        for split in self.split:
            folder = split['folder']
            split['densities_path'] =  os.path.join(folder, 'densities.mgn')
            split['gt_3dpoints_path'] =  os.path.join(folder, 'gt_3dpoints.mgn')

    def __getitem__(self, index):
        sample = super(IGMGNDataset, self).__getitem__(index)

        sample['sequence_id'] = sample['sample_id']
        sample['mesh_points'] = np.fromfile(
            sample['gt_3dpoints_path'], dtype=np.float).reshape(-1, 3).astype(np.float32)
        sample['densities'] = np.fromfile(
            sample['densities_path'], dtype=np.float).astype(np.float32)

        if self.mode == 'train':
            p_ids = np.random.choice(sample['mesh_points'].shape[0], 5000, replace=False)
            sample['mesh_points'] = sample['mesh_points'][p_ids, :]
            sample['densities'] = sample['densities'][p_ids]

        return sample


def MGNet_dataloader(config, mode='train'):
    dataloader = DataLoader(dataset=IGMGNDataset(config, mode),
                            num_workers=config['device']['num_workers'],
                            batch_size=config[mode]['batch_size'],
                            shuffle=(mode == 'train'),
                            collate_fn=collate_fn)
    return dataloader
