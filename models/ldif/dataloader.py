import numpy as np
import os

from torch.utils.data import DataLoader

from external.ldif.util import gaps_util, file_util
from ..datasets import IGObjRecDataset, collate_fn


class IGLDIFDataset(IGObjRecDataset):
    def __init__(self, config, mode):
        super(IGLDIFDataset, self).__init__(config, mode)

        for split in self.split:
            folder = split['folder']
            split['nss_points_path'] = os.path.join(folder, 'nss_points.sdf')
            split['uniform_points_path'] = os.path.join(folder, 'uniform_points.sdf')
            split['coarse_grid_path'] = os.path.join(folder, 'coarse_grid.grd')

    def __getitem__(self, index):
        sample = super(IGLDIFDataset, self).__getitem__(index)

        if self.mode == 'test':
            sample['mesh'] = file_util.read_mesh(sample['mesh_path'])
            occnet2gaps = file_util.read_txt_to_np(sample['occnet2gaps_path'])
            sample['occnet2gaps'] = np.reshape(occnet2gaps, [4, 4])
        else:
            near_surface_samples = gaps_util.read_pts_file(sample['nss_points_path'])
            p_ids = np.random.choice(near_surface_samples.shape[0],
                                     self.config['data']['near_surface_samples'],
                                     replace=False)
            near_surface_samples = near_surface_samples[p_ids, :]
            sample['near_surface_class'] = (near_surface_samples[:, 3:] > 0).astype(np.float32)
            sample['near_surface_samples'] = near_surface_samples[:, :3]

            uniform_samples = gaps_util.read_pts_file(sample['uniform_points_path'])
            p_ids = np.random.choice(uniform_samples.shape[0],
                                     self.config['data']['uniform_samples'],
                                     replace=False)
            uniform_samples = uniform_samples[p_ids, :]
            sample['uniform_class'] = (uniform_samples[:, 3:] > 0).astype(np.float32)
            sample['uniform_samples'] = uniform_samples[:, :3]

            sample['world2grid'], sample['grid'] = file_util.read_grd(sample['coarse_grid_path'])
            # from external.PIFu.lib import sample_util
            # sample_util.save_samples_truncted_prob('near_surface_samples.ply', sample['near_surface_samples'], sample['near_surface_class'])
            # sample_util.save_samples_truncted_prob('uniform_samples.ply', sample['uniform_samples'], sample['uniform_class'])

        return sample


def LDIF_dataloader(config, mode='train'):
    dataloader = DataLoader(dataset=IGLDIFDataset(config, mode),
                            num_workers=config['device']['num_workers'],
                            batch_size=config[mode]['batch_size'],
                            shuffle=(mode == 'train'),
                            collate_fn=collate_fn)
    return dataloader
