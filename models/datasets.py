import os

import torch
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from glob import glob
import shutil
from torch.utils.data import Dataset

from configs.data_config import IG56CLASSES
from utils.basic_utils import read_json
from utils.image_utils import load_image
from utils.mesh_utils import load_mesh


class Pano3DDataset(Dataset):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    patch_width = 256
    crop_width = 280
    crop_transforms = {}

    def __init__(self, config, mode=None):
        assert isinstance(config, dict)
        self.config = config

        # if save intermedia results as dataset
        save_path = config.get('log', {}).get('save_as_dataset')
        if save_path:
            mode = None
            # copy json split file to dst folder
            os.makedirs(save_path, exist_ok=True)
            split_files = glob(os.path.join(config['data']['split'], '*.json'))
            for split_file in split_files:
                shutil.copy(split_file, os.path.join(save_path, os.path.basename(split_file)))

        # merge train and test set if mode set to None
        self.mode = mode
        if mode == 'val':
            mode = ['test']
        elif mode is None:
            mode = ['train', 'test']
        elif mode in ['train', 'test']:
            mode = [mode]
        else:
            raise Exception("'mode' must be one of 'train', 'test' and None!")

        # load split from json file
        split = config['data']['split']
        if split.endswith('.json'):
            self.root = os.path.dirname(split)
            split_files = [split]
            print(f"Using specified split file {split} with {mode} mode!")
        else:
            self.root = split
            split_files = [os.path.join(self.root, m + '.json') for m in mode]
        self.split = []
        for split_file in split_files:
            if os.path.exists(split_file):
                self.split.extend(read_json(split_file))
            else:
                split = []
                for f in glob(os.path.join(self.root, '*')):
                    if f.lower().endswith(('png', 'jpg')):
                        split.append(os.path.basename(f))
                self.split = split
                break
        self.split = [os.path.join(self.root, folder) for folder in self.split]
        print(f"Dataset mode: {mode}, cameras: {len(self.split)}")

        # crop image argumentation
        self.crop_transforms['train'] = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.crop_width, self.crop_width)),
            transforms.RandomCrop((self.patch_width, self.patch_width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.crop_transforms['test'] = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.patch_width, self.patch_width)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        
        self.crop_transforms['val'] = self.crop_transforms['test']

    def __len__(self):
        return len(self.split)


class IGObjRecDataset(Pano3DDataset):

    def __init__(self, config, mode):
        super(IGObjRecDataset, self).__init__(config, mode)

        split = []
        for i_image, image in enumerate(self.split):
            folder = os.path.dirname(image)
            category = folder.split('/')[-2]
            split.append({
                'folder': folder,
                'img_path': image + '.png',
                'occnet2gaps_path': os.path.join(folder, 'orig_to_gaps.txt'),
                'mesh_path': os.path.join(folder, 'mesh_watertight.ply'),
                'sample_id': i_image,
                'class_id': IG56CLASSES.index(category),
                'class_name': category
            })
        self.split = split

    def __getitem__(self, index):
        sample_info = self.split[index]
        sample = {}

        image = load_image(sample_info['img_path'])
        sample['img'] = self.crop_transforms[self.mode](image)

        cls_codes = torch.zeros(len(IG56CLASSES), dtype=torch.float32)
        cls_codes[sample_info['class_id']] = 1
        sample['cls'] = cls_codes

        if self.mode == 'test':
            sample['mesh'] = load_mesh(sample_info['mesh_path'])

        sample.update(sample_info)
        return sample


def collate_fn(batch):
    collated_batch = {}
    for key in batch[0]:
        try:
            collated_batch[key] = default_collate([elem[key] for elem in batch])
        except TypeError:
            collated_batch[key] = [elem[key] for elem in batch]

    return collated_batch