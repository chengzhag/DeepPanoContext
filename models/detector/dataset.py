from detectron2.utils.logger import setup_logger
setup_logger()

import random, argparse
from tqdm import tqdm
import os

from detectron2 import model_zoo
from detectron2.config import get_cfg as default_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog

from configs.data_config import IG56CLASSES, WIMR11CLASSES, PC12CLASSES, get_dataset_name
from utils.visualize_utils import detectron_gt_sample, visualize_igibson_detectron_gt
from utils.image_utils import show_image
from models.pano3d.dataloader import IGSceneDataset


def register_igibson_detection_dataset(path, real=None):
    dataset = get_dataset_name(path)
    for d in ["train" , "test"]:
        DatasetCatalog.register(
            f"{dataset}_{d}", lambda d=d: get_igibson_dicts(path, d))
        if dataset.startswith('igibson') or real == False:
            thing_classes = IG56CLASSES
        elif dataset.startswith(('pano_context', 'wimr')) or real == True:
            thing_classes = WIMR11CLASSES
        else:
            raise NotImplementedError
        MetadataCatalog.get(f"{dataset}_{d}").set(thing_classes=thing_classes)


def get_igibson_dicts(folder, mode):
    igibson_dataset = IGSceneDataset({'data': {'split': folder}}, mode)
    dataset_dicts = []

    for idx in tqdm(range(len(igibson_dataset)), desc='Loading iGibson'):
        igibson_scene = igibson_dataset.get_igibson_scene(idx)
        record = detectron_gt_sample(igibson_scene, idx)
        record['scene'] = igibson_scene
        dataset_dicts.append(record)

    return dataset_dicts


def get_cfg(dataset='igibson', config='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'):
    dataset = get_dataset_name(dataset)
    cfg = default_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config))
    cfg.DATASETS.TRAIN = (f"{dataset}_train",)
    cfg.DATASETS.TEST = (f"{dataset}_test",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config)  # Let training initialize from model zoo
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    num_classes = len(MetadataCatalog.get(f"{dataset}_train").get('thing_classes'))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.INPUT.FORMAT = 'RGB'
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description='Visualize iGibson detection GT.')
    parser.add_argument('--dataset', type=str, default='data/igibson',
                        help='The path of the dataset')
    args = parser.parse_args()

    register_igibson_detection_dataset(args.dataset)
    dataset_dicts = DatasetCatalog.get(f"{get_dataset_name(args.dataset)}_train")
    for sample in random.sample(dataset_dicts, 3):
        image = visualize_igibson_detectron_gt(sample)
        show_image(image)
    return

if __name__ == "__main__":
    main()
