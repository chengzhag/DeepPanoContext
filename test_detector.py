import os, random
import argparse
from PIL import Image
import numpy as np
from utils.visualize_utils import visualize_igibson_detectron_gt, visualize_igibson_detectron_pred
from utils.image_utils import save_image, show_image
from tqdm import tqdm
import shutil

from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from models.detector.dataset import register_igibson_detection_dataset, \
    get_cfg, DatasetCatalog
from configs.data_config import get_dataset_name


def main():
    parser = argparse.ArgumentParser(
        description='Finetune 2D detector with iGibson dataset.')
    parser.add_argument('--dataset', type=str, default='data/igibson',
                        help='The path of the dataset')
    parser.add_argument('--weights', type=str, default='out/detector/detector_mask_rcnn/model_final.pth',
                        help='cfg.MODEL.WEIGHTS')
    parser.add_argument('--score_thresh', type=float, default=0.7,
                        help='cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST')
    parser.add_argument('--sample', type=int, default=None,
                        help='Number of samples to be visualized')
    parser.add_argument('--output', type=str, default=None,
                        help='The path of the output dateset as inputs for other modules')
    parser.add_argument('--visualize', default=False, action='store_true',
                        help="Save visualization of detection results and GT to the '--weights' folder")
    parser.add_argument('--split', type=str, default=['train', 'test'], nargs='+',
                        help='Which train/test (default all) split of iGibson dataset to be tested.')
    parser.add_argument('--strict', default=False, action='store_true',
                        help='Strictly match predicted detection and ground truth bounding box with the same label')
    parser.add_argument('--min_iou', type=float, default=0.1,
                        help='Min IoU threshold to match predicted detection and ground tructh bounding box')
    parser.add_argument('--config', type=str, default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
                        help='Detectron yaml configuration file')
    args = parser.parse_args()
    if args.sample:
        args.visualize=True

    dataset = get_dataset_name(args.dataset)
    register_igibson_detection_dataset(args.dataset)
    cfg = get_cfg(args.dataset, args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_thresh
    predictor = DefaultPredictor(cfg)

    for split in args.split:
        dataset_split = f'{dataset}_{split}'

        if args.output or args.visualize:

            dataset_dicts = DatasetCatalog.get(dataset_split)

            if args.sample is not None:
                dataset_dicts = random.sample(dataset_dicts, args.sample)

            for sample in tqdm(dataset_dicts):
                image = np.array(Image.open(sample["file_name"]))
                outputs = predictor(image)

                # visualize detection output and GT
                vis_pred = visualize_igibson_detectron_pred(outputs, image, dataset_split)
                vis_gt = visualize_igibson_detectron_gt(sample, image, dataset_split)

                # save visualization
                source_path = sample['file_name'].split('/')
                save_path = os.path.join(
                    os.path.dirname(args.weights),
                    f'visualization_{split}',
                    source_path[-3],
                    source_path[-2]
                )
                save_image(vis_pred, save_path + '_pred.png')
                save_image(vis_gt, save_path + '_gt.png')

                # show visualization
                if args.sample:
                    show_image(vis_pred)
                    show_image(vis_gt)

            # copy json split file to dst folder
            if args.output and not args.sample:
                split_file = split + '.json'
                shutil.copy(os.path.join(args.dataset, split_file), os.path.join(args.output, split_file))

        else:
            # evaluate on dataset split
            output_dir = os.path.join(os.path.dirname(args.weights), f'evaluation_{split}')
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            evaluator = COCOEvaluator(dataset_split, ("bbox", "segm"), False, output_dir=output_dir)
            val_loader = build_detection_test_loader(cfg, dataset_split)
            print(inference_on_dataset(predictor.model, val_loader, evaluator))


if __name__ == "__main__":
    main()
