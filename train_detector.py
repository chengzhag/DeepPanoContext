import os
import argparse

from models.detector.dataset import register_igibson_detection_dataset, get_cfg
from models.detector.training import Trainer


def main():
    parser = argparse.ArgumentParser(
        description='Finetune 2D detector with iGibson dataset.')
    parser.add_argument('--dataset', type=str, default='data/igibson',
                        help='The path of the dataset')
    parser.add_argument('--batchsize', type=int, default=8,
                        help='cfg.SOLVER.IMS_PER_BATCH')
    parser.add_argument('--lr', type=float, default=0.002,
                        help='cfg.SOLVER.BASE_LR')
    parser.add_argument('--it', type=int, default=100000,
                        help='cfg.SOLVER.MAX_ITER')
    parser.add_argument('--imagebatch', type=int, default=256,
                        help='cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE')
    parser.add_argument('--out', type=str, default='out/detector/detector_mask_rcnn',
                        help='cfg.OUTPUT_DIR')
    parser.add_argument('--steps', type=int, default=[50000, 75000], nargs='+',
                        help='cfg.SOLVER.STEPS')
    parser.add_argument('--eval_period', type=int, default=2500,
                        help='cfg.TEST.EVAL_PERIOD')
    parser.add_argument('--config', type=str, default='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
                        help='Detectron yaml configuration file')
    args = parser.parse_args()

    register_igibson_detection_dataset(args.dataset)
    cfg = get_cfg(args.dataset, args.config)
    cfg.SOLVER.IMS_PER_BATCH = args.batchsize
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.it
    cfg.SOLVER.STEPS = args.steps
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.imagebatch
    cfg.OUTPUT_DIR = args.out
    cfg.TEST.EVAL_PERIOD = args.eval_period
    cfg.SOLVER.CHECKPOINT_PERIOD = args.eval_period
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()