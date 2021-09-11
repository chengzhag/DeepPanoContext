import os

from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "evaluation_test")
        assert len(cfg.DATASETS.TEST) == 1
        evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], ("bbox", "segm"), False, output_dir=output_folder)
        return evaluator
