#!/usr/bin/env python3
"""
Train a Faster R-CNN model on a custom digit detection dataset using Detectron2.

This script registers the training and validation datasets (in COCO format),
sets up the model configuration using a model zoo config file, and trains the model.
A custom Trainer (MyTrainer) is used that adds a BestCheckpointer hook to track the best
model (based on "bbox/AP"). Parameters can be overridden via command-line arguments.
"""

import os
import argparse

# All Detectron2 imports are placed at the top.
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, BestCheckpointer
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances

# Set up the logger.
setup_logger()


def get_parser():
    """
    Create a command-line argument parser.

    Returns:
        argparse.ArgumentParser: The argument parser.
    """
    parser = argparse.ArgumentParser(
        description=("Train Faster R-CNN on a custom digit detection dataset "
                     "using Detectron2.")
    )
    parser.add_argument(
        "--config-file",
        default="COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of classes (default: 10)"
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/anchor/X_8_256",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=50000,
        help="Maximum number of training iterations (default: 50000)"
    )
    parser.add_argument(
        "--eval-period",
        type=int,
        default=2000
    )
    parser.add_argument(
        "--ims-per-batch",
        type=int,
        default=8
    )
    parser.add_argument(
        "--checkpoint-period",
        type=int,
        default=50000
    )
    parser.add_argument(
        "--base-lr",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--rpn_box_reg",
        default="ciou",
        help="Options are: smooth_l1, giou, diou, ciou"
    )
    parser.add_argument(
        "--roi_box_reg",
        default="ciou",
        help="Options are: smooth_l1, giou, diou, ciou"
    )
    parser.add_argument(
        "--anchor_size",
        nargs='+', type=int,
        default=[8, 16, 32, 64, 128, 256]
    )
    parser.add_argument(
        "--device",
        default="cuda:0"
    )
    return parser


# Register the training and validation datasets in COCO format.
register_coco_instances(
    "my_dataset_train", {}, "data/train.json", "data/train"
)
register_coco_instances(
    "my_dataset_val", {}, "data/valid.json", "data/valid"
)


class MyTrainer(DefaultTrainer):
    """
    Custom Trainer that extends DefaultTrainer by adding a BestCheckpointer hook
    based on the 'bbox/AP' metric.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Build and return a COCOEvaluator.

        Args:
            cfg (CfgNode): The configuration.
            dataset_name (str): The dataset name.
            output_folder (str): Directory to store evaluation outputs.

        Returns:
            COCOEvaluator: The evaluator instance.
        """
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        """
        Build training hooks, including BestCheckpointer.

        Returns:
            list: A list of training hooks.
        """
        hooks = super().build_hooks()
        hooks.append(
            BestCheckpointer(
                self.cfg.TEST.EVAL_PERIOD, self.checkpointer, "bbox/AP"
            )
        )
        return hooks


def main():
    """Main function: parse arguments, set up configuration, and start training."""
    parser = get_parser()
    args = parser.parse_args()

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.config_file))

    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.config_file)

    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.TEST.EVAL_PERIOD = args.eval_period
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    cfg.SOLVER.BASE_LR = args.base_lr

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes

    cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = args.rpn_box_reg
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = args.roi_box_reg

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [args.anchor_size]

    # Disable random flip during training.
    cfg.INPUT.RANDOM_FLIP = "none"
    # Set visualization period.
    cfg.VIS_PERIOD = 10000
    cfg.MODEL.DEVICE = args.device

    # Save the configuration to a file.
    config_file_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with open(config_file_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(cfg.dump())

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
