import os
import argparse
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer, BestCheckpointer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Detectron2 with best AP checkpointing")
    parser.add_argument("--dataset_name", default="my_dataset_train", type=str)
    parser.add_argument("--json_path", default="data/instances_train.json", type=str)
    parser.add_argument("--img_dir", default="data/train_img", type=str)
    
    parser.add_argument("--max_iter", default=50000, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval_period", default=1000, type=int, help="Evaluation period in iterations")

    parser.add_argument( "--anchor_size", nargs='+', type=int, default=[8, 16, 32, 64, 128])

    parser.add_argument("--model", default="mask_rcnn_X_101_32x8d_FPN_3x.yaml", type=str)
    parser.add_argument("--output_dir", default="checkpoints/anchor/X_anchor_8_128", type=str)
    return parser.parse_args()


class TrainerWithBestAP(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(
            BestCheckpointer(
                self.cfg.TEST.EVAL_PERIOD, self.checkpointer, "bbox/AP"
            )
        )
        return hooks


def main():
    args = parse_args()

    # Register dataset
    register_coco_instances(args.dataset_name, {}, args.json_path, args.img_dir)
    MetadataCatalog.get(args.dataset_name)
    DatasetCatalog.get(args.dataset_name)

    # Build config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-InstanceSegmentation/{args.model}"))
    cfg.DATASETS.TRAIN = (args.dataset_name,)
    cfg.DATASETS.TEST = (args.dataset_name,)
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-InstanceSegmentation/{args.model}")
    cfg.OUTPUT_DIR = args.output_dir
    cfg.TEST.EVAL_PERIOD = args.eval_period

    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [args.anchor_size]

    cfg.SOLVER.CHECKPOINT_PERIOD = 50000

    cfg.VIS_PERIOD = args.eval_period

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Save the configuration to a file.
    config_file_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with open(config_file_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(cfg.dump())
    
    trainer = TrainerWithBestAP(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

if __name__ == "__main__":
    main()
