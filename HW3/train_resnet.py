import os
import argparse
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer
from detectron2.engine.hooks import BestCheckpointer
from detectron2.config import LazyConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Train Detectron2 with best AP checkpointing")

    parser.add_argument("--max_iter", default= 50000, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_classes", default=4, type=int)
    parser.add_argument("--eval_period", default=1000, type=int, help="Evaluation period in iterations")

    parser.add_argument("--model", default="mask_rcnn_X_101_32x8d_FPN_3x.yaml", type=str)
    parser.add_argument("--output_dir", default="checkpoints/models/X101_FPN3x", type=str)
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
            BestCheckpointer(self.cfg.TEST.EVAL_PERIOD, self.checkpointer, "segm/AP")
        )
        return hooks

def main():
    args = parse_args()

    # Register dataset
    register_coco_instances("my_dataset_train", {}, "../data/full.json", "../data/train")
    MetadataCatalog.get("my_dataset_train")
    DatasetCatalog.get("my_dataset_train")

    # Build config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-InstanceSegmentation/{args.model}"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-InstanceSegmentation/{args.model}")

    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_train",)
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.OUTPUT_DIR = args.output_dir
    cfg.TEST.EVAL_PERIOD = args.eval_period

    ##### Setting for Nuclei #####
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4], [8], [16], [32], [64]]
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.05
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 3000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST  = 1500
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    #############################

    cfg.SOLVER.CHECKPOINT_PERIOD = 50000

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Save the configuration to a file.
    config_file_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
    with open(config_file_path, "w", encoding="utf-8") as file_obj:
        file_obj.write(cfg.dump())
    
    trainer = TrainerWithBestAP(cfg)
    trainer.resume_or_load(resume=False)

    # Print total and trainable parameters
    model = trainer.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel parameter summary:")
    print(f"  Total parameters:     {int(total_params / 1e6)}M")
    print(f"  Trainable parameters: {int(trainable_params / 1e6)}M\n")

    trainer.train()

if __name__ == "__main__":
    main()
