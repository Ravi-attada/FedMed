"""
Member 2 - Ranjith (ML Engineer)
Task: Segmentation Metrics - Dice Score and Hausdorff Distance
"""
import torch
from typing import Dict
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction

REGION_NAMES = {0: 'Background', 1: 'NCR', 2: 'ED', 3: 'ET'}


class SegmentationEvaluator:
    """Computes Dice Score and HD95 for BraTS 2021 segmentation."""
    def __init__(self, num_classes=4, device='cpu'):
        self.num_classes = num_classes
        self.dice_metric = DiceMetric(include_background=False,
                                      reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
        self.hd95_metric = HausdorffDistanceMetric(include_background=False,
                                                   percentile=95, reduction=MetricReduction.MEAN_BATCH)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
        self.post_label = AsDiscrete(to_onehot=num_classes)

    def update(self, outputs, labels):
        op = torch.stack([self.post_pred(o) for o in outputs])
        lp = torch.stack([self.post_label(lbl) for lbl in labels])
        self.dice_metric(y_pred=op, y=lp)
        self.hd95_metric(y_pred=op, y=lp)

    def compute(self):
        dice, _ = self.dice_metric.aggregate()
        hd95, _ = self.hd95_metric.aggregate()
        return {
            'Dice_WT': dice[[0, 1, 2]].mean().item(),
            'Dice_TC': dice[[0, 2]].mean().item(),
            'Dice_ET': dice[2].item(),
            'Dice_Mean': dice.mean().item(),
            'HD95_Mean': hd95.mean().item(),
        }

    def reset(self):
        self.dice_metric.reset()
        self.hd95_metric.reset()


def compute_dice_score(outputs, labels):
    """Quick single-batch Dice computation for FL training rounds."""
    post_pred = AsDiscrete(argmax=True, to_onehot=4)
    post_label = AsDiscrete(to_onehot=4)
    metric = DiceMetric(include_background=False, reduction=MetricReduction.MEAN)
    metric(y_pred=torch.stack([post_pred(o) for o in outputs]),
           y=torch.stack([post_label(lbl) for lbl in labels]))
    result, _ = metric.aggregate()
    return result.item()


def print_baseline_report(metrics, epoch):
    print('=' * 55)
    print(f'  FedMed Baseline Report - Epoch {epoch}')
    print('=' * 55)
    print(f'  Dice (Whole Tumor):      {metrics.get("Dice_WT", 0):.4f}')
    print(f'  Dice (Tumor Core):       {metrics.get("Dice_TC", 0):.4f}')
    print(f'  Dice (Enhancing Tumor):  {metrics.get("Dice_ET", 0):.4f}')
    print(f'  Dice (Mean):             {metrics.get("Dice_Mean", 0):.4f}')
    print(f'  HD95 (Mean):             {metrics.get("HD95_Mean", 0):.2f} mm')
    print('=' * 55)
