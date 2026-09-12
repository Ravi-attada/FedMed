"""
Member 1 - Chaitanya (ML Lead)
Task: Centralized Training Script - Baseline on BraTS 2021
"""
import os
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.losses import DiceCELoss
from monai.inferers import SlidingWindowInferer
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.unet3d import build_model
from data.dataset import get_dataloaders
from eval.metrics import compute_dice_score
from utils.logger import get_logger

logger = get_logger('CentralizedTrainer')


class CentralizedTrainer:
    """
    Centralized training pipeline for 3D U-Net on BraTS 2021.
    Establishes baseline accuracy before federated training.
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build_model(str(self.device))
        self.criterion = DiceCELoss(to_onehot_y=True, softmax=True, lambda_dice=0.5, lambda_ce=0.5)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.get('lr', 1e-4), weight_decay=1e-5)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.get('epochs', 100), eta_min=1e-6)
        self.inferer = SlidingWindowInferer(roi_size=(128, 128, 64), sw_batch_size=2, overlap=0.25)
        self.train_loader, self.val_loader = get_dataloaders(
            data_dir=config.get('data_dir', 'data/raw/brats2021'),
            batch_size=config.get('batch_size', 1),
        )
        self.best_dice = 0.0
        os.makedirs('logs', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]'):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(self.train_loader)
        logger.info(f'Epoch {epoch} | Train Loss: {avg_loss:.4f}')
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        dice_scores = []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]'):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.inferer(images, self.model)
                dice_scores.append(compute_dice_score(outputs, labels))
        mean_dice = sum(dice_scores) / len(dice_scores)
        logger.info(f'Epoch {epoch} | Val Dice: {mean_dice:.4f}')
        return mean_dice

    def run(self):
        epochs = self.config.get('epochs', 100)
        logger.info(f'Starting centralized training for {epochs} epochs on {self.device}')
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_dice = self.validate(epoch)
            self.scheduler.step()
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                torch.save(self.model.state_dict(), f'checkpoints/best_model_epoch{epoch}.pth')
                logger.info(f'New best Dice: {val_dice:.4f}')
            with open('logs/training_log.txt', 'a') as f:
                f.write(f'Epoch {epoch} | Loss: {train_loss:.4f} | Dice: {val_dice:.4f}\n')
        logger.info(f'Training complete. Best Dice: {self.best_dice:.4f}')


if __name__ == '__main__':
    config = {'data_dir': 'data/raw/brats2021', 'epochs': 100, 'batch_size': 1, 'lr': 1e-4}
    CentralizedTrainer(config).run()
