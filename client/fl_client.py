"""
Member 4 - Vasu Sree (Backend/DevOps)
Task: Hospital Node FL Client - 3 mock hospitals
"""
import sys
import os
import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple
import flwr as fl
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.unet3d import build_model
from data.dataset import get_dataloaders
from eval.metrics import compute_dice_score
from utils.logger import get_logger

HOSPITAL_NAMES = {1: 'Hospital-A (AIIMS)', 2: 'Hospital-B (Mayo Clinic)', 3: 'Hospital-C (NHS Trust)'}


class HospitalClient(fl.client.NumPyClient):
    """
    FedMed Hospital Node - Flower NumPyClient.
    Trains locally on private MRI data. Raw data NEVER leaves this node.
    """
    def __init__(self, hospital_id, data_dir, device='cpu'):
        self.hospital_id = hospital_id
        self.hospital_name = HOSPITAL_NAMES.get(hospital_id, f'Hospital-{hospital_id}')
        self.device = torch.device(device)
        self.logger = get_logger(f'Node-{hospital_id}')
        self.model = build_model(device)
        self.train_loader, self.val_loader = get_dataloaders(
            data_dir=data_dir, batch_size=1, num_workers=2, hospital_id=hospital_id)
        self.logger.info(f'{self.hospital_name} node initialized')

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = OrderedDict({k: torch.tensor(v)
                                  for k, v in zip(self.model.state_dict().keys(), parameters)})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        server_round = config.get('server_round', 0)
        local_epochs = config.get('local_epochs', 3)
        lr = config.get('learning_rate', 1e-4)
        proximal_mu = config.get('proximal_mu', 0.1)
        self.set_parameters(parameters)
        global_params = [p.clone().detach() for p in self.model.parameters()]
        from monai.losses import DiceCELoss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = DiceCELoss(to_onehot_y=True, softmax=True)
        self.model.train()
        total_loss, num_examples = 0.0, 0
        for epoch in range(local_epochs):
            for batch in self.train_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                proximal = sum((lp - gp.to(self.device)).norm(2)**2
                               for lp, gp in zip(self.model.parameters(), global_params))
                loss += (proximal_mu / 2) * proximal
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                num_examples += images.shape[0]
        avg_loss = total_loss / (local_epochs * len(self.train_loader))
        self.logger.info(f'Round {server_round} | {self.hospital_name} | Loss: {avg_loss:.4f}')
        return self.get_parameters(config={}), num_examples, {'train_loss': avg_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        dice_scores, num_examples = [], 0
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                dice_scores.append(compute_dice_score(self.model(images), labels))
                num_examples += images.shape[0]
        mean_dice = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
        self.logger.info(f'{self.hospital_name} | Dice: {mean_dice:.4f}')
        return 1.0 - mean_dice, num_examples, {'dice_score': mean_dice}


def start_hospital_node(hospital_id, server_address='localhost:8080', data_dir='data/raw/brats2021'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    client = HospitalClient(hospital_id=hospital_id, data_dir=data_dir, device=device)
    fl.client.start_numpy_client(server_address=server_address, client=client)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hospital-id', type=int, required=True, choices=[1, 2, 3])
    parser.add_argument('--server', type=str, default='localhost:8080')
    parser.add_argument('--data-dir', type=str, default='data/raw/brats2021')
    args = parser.parse_args()
    start_hospital_node(args.hospital_id, args.server, args.data_dir)
