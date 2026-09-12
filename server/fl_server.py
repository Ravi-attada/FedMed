"""
Member 3 - Kushi (FL Systems Lead)
Task: Flower FL Server with FedProx Strategy
"""
import sys
import os
from typing import List, Tuple, Optional, Dict
from collections import OrderedDict
import flwr as fl
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.unet3d import build_model
from utils.logger import get_logger

logger = get_logger('FedMedServer')


def weighted_average(metrics):
    """Weighted average of Dice scores across hospital nodes."""
    total = sum(n for n, _ in metrics)
    dice = sum(n * m.get('dice_score', 0.0) for n, m in metrics)
    return {'dice_score': dice / total if total > 0 else 0.0}


def get_evaluate_fn(device='cpu'):
    """Server-side global model evaluation function."""
    def evaluate(server_round, parameters, config):
        model = build_model(device)
        state_dict = OrderedDict({k: torch.tensor(v)
                                  for k, v in zip(model.state_dict().keys(), parameters)})
        model.load_state_dict(state_dict, strict=True)
        logger.info(f'[Server] Round {server_round} evaluation complete')
        return 0.0, {'round': server_round}
    return evaluate


def fit_config(server_round):
    """Training config sent to clients each round."""
    return {'server_round': server_round,
            'local_epochs': 3 if server_round <= 10 else 2,
            'batch_size': 1, 'learning_rate': 1e-4}


def start_server(server_address='0.0.0.0:8080', num_rounds=50):
    """Start FedMed FL server with FedProx strategy."""
    logger.info(f'[Server] Starting FedMed on {server_address} | Rounds: {num_rounds}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    global_model = build_model(device)
    initial_params = fl.common.ndarrays_to_parameters(
        [v.cpu().numpy() for _, v in global_model.state_dict().items()]
    )
    strategy = fl.server.strategy.FedProx(
        proximal_mu=0.1,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(device),
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=initial_params,
    )
    history = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    logger.info('FL Training Complete!')
    return history


if __name__ == '__main__':
    start_server(server_address='0.0.0.0:8080', num_rounds=50)
