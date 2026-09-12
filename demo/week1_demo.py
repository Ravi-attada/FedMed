"""
Member 5 - Ravi (Integration & QA)
Task: Week 1 End-to-End Demo Script
Runs FL server + 3 hospital nodes, completes 3 FL rounds
"""
import os
import sys
import time
import threading
from utils.logger import get_logger

logger = get_logger('Week1Demo')
DATA_DIR = 'data/raw/brats2021'
SERVER_ADDR = 'localhost:8080'
NUM_ROUNDS = 3

BANNER = '''
+=====================================================+
|       FedMed -- Week 1 Demo                        |
|  Cross-Silo Federated Learning Engine              |
|  3 Hospital Nodes | 3 FL Rounds | BraTS 2021       |
+=====================================================+
'''


def run_server():
    from server.fl_server import start_server
    start_server(server_address=SERVER_ADDR, num_rounds=NUM_ROUNDS)


def run_hospital_node(hospital_id):
    time.sleep(3)
    from client.fl_client import start_hospital_node
    start_hospital_node(hospital_id=hospital_id, server_address=SERVER_ADDR, data_dir=DATA_DIR)


def main():
    print(BANNER)
    logger.info(f'Data: {DATA_DIR} | Rounds: {NUM_ROUNDS} | Nodes: 3')
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    time.sleep(2)
    threads = []
    for hid in [1, 2, 3]:
        t = threading.Thread(target=run_hospital_node, args=(hid,), daemon=True)
        threads.append(t)
        t.start()
        time.sleep(0.5)
    for t in threads:
        t.join(timeout=600)
    server_thread.join(timeout=60)
    logger.info('Week 1 Demo Complete! Check logs/ for metrics.')


if __name__ == '__main__':
    main()
