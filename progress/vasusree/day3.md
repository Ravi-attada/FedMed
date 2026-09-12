# Vasu Sree Day 3 - feat(client): implement fit() local training loop with FedProx proximal term

## Tasks Completed
* Implemented the `fit()` method.
* Integrated the PyTorch training loop locally so the hospital node can train the model on its private data.
* Added local epoch iterations before sending updates.

## Notes & Challenges
* Local training is working, but it hits the GPU memory hard when running multiple clients.
