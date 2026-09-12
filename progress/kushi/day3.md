# Kushi Day 3 - feat(server): upgrade to FedProx strategy with proximal_mu=0.1 for non-IID data

## Tasks Completed
* Upgraded the aggregation strategy from `FedAvg` to `FedProx`.
* Set the proximal term `mu=0.1` to handle the statistical heterogeneity (non-IID data) of the different hospital datasets.

## Notes & Challenges
* FedProx prevents local models from drifting too far from the global model.
