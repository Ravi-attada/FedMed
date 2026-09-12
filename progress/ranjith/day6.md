# Ranjith Kumar Day 6 - refactor(data): improve hospital data partition for non-IID simulation and add stats

## Tasks Completed
* Refactored the data partitioning logic to allow configurable Dirichlet distributions.
* This ensures the Federated Learning simulation can test highly skewed, non-IID client data.
* Added scripts to visualize data distribution.

## Notes & Challenges
* Non-IID partitioning will really test the FedProx server implementation next week.
