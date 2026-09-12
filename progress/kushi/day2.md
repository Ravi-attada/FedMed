# Kushi Day 2 - feat(server): implement Flower FL server with FedAvg strategy on port 8080

## Tasks Completed
* Implemented the actual FedMed server using Flower FedAvg strategy.
* Configured the server to listen on port `8080`.
* Set `min_fit_clients=3` to wait for all hospital nodes before starting a round.

## Notes & Challenges
* Need to handle cases where a client drops mid-round.
