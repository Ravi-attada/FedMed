# Kushi Day 6 - refactor(server): add per-round CSV logging and graceful shutdown handler

## Tasks Completed
* Implemented CSV logging for the server to track loss and Dice scores across rounds.
* Added a `SIGINT` signal handler for graceful shutdown when the training completes or is aborted.

## Notes & Challenges
* Logs are saving properly to `logs/server_metrics.csv`.
