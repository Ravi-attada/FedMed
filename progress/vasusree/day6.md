# Vasu Sree Day 6 - refactor(client): add retry logic on connection failure and GPU memory logging

## Tasks Completed
* Added robust error handling and retry logic to the gRPC connection.
* If the central server is slow to start, clients will retry every 5 seconds.
* Added GPU memory logging for the client instances.

## Notes & Challenges
* Retry logic prevents Docker compose from failing if the server container takes time to boot.
