# Kushi Day 4 - feat(server): add server-side evaluate_fn and verify gRPC client connection

## Tasks Completed
* Added a server-side `evaluate_fn`.
* This tests the global aggregated model against a centralized validation set after every FL round.
* Verified client ping/pong connections.

## Notes & Challenges
* Server-side evaluation is faster than aggregating client-side validation scores.
