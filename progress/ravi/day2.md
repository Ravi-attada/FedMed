# Ravi Day 2 - feat(utils): implement centralized logging module with file and console output

## Tasks Completed
* Built the `utils/logger.py` module to standardize logging across the server and clients.
* Configured dual-output logging (console + file).
* Added formatted timestamps and module-level tags.

## Notes & Challenges
* Logging to both stdout and a rolling file makes debugging the FL network much easier.
