# Vasu Sree Day 2 - feat(client): implement HospitalClient get_parameters and set_parameters methods

## Tasks Completed
* Implemented the `get_parameters` and `set_parameters` methods for the `HospitalClient`.
* These functions convert PyTorch state dicts to NumPy arrays for transmission over the network.

## Notes & Challenges
* Ensured strict ordering of `state_dict.keys()` so weights do not get mismatched.
