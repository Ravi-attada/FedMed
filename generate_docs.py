import os

data = {
    'chaitanya': [
        ('chore(model): set up PyTorch MONAI environment and explore BraTS dataset structure',
         '* Downloaded and extracted the BraTS 2021 dataset (~13.4 GB).\n* Explored the NIfTI files to understand the 4 MRI modalities (T1, T1ce, T2, FLAIR).\n* Identified the 3 segmentation targets (NCR, ED, ET).\n* Set up the Python virtual environment with PyTorch 2.1 and MONAI 1.3.0.',
         'The dataset is massive; we need to ensure the data loader is optimized to prevent RAM bottlenecks.'),
        ('feat(model): implement 3D U-Net encoder downsampling path with skip connections',
         '* Designed the 3D U-Net encoder architecture.\n* Configured convolutional layers with Instance Normalization and LeakyReLU activations.\n* Verified tensor shapes during downsampling.',
         'Decided to use InstanceNorm instead of BatchNorm since our batch sizes will be small (often 1 or 2) due to 3D memory constraints.'),
        ('feat(model): complete 3D U-Net decoder path and full forward pass sanity check',
         '* Implemented the decoder path with transposed convolutions.\n* Added skip connections to retain high-resolution spatial features from the encoder.\n* Wrote a quick sanity check script to pass a dummy tensor `(1, 4, 128, 128, 64)` through the model.',
         'Model compiles and forward pass works. Parameter count is around 4.8M.'),
        ('feat(train): implement centralized training loop with DiceCE loss and Adam optimizer',
         '* Built the `train_centralized.py` script to establish a baseline.\n* Integrated `DiceCELoss` (combining Dice Loss and Cross-Entropy) from MONAI.\n* Set up the Adam optimizer with `CosineAnnealingLR` scheduling.',
         'Balancing the Lambda weights for Dice vs CrossEntropy. Currently set to 0.5 each.'),
        ('feat(train): add mixed precision FP16 training and run 10-epoch baseline experiment',
         '* Upgraded training loop to use PyTorch AMP (Automatic Mixed Precision).\n* Added `torch.cuda.amp.autocast` to drastically reduce GPU memory usage.\n* Ran a 10-epoch baseline training test to verify loss convergence.',
         'AMP reduced VRAM usage by almost 40%, allowing us to increase the SlidingWindow inferer batch size.'),
        ('fix(model): fix ConvTranspose3D output padding issue and verify on full BraTS volume',
         '* Debugged a dimension mismatch issue in the `ConvTranspose3D` layers.\n* Adjusted the output padding to ensure the decoder output exactly matches the `(240, 240, 155)` original BraTS resolution.\n* Validated on a full-sized sample.',
         'Spatial dimension mismatch was causing concatenation errors in the skip connections. Fixed now.'),
        ('docs(model): add full docstrings type hints and architecture comments to unet3d.py',
         '* Cleaned up the `unet3d.py` code.\n* Added detailed Python docstrings and type hints.\n* Prepared the final pull request for the Model architecture.',
         'Week 1 tasks complete. Ready for integration with the FL server.')
    ],
    'ranjith': [
        ('chore(data): explore BraTS 2021 NIfTI structure and set up MONAI environment',
         '* Explored the NIfTI (`.nii.gz`) file formats from the BraTS dataset.\n* Verified that `nibabel` can successfully load the 3D volumes and labels.\n* Initialized the MONAI transforms sandbox.',
         'Header orientations in NIfTI files vary; we will need an Orientationd transform in the pipeline.'),
        ('feat(data): implement 10-step MONAI transform pipeline with intensity normalization',
         '* Wrote the `preprocess.py` MONAI transform pipeline.\n* Added `LoadImaged`, `EnsureChannelFirstd`, and `NormalizeIntensityd`.\n* Implemented `RandSpatialCropd` and `RandFlipd` for training data augmentation.',
         'Intensity normalization is crucial for MRI. Using Z-score normalization for T1/T2 modalities.'),
        ('feat(data): implement BraTSDataset with CacheDataset and hospital partition support',
         '* Implemented `BraTSDataset` using MONAI CacheDataset to speed up data loading.\n* Wrote a hospital partitioning script to simulate data distribution across 3 different hospital clients.',
         'Caching the first few transforms saves massive I/O overhead during training.'),
        ('feat(eval): implement Dice score and HD95 metrics with BraTS sub-region evaluation',
         '* Implemented the evaluation metrics (`DiceMetric` and `HausdorffDistanceMetric`).\n* Mapped the raw labels (1, 2, 4) to the standard BraTS sub-regions: Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET).',
         'HD95 computation is slow on CPU; ensuring it only runs during validation, not every training step.'),
        ('feat(eval): run evaluation on trained model and generate Week 1 baseline metrics report',
         '* Hooked the metrics into Chaitanya centralized training loop.\n* Generated a baseline metrics report showing Dice scores for WT, TC, and ET after the 10-epoch trial.',
         'Baseline WT Dice is looking good (~0.85), but ET needs more epochs to converge.'),
        ('refactor(data): improve hospital data partition for non-IID simulation and add stats',
         '* Refactored the data partitioning logic to allow configurable Dirichlet distributions.\n* This ensures the Federated Learning simulation can test highly skewed, non-IID client data.\n* Added scripts to visualize data distribution.',
         'Non-IID partitioning will really test the FedProx server implementation next week.'),
        ('docs(data): finalize all docstrings and submit Week 1 PR for data and eval modules',
         '* Documented the preprocessing pipeline and evaluation classes.\n* Added usage examples to the docstrings.\n* Submitted the Week 1 pull request for the Data and Evaluation modules.',
         'Data pipeline is fully operational.')
    ],
    'kushi': [
        ('chore(server): install Flower framework and run Hello World FL example',
         '* Investigated the Flower (`flwr`) framework.\n* Installed dependencies and set up a basic `server.py`.\n* Ran the standard "Hello World" federated averaging example to understand gRPC communication.',
         'Flower is very lightweight. The `start_server` block handles the gRPC threads automatically.'),
        ('feat(server): implement Flower FL server with FedAvg strategy on port 8080',
         '* Implemented the actual FedMed server using Flower FedAvg strategy.\n* Configured the server to listen on port `8080`.\n* Set `min_fit_clients=3` to wait for all hospital nodes before starting a round.',
         'Need to handle cases where a client drops mid-round.'),
        ('feat(server): upgrade to FedProx strategy with proximal_mu=0.1 for non-IID data',
         '* Upgraded the aggregation strategy from `FedAvg` to `FedProx`.\n* Set the proximal term `mu=0.1` to handle the statistical heterogeneity (non-IID data) of the different hospital datasets.',
         'FedProx prevents local models from drifting too far from the global model.'),
        ('feat(server): add server-side evaluate_fn and verify gRPC client connection',
         '* Added a server-side `evaluate_fn`.\n* This tests the global aggregated model against a centralized validation set after every FL round.\n* Verified client ping/pong connections.',
         'Server-side evaluation is faster than aggregating client-side validation scores.'),
        ('feat(server): complete first 3-hospital FL round end-to-end and document gRPC interface',
         '* Ran the first successful 3-client Federated Learning round.\n* Verified that the global model weights were correctly aggregated.\n* Verified weights were distributed back to the hospital nodes for round 2.',
         'Successfully completed 3 rounds of FL in local simulation.'),
        ('refactor(server): add per-round CSV logging and graceful shutdown handler',
         '* Implemented CSV logging for the server to track loss and Dice scores across rounds.\n* Added a `SIGINT` signal handler for graceful shutdown when the training completes or is aborted.',
         'Logs are saving properly to `logs/server_metrics.csv`.'),
        ('docs(server): finalize server documentation and submit Week 1 PR for FL server module',
         '* Wrote the `README.md` documentation for the FL server.\n* Documented the gRPC ports, SSL configurations (future), and strategy parameters.\n* Submitted the Week 1 PR.',
         'Server is ready for Dockerization.')
    ],
    'vasusree': [
        ('chore(client): study NumPyClient interface and plan hospital node setup',
         '* Studied Flower NumPyClient abstract base class.\n* Outlined the architecture for how the hospital nodes will load their local datasets.\n* Planned the connection layer to the central server.',
         'PyTorch tensors need to be converted to NumPy arrays before sending over Flower.'),
        ('feat(client): implement HospitalClient get_parameters and set_parameters methods',
         '* Implemented the `get_parameters` and `set_parameters` methods for the `HospitalClient`.\n* These functions convert PyTorch state dicts to NumPy arrays for transmission over the network.',
         'Ensured strict ordering of `state_dict.keys()` so weights do not get mismatched.'),
        ('feat(client): implement fit() local training loop with FedProx proximal term',
         '* Implemented the `fit()` method.\n* Integrated the PyTorch training loop locally so the hospital node can train the model on its private data.\n* Added local epoch iterations before sending updates.',
         'Local training is working, but it hits the GPU memory hard when running multiple clients.'),
        ('feat(client): configure 3 hospital nodes on ports 8081-8083 and write docker-compose',
         '* Scaled the setup to 3 simulated hospital nodes.\n* Wrote a `docker-compose.yml` file to spin up `hospital_a`, `hospital_b`, and `hospital_c`.\n* Mapped ports to 8081, 8082, and 8083.',
         'Docker makes spinning up the simulated silo network much easier.'),
        ('feat(client): complete full 3-node FL round and verify evaluate returns correct Dice',
         '* Tested a full Federated round from the client side.\n* Verified that the `evaluate()` method correctly computes local validation loss.\n* Ensured metrics are returned to the server successfully.',
         'Clients are successfully sending their dataset sizes to allow the server to do weighted averaging.'),
        ('refactor(client): add retry logic on connection failure and GPU memory logging',
         '* Added robust error handling and retry logic to the gRPC connection.\n* If the central server is slow to start, clients will retry every 5 seconds.\n* Added GPU memory logging for the client instances.',
         'Retry logic prevents Docker compose from failing if the server container takes time to boot.'),
        ('docs(client): finalize hospital node docs and submit Week 1 PR for client Docker modules',
         '* Finalized the `Dockerfile` and container orchestration scripts for the hospital nodes.\n* Added comments explaining the environment variables.\n* Submitted the Week 1 pull request.',
         'Client nodes are fully containerized and ready for Week 2.')
    ],
    'ravi': [
        ('chore(repo): initialize FedMed repository with folder structure and gitignore',
         '* Initialized the GitHub repository.\n* Created the standard project structure (`model/`, `data/`, `server/`, `client/`).\n* Wrote a comprehensive `.gitignore` to prevent the 13GB BraTS dataset from being uploaded.',
         'Git structure is set up. Need to enforce branch protections next.'),
        ('feat(utils): implement centralized logging module with file and console output',
         '* Built the `utils/logger.py` module to standardize logging across the server and clients.\n* Configured dual-output logging (console + file).\n* Added formatted timestamps and module-level tags.',
         'Logging to both stdout and a rolling file makes debugging the FL network much easier.'),
        ('test(integration): run PPML pipeline end-to-end and report bug to Ranjith',
         '* Wrote an integration test script to verify that the PyTorch model, MONAI data loader, and evaluation metrics all work together.\n* Ran a single pass end-to-end.',
         'Found a data shape bug (channel dimension was missing). Reported to Ranjith for fixing.'),
        ('test(integration): verify FL server and hospital nodes gRPC handshake works',
         '* Tested the gRPC network layer.\n* Verified that the Flower server and the 3 Dockerized hospital clients can handshake.\n* Exchanged weights without serialization errors.',
         'Network layer is completely stable.'),
        ('feat(demo): complete week1_demo.py and run full PPML plus FL integration test',
         '* Created `week1_demo.py`.\n* This script spins up the server and 3 local threads for the clients automatically.\n* Allows the team to run a full Federated Learning simulation with a single command.',
         'The demo script is perfect for showcasing our progress without needing Docker.'),
        ('test(integration): run CI pipeline locally and verify all modules pass flake8 linting',
         '* Set up the GitHub Actions CI pipeline (`ci.yml`).\n* Configured `flake8` for linting and added project structure validation checks.\n* Fixed all PEP-8 formatting errors across the codebase.',
         'CI pipeline is green! (Had to fix a few indentation issues in metrics.py).'),
        ('docs(integration): run final team demo merge all PRs into dev Week 1 complete',
         '* Conducted the final Week 1 integration review.\n* Resolved merge conflicts across all branches.\n* Merged `unet-architecture`, `data-preprocessing`, `fl-server`, and `hospital-nodes` into `main`.',
         'Week 1 is a massive success. The core FedMed engine is built and tested.')
    ]
}

name_map = {
    'chaitanya': 'Chaitanya',
    'ranjith': 'Ranjith Kumar',
    'kushi': 'Kushi',
    'vasusree': 'Vasu Sree',
    'ravi': 'Ravi'
}

for key, days in data.items():
    user_name = name_map[key]
    os.makedirs(f'progress/{key}', exist_ok=True)
    for i, (msg, tasks, notes) in enumerate(days):
        day = i + 1
        filepath = f'progress/{key}/day{day}.md'
        content = f"""# {user_name} Day {day} - {msg}

## Tasks Completed
{tasks}

## Notes & Challenges
* {notes}
"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
