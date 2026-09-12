# FedMed: Privacy-Preserving Federated Learning for Medical Imaging 🧠🏥

Welcome to **FedMed**! This repository contains the source code for a Cross-Silo Federated Learning engine designed specifically for training 3D medical imaging models (Brain Tumor Segmentation) without ever centralizing patient data.

## 🚀 Week 1 Milestone Completed
During Week 1, our team successfully built the core foundational infrastructure for FedMed. We went from an empty repository to a fully functional Federated Learning simulation using the **BraTS 2021 dataset**.

### 🌟 Key Achievements (Week 1)
* **🧠 3D U-Net Model (Chaitanya):** Built a high-performance 3D U-Net using PyTorch and MONAI, equipped with AMP (Automatic Mixed Precision) and Instance Normalization for 3D MRI volumes.
* **📊 Data Pipeline (Ranjith Kumar):** Implemented a robust 10-step MONAI transform pipeline for NIfTI images, along with configurable non-IID partitioning to simulate heterogeneous hospital datasets.
* **🌐 Federated Server (Kushi):** Deployed a centralized Flower (`flwr`) server using the **FedProx** strategy to handle statistical heterogeneity across hospital silos.
* **🏥 Hospital Nodes (Vasu Sree):** Dockerized hospital client nodes that securely train locally on their private data partitions and communicate with the central server via gRPC.
* **⚙️ Integration & QA (Ravi):** Set up end-to-end continuous integration (CI) workflows, logging, and an automated full-stack simulation demo script.

## 🏗️ Architecture
The current engine runs on the **Flower (flwr)** framework and simulates a hub-and-spoke topology:
* **Central Aggregation Server:** Listens on port `8080`, aggregating weights using FedProx.
* **Silo Nodes:** 3 independent hospital nodes (simulated on ports `8081`, `8082`, `8083`) that execute local training epochs on their private BraTS data partitions.

## 💻 Tech Stack
* **Deep Learning:** PyTorch, PyTorch Lightning
* **Medical Imaging:** MONAI, NiBabel, SimpleITK
* **Federated Learning:** Flower (`flwr`)
* **Infrastructure:** Docker, gRPC, GitHub Actions

## 👨‍💻 Team Members
* **Chaitanya** - ML Lead (Model Architecture)
* **Ranjith Kumar** - ML Engineer (Data Preprocessing & Evaluation)
* **Kushi** - Distributed Systems Engineer (FL Server)
* **Vasu Sree** - Systems Engineer (Hospital Clients)
* **Ravi** - DevOps & Integration (QA & CI/CD)

---
*Progress logs for each team member's daily tasks can be found in the `progress/` directory.*