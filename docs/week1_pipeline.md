# FedMed Week 1: Pipeline Architecture

During Week 1, we successfully constructed four independent but interconnected pipelines that form the core of the FedMed engine. 

## 1. Federated Learning (FL) Pipeline
This is the core hub-and-spoke architecture that allows multiple hospitals to train the central model without sharing raw patient data.

```mermaid
graph TD
    Server[Central FL Server<br>Strategy: FedProx<br>Port: 8080]
    
    ClientA[Hospital A Client<br>Port: 8081]
    ClientB[Hospital B Client<br>Port: 8082]
    ClientC[Hospital C Client<br>Port: 8083]

    Server <-->|gRPC / Send Global Weights| ClientA
    Server <-->|gRPC / Send Global Weights| ClientB
    Server <-->|gRPC / Send Global Weights| ClientC
    
    ClientA -.->|Train Locally| DataA[(Local BraTS Partition A)]
    ClientB -.->|Train Locally| DataB[(Local BraTS Partition B)]
    ClientC -.->|Train Locally| DataC[(Local BraTS Partition C)]
```

## 2. Data Preprocessing Pipeline (MONAI)
The sequence of transformations applied to the raw BraTS 2021 NIfTI files before they are fed into the 3D U-Net.

```mermaid
graph LR
    Raw[(Raw .nii.gz)] --> Load[LoadImaged]
    Load --> Channel[EnsureChannelFirstd]
    Channel --> Orient[Orientationd RAS]
    Orient --> Norm[NormalizeIntensityd]
    Norm --> Crop[RandSpatialCropd 128x128x64]
    Crop --> Tensor[ToTensord]
    Tensor --> Model((3D U-Net))
```

## 3. Training & Evaluation Pipeline
The internal machine learning pipeline executed locally on each hospital node during their `fit()` loop.

```mermaid
graph TD
    Input[3D Tensor 128x128x64] --> Unet[3D U-Net Encoder-Decoder]
    Unet --> Output[4-Channel Output Prediction]
    Output --> Loss[DiceCELoss Computation]
    Loss --> Backprop[Adam Optimizer + AMP]
    Backprop --> Weights[Update Local Weights]
    
    Output --> Eval[Compute Metrics]
    Eval --> Dice[Dice Score WT, TC, ET]
    Eval --> HD95[Hausdorff Distance 95]
```

## 4. Continuous Integration (CI/CD) Pipeline
The automated quality assurance pipeline triggered every time a team member pushes to GitHub.

```mermaid
graph LR
    Push[Git Push / PR] --> GitHub[GitHub Actions Trigger]
    GitHub --> Lint[Flake8 Syntax & Style Check]
    GitHub --> Struct[Project Structure Validation]
    Lint --> Pass{All Checks Pass?}
    Struct --> Pass
    Pass -->|Yes| Merge[Allow Merge to Main]
    Pass -->|No| Block[Block Merge / Alert Developer]
```
