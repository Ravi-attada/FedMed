"""
Member 1 - Chaitanya (ML Lead)
Task: 3D U-Net Architecture for Brain Tumor Segmentation
"""
import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.networks.layers import Norm


class FedMed3DUNet(nn.Module):
    """
    3D U-Net for brain tumor segmentation (BraTS 2021).
    Input:  (B, 4, 240, 240, 155) - 4 MRI modalities: T1, T1ce, T2, FLAIR
    Output: (B, 4, 240, 240, 155) - 4 classes: Background, NCR, ED, ET
    """
    def __init__(self, in_channels=4, out_channels=4, spatial_dims=3, init_filters=32):
        super(FedMed3DUNet, self).__init__()
        self.unet = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(init_filters, init_filters*2, init_filters*4, init_filters*8, init_filters*16),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.INSTANCE,
            dropout=0.1,
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.unet(x)

    def predict(self, x):
        return self.softmax(self.forward(x))

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(device='cuda'):
    model = FedMed3DUNet(in_channels=4, out_channels=4, spatial_dims=3, init_filters=32)
    model = model.to(device)
    print(f'[FedMed] 3D U-Net | Parameters: {model.get_num_parameters():,} | Device: {device}')
    return model


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(device)
    dummy = torch.randn(1, 4, 128, 128, 64).to(device)
    out = model(dummy)
    print(f'Input: {dummy.shape} -> Output: {out.shape}')
    assert out.shape == dummy.shape
    print('[FedMed] Model sanity check passed.')
