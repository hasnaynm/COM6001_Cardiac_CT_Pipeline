# model.py
# defines the UNetDeep architecture used for cardiac chamber segmentation
# 2.5D input (5 slices stacked as channels), 5 class output (background + LV, RV, LA, RA)

import torch
import torch.nn as nn
import torch.nn.functional as F




class DoubleConv(nn.Module):
    # reusable building block - two conv layers back to back
    # each conv followed by batch norm and ReLU activation
    # optional dropout at the end (only used in bottleneck)

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # 3x3 conv, padding=1 keeps spatial size the same
            nn.BatchNorm2d(out_channels),  # normalises activations to stabilise training
            nn.ReLU(inplace=True),         # activation function, inplace saves memory
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # second conv
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))  # randomly zero out channels to reduce overfitting
        self.block = nn.Sequential(*layers)  # chain all layers together

    def forward(self, x):
        return self.block(x)  # pass input through all layers in sequence




class UNetDeep(nn.Module):
    # full depth U-Net for 2.5D cardiac CT segmentation
    # 4 encoder levels (deeper than standard 2-level U-Net)
    # skip connections pass encoder features directly to decoder at each level
    # input:  [B, C, H, W] where C = num_input_slices (5 slices)
    # output: [B, num_classes, H, W] - one score per class per pixel


    def __init__(self, in_channels: int = 5, out_channels: int = 5):
        super().__init__()

        # ENCODER - progressively extracts features while halving spatial size
        # channel sizes double at each level: 32 -> 64 -> 128 -> 256
        self.down1 = DoubleConv(in_channels, 32)   # level 1
        self.pool1 = nn.MaxPool2d(2)               # halves H and W

        self.down2 = DoubleConv(32, 64)            # level 2
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(64, 128)           # level 3
        self.pool3 = nn.MaxPool2d(2)

        self.down4 = DoubleConv(128, 256)          # level 4
        self.pool4 = nn.MaxPool2d(2)


        # BOTTLENECK - deepest part of the network, most abstract features
        # dropout=0.3 here only - regularisation at the most compressed representation
        self.bottleneck = DoubleConv(256, 512, dropout=0.3)

        # DECODER - progressively upsamples back to original resolution
        # ConvTranspose2d doubles spatial size (opposite of MaxPool)
        # after each upsample, concatenate with matching encoder feature map (skip connection)
        # channel count doubles after concat, then DoubleConv brings it back down



        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 512 -> 256, doubles size
        self.conv4 = DoubleConv(512, 256)  # 512 because we concat 256 (up) + 256 (skip)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)  # 128 + 128 skip = 256 in

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)   # 64 + 64 skip = 128 in

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64, 32)    # 32 + 32 skip = 64 in



        # final 1x1 conv maps 32 channels to num_classes (5)
        # no activation here - raw logits go to softmax in the loss function
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)



    def forward(self, x):
        # ENCODER PATH
        # save output of each DoubleConv (before pooling) for skip connections
        d1 = self.down1(x)   # full resolution features
        p1 = self.pool1(d1)  # halved

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        d4 = self.down4(p3)
        p4 = self.pool4(d4)


        # BOTTLENECK
        b = self.bottleneck(p4)




        # DECODER PATH
        # upsample then concatenate with matching encoder features (skip connection)
        # interpolate check handles edge cases where sizes dont match exactly after pooling

        u4 = self.up4(b)
        if u4.shape[-2:] != d4.shape[-2:]:  # if spatial sizes dont match
            u4 = F.interpolate(u4, size=d4.shape[-2:], mode='bilinear', align_corners=False)
        u4 = torch.cat([u4, d4], dim=1)  # concat along channel dimension
        u4 = self.conv4(u4)

        u3 = self.up3(u4)
        if u3.shape[-2:] != d3.shape[-2:]:
            u3 = F.interpolate(u3, size=d3.shape[-2:], mode='bilinear', align_corners=False)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        if u2.shape[-2:] != d2.shape[-2:]:
            u2 = F.interpolate(u2, size=d2.shape[-2:], mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        if u1.shape[-2:] != d1.shape[-2:]:
            u1 = F.interpolate(u1, size=d1.shape[-2:], mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)

        # 1x1 conv to get per-class scores at each pixel
        return self.final(u1)
