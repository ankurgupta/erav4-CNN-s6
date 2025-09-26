import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Model 1
# -------------------------
# Large model (>100K parameters) with correct architectural skeleton.
# Design:
# - Follows the skeleton: Conv → Transition → Conv → Output
# Results:
# - Total parameters: 134,996
# - Max Training Accuracy: 99.71%
# - Max Test Accuracy: 98.98%
# Observations:
# - Model is clearly overfitting (high train vs test gap).
# - MaxPooling applied when RF=7, but RF=5 would be preferable.

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=28x28, output_size=26x26, rf_in=1, rf_out=3, jin=1, jout=1

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=26x26, output_size=24x24, rf_in=3, rf_out=5, jin=1, jout=1

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=24x24, output_size=22x22, rf_in=5, rf_out=7, jin=1, jout=1

        # TRANSITION BLOCK 1
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
        ) # input_size=22x22, output_size=22x22, rf_in=7, rf_out=7, jin=1, jout=1

        self.pool1 = nn.MaxPool2d(2, 2) 
        # input_size=22x22, output_size=11x11, rf_in=7, rf_out=8, jin=1, jout=2

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=11x11, output_size=9x9, rf_in=8, rf_out=12, jin=2, jout=2

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=9x9, output_size=7x7, rf_in=12, rf_out=16, jin=2, jout=2

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=7x7, output_size=5x5, rf_in=16, rf_out=20, jin=2, jout=2

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
        ) # input_size=5x5, output_size=5x5, rf_in=20, rf_out=24, jin=2, jout=2

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(5, 5), padding=0, bias=False),
        ) # input_size=5x5, output_size=1x1, rf_in=24, rf_out=32, jin=2, jout=2

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool1(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
