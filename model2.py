import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Model 2
# -------------------------
# Optimized model with fewer parameters while preserving the skeleton.
# Design:
# - Moved MaxPooling to RF=5
# - Reduced parameter count significantly
# - Retained Conv → Transition → Conv → Output structure
# Results:
# - Total parameters: 11,068
# - Max Training Accuracy: 99.28%
# - Max Test Accuracy: 98.50%
# Observations:
# - Still some overfitting, though reduced compared to Model1.
# - Training accuracy does not exceed 99%.
class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=28x28, output_size=26x26, rf_in=1, rf_out=3, jin=1, jout=1

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=26x26, output_size=24x24, rf_in=3, rf_out=5, jin=1, jout=1

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # input_size=24x24, output_size=24x24, rf_in=5, rf_out=5, jin=1, jout=1
        
        self.pool1 = nn.MaxPool2d(2, 2) 
        # input_size=24x24, output_size=12x12, rf_in=5, rf_out=6, jin=1, jout=2

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=12x12, output_size=10x10, rf_in=6, rf_out=10, jin=2, jout=2
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=10x10, output_size=8x8, rf_in=10, rf_out=14, jin=2, jout=2
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # input_size=8x8, output_size=6x6, rf_in=14, rf_out=18, jin=2, jout=2
        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
        ) # input_size=6x6, output_size=6x6, rf_in=18, rf_out=22, jin=2, jout=2

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(6, 6), padding=0, bias=False),
        ) # input_size=6x6, output_size=1x1, rf_in=22, rf_out=32, jin=2, jout=2

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
