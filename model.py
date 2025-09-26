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

# Preserve the skeleton, reduce number of parameters.
# Target:
# - Move max pooling to rf = 5
# - Reduce number of parameters
# - Preserve the skeleton

# Result:
# - total # parameters = 11,068
# - max training accuracy = 99.28
# - max test accuracy = 98.5

# Analysis:
# - Model is still overfitting but the gap is reduced.
# - Max training accuracy is still not above 99%

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

# -------------------------
# Model 3
# -------------------------
# Introduced Global Average Pooling (GAP) to further cut parameters.
# Design:
# - Kept same skeleton but replaced final layers with GAP
# - Reduced parameters below 8K
# Results:
# - Total parameters: 7,468
# - Max Training Accuracy: 99.00%
# - Max Test Accuracy: 98.81%
# Observations:
# - Model generalizes better, overfitting reduced.
# - Parameters are under 8K → good balance of efficiency and accuracy.
class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
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
        ) # input_size=6x6, output_size=6x6, rf_in=18, rf_out=22, jin=2, jout=2

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # input_size=6x6, output_size=1x1, rf_in=22, rf_out=32, jin=2, jout=2

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# -------------------------
# Model 4
# -------------------------
# Added BatchNorm and Dropout for regularization and stability.
# Design:
# - BatchNorm applied after each conv layer
# - Dropout introduced to reduce overfitting
# Results:
# - Total parameters: 7,600
# - Max Training Accuracy: 99.08%
# - Max Test Accuracy: 99.28%
# Observations:
# - Accuracy improved slightly but plateaued near 99.25%
# - Model is starting to underfit.
class Model4(nn.Module):
    def __init__(self):
        super(Model4, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # input_size=28x28, output_size=26x26, rf_in=1, rf_out=3, jin=1, jout=1

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
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
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # input_size=12x12, output_size=10x10, rf_in=6, rf_out=10, jin=2, jout=2
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # input_size=10x10, output_size=8x8, rf_in=10, rf_out=14, jin=2, jout=2
        
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # input_size=8x8, output_size=6x6, rf_in=14, rf_out=18, jin=2, jout=2
        
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            #nn.ReLU(),
            #nn.BatchNorm2d(10),
            #nn.Dropout(dropout_value)
        ) # input_size=6x6, output_size=6x6, rf_in=18, rf_out=22, jin=2, jout=2

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # input_size=6x6, output_size=1x1, rf_in=22, rf_out=32, jin=2, jout=2

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        #x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


# -------------------------
# Model 5
# -------------------------
# Modified dataset with image rotations for data augmentation.
# Design:
# - Same architecture as Model4
# - Added rotations to input pipeline
# Results:
# - Total parameters: 7,600
# - Max Training Accuracy: 98.90%
# - Max Test Accuracy: 99.31%
# Observations:
# - Training harder → training accuracy decreased
# - Test accuracy slightly improved but capped at ~99.3%
# - Model underfitting due to added data complexity.

class Model5(nn.Module):
    def __init__(self):
        super(Model5, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            #nn.ReLU(),
            #nn.BatchNorm2d(10),
            #nn.Dropout(dropout_value)
        ) # output_size = 6

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        #x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


# -------------------------
# Model 6
# -------------------------
# Removed Dropout to test its necessity in CNNs for MNIST.
# Design:
# - Same skeleton, but Dropout disabled
# Results:
# - Total parameters: 7,600
# - Max Training Accuracy: 99.38%
# - Max Test Accuracy: 99.38%
# Observations:
# - Removing Dropout improved both train/test accuracy.
# - Accuracy fluctuates around 99.3 after 10th epoch.
# - Suggestion: reduce learning rate after 10th epoch for stability.
class Model6(nn.Module):
    def __init__(self):
        dropout_value = 0
        super(Model6, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            #nn.ReLU(),
            #nn.BatchNorm2d(10),
            #nn.Dropout(dropout_value)
        ) # output_size = 6

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        #x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# -------------------------
# Model 7
# -------------------------
# Final optimization: added Learning Rate Scheduler.
# Design:
# - Same skeleton as Model6
# - LR reduced dynamically after ~10 epochs
# Results:
# - Total parameters: 7,264
# - Max Training Accuracy: 99.44%
# - Max Test Accuracy: 99.52%
# - Accuracy consistently above 99.4% after epoch 11
# Observations:
# - LR scheduling effectively pushed accuracy beyond previous plateau.
# - Achieved best performance while reducing parameter count further.
class Model7(nn.Module):
    def __init__(self):
        dropout_value = 0
        super(Model7, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        ) # output_size = 6
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            #nn.ReLU(),
            #nn.BatchNorm2d(10),
            #nn.Dropout(dropout_value)
        ) # output_size = 6

        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        #x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
