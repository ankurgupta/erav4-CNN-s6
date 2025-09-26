# MNIST Digit Classification with PyTorch

This repository contains a PyTorch implementation of a CNN model for MNIST digit classification.

## Model Evolution

### Model1: Initial Architecture
- Target: Establish right skeleton (Conv -> Transition -> Conv -> Output)
- Parameters: 134,996
- Results:
  - Max Training Accuracy: 99.71%
  - Max Test Accuracy: 98.98%
- Analysis: 
  - Overfitting due to large parameter count
  - MaxPooling at RF=7 (recommended RF=5)

### Model2: Parameter Reduction
- Target: Move MaxPool to RF=5, reduce parameters
- Parameters: 11,068
- Results:
  - Max Training Accuracy: 99.28%
  - Max Test Accuracy: 98.5%
- Analysis:
  - Reduced overfitting but gap still exists
  - Training accuracy needs improvement

### Model3: GAP Introduction
- Target: Reduce parameters to <8K using Global Average Pooling
- Parameters: 7,468
- Results:
  - Max Training Accuracy: 99.00%
  - Max Test Accuracy: 98.81%
- Analysis:
  - Good balance between train/test accuracy
  - Successfully reduced parameters while maintaining performance

### Model4: Regularization
- Target: Add BatchNorm and Dropout for accuracy
- Parameters: 7,600
- Results:
  - Max Training Accuracy: 99.08%
  - Max Test Accuracy: 99.28%
- Analysis:
  - Accuracy plateaued at 99.25%
  - Signs of underfitting

### Model5: Data Augmentation
- Target: Add rotation augmentation
- Parameters: 7,600
- Results:
  - Max Training Accuracy: 98.90%
  - Max Test Accuracy: 99.31%
- Analysis:
  - Made training harder as expected
  - Still hitting accuracy ceiling at 99.3%

### Model6: Dropout Removal
- Target: Remove dropout to improve accuracy
- Parameters: 7,600
- Results:
  - Max Training Accuracy: 99.38%
  - Max Test Accuracy: 99.38%
- Analysis:
  - Improved performance
  - Accuracy fluctuating after epoch 10

### Model7: Learning Rate Scheduling
- Target: Reduce LR after epoch 10
- Parameters: 7,264
- Results:
  - Max Training Accuracy: 99.44%
  - Max Test Accuracy: 99.52%
  - Consistent >99.4% after reaching peak
- Analysis:
  - Best performing model
  - LR reduction helped stabilize and improve accuracy
  - Further optimized parameter count once 99.4% was reached under 15th epoch

## Project Structure

project_folder/
├── model.py # CNN architecture definition
├── train.py # Training script
└── README.md # This file


The script will:
- Download the MNIST dataset automatically
- Train the model for 20 epochs
- Display progress bars for each epoch
- Show test accuracy after each epoch
- Display a summary table at the end

## Model Architecture

The CNN model (`model.py`) consists of:
- Input Block: 1→8 channels
- Convolution Block 1: 8→16 channels
- Transition Block: 16→10 channels + MaxPool
- Convolution Block 2: Multiple layers with varying channels
- Global Average Pooling
- Output: 10 classes (digits 0-9)

## Training Details

- Optimizer: SGD with momentum (0.9)
- Learning Rate: 0.01 with StepLR scheduler
- Batch Size: 128 (GPU) / 64 (CPU)
- Data Augmentation: Random rotation (-7° to 7°)

## Results

The script will display the following:
- Test accuracy after each epoch
- A summary table at the end with training and test losses, accuracies, and total time taken
