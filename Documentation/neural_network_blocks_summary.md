# Neural Network Building Blocks

## Overview
This module defines the fundamental building blocks used to construct the UNet architecture. These blocks handle the creation of convolutional layers, downsampling, and upsampling operations.

## Key Components

### `conv_block(in_channels, out_channels, dropout=0.1)`
- **Purpose**: Creates a convolutional block with batch normalization and ReLU activation
- **Parameters**:
  - `in_channels`: Number of input channels
  - `out_channels`: Number of output channels
  - `dropout`: Dropout rate for regularization (default: 0.1)
- **Components**:
  - 3x3 Convolutional layer with padding
  - Batch normalization
  - ReLU activation
  - Dropout for preventing overfitting
- **Returns**: Sequential container of layers

### `down_block(in_channels, out_channels, dropout=0.1)`
- **Purpose**: Creates a downsampling block for the encoder part of UNet
- **Parameters**:
  - `in_channels`: Number of input channels
  - `out_channels`: Number of output channels
  - `dropout`: Dropout rate (default: 0.1)
- **Process**:
  - Applies two convolutional blocks
  - Performs max pooling to reduce spatial dimensions
- **Returns**: Sequential container of layers
- **Note**: Reduces spatial dimensions by half while increasing feature depth

### `up_block(in_channels, out_channels, dropout=0.1)`
- **Purpose**: Creates an upsampling block for the decoder part of UNet
- **Parameters**:
  - `in_channels`: Number of input channels
  - `out_channels`: Number of output channels
  - `dropout`: Dropout rate (default: 0.1)
- **Process**:
  - Upsamples the input by a factor of 2
  - Applies two convolutional blocks
- **Returns**: Sequential container of layers
- **Note**: Increases spatial dimensions while decreasing feature depth

## Usage
These building blocks are combined to create the encoder-decoder structure of the UNet architecture, enabling the network to capture both fine details and broader context in images.